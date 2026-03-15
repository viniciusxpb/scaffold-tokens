#!/usr/bin/env python3
"""
FF Countdown model inference.

Loads the full model from training (same architecture) and generates text
with forced ff countdown.

Usage:
    python inference.py --words 100
    python inference.py --words 50 --prompt "The president"
    python inference.py --checkpoint checkpoints/model-XXXXX.pt --words 200
"""

import argparse
import glob
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs.*")
warnings.filterwarnings("ignore", message=".*Tensor.item.*")
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
import tiktoken
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

FF_BASE_ID = cfg['tokenizer']['ff_base_id']
FF_MAX = cfg['tokenizer']['ff_max']
EOT_ID = cfg['tokenizer']['eot_id']
LOGIT_SCALE = cfg['model']['logit_scale']
VOCAB_SIZE = cfg['model']['vocab_size']
N_EMBD = cfg['model']['n_embd']
N_HEAD = cfg['model']['n_head']
N_LAYER = cfg['model']['n_layer']

enc = tiktoken.get_encoding("gpt2")


# -- Full model (same architecture as training, without FlexAttention) --

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        d = x.shape[3] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3).type_as(x)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lamb = torch.nn.Parameter(torch.tensor(0.5))
        self.rotary = Rotary(dim // n_head)
        self.c_proj = CastedLinear(dim, dim)
        self.attn_gate = CastedLinear(12, n_head)

    def forward(self, x, v1):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        # Standard causal attention (without FlexAttention)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, scale=0.1
        )
        y = y.transpose(1, 2).contiguous()
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)
        y = y * gate
        y = y.view_as(x)
        y = self.c_proj(y)
        return y, v1


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head) if layer_idx not in [0, 7] else None
        self.mlp = MLP(n_embd) if layer_idx != 0 else None
        self.lambdas = torch.nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1


class GPTInference(torch.nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER):
        super().__init__()
        self.num_encoder_layers = n_layer // 2
        self.num_decoder_layers = n_layer - self.num_encoder_layers
        self.skip_weights = torch.nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(vocab_size, n_embd),
            h=torch.nn.ModuleList([Block(n_embd, n_head, i) for i in range(n_layer)]),
        ))
        self.lm_head = CastedLinear(n_embd, vocab_size)
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = torch.nn.ModuleList([torch.nn.Embedding(vocab_size, n_embd) for _ in range(3)])
        self.smear_lambda = torch.nn.Parameter(torch.zeros(1))
        self.backout_lambda = torch.nn.Parameter(0.5 * torch.ones(1))

    def forward(self, idx):
        x = self.transformer.wte(idx)
        smear_gate_out = self.smear_lambda * torch.sigmoid(
            self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)])
        )
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)

        x = norm(x)
        x0 = x
        v1 = None

        S = idx.size(1)
        ve = [ve_embed(idx) for ve_embed in self.value_embeds]
        n_layers = len(self.transformer.h)
        ve_list = [None, ve[1], ve[2]] + [None] * (n_layers - 6) + [ve[0], ve[1], ve[2]]

        skip_connections = []
        x_backout = None

        for i in range(self.num_encoder_layers):
            if ve_list[i] is not None and v1 is None:
                n_head = self.transformer.h[i].attn.n_head if self.transformer.h[i].attn else N_HEAD
                v1 = ve_list[i].view(1, S, n_head, -1)
            x, v1 = self.transformer.h[i](x, v1, x0)
            skip_connections.append(x)
            if i == 8:
                x_backout = x

        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None and v1 is None:
                v1 = ve_list[layer_idx].view(1, S, N_HEAD, -1)
            x, v1 = self.transformer.h[layer_idx](x, v1, x0)
            if layer_idx == 8:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)
        logits = self.lm_head(x)
        logits = LOGIT_SCALE * torch.tanh(logits / LOGIT_SCALE)
        return logits


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    print(f"Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    val_loss = ckpt.get('val_loss', None)
    print(f"  Step: {ckpt.get('step', '?')}, Val loss: {f'{val_loss:.4f}' if val_loss else 'N/A'}")

    model = GPTInference()

    # Remove _orig_mod. prefix from torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model = model.to(device).bfloat16()
    model.eval()

    # torch.compile speeds up ~2-3x at inference but requires Triton (Linux only)
    import platform
    if platform.system() != "Windows":
        model = torch.compile(model)
        compiled = "compiled"
    else:
        compiled = "eager"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params/1e6:.1f}M parameters, all weights OK ({compiled})")
    return model


@torch.no_grad()
def generate(model, n_words: int, prompt: str = "",
             temperature: float = 0.8, top_k: int = 50,
             device: str = "cuda", max_subwords: int = 15):

    total_words = n_words
    tokens = []

    prompt_words = []
    if prompt:
        prompt_words = prompt.split()
        total_words += len(prompt_words)

    ff_val = total_words - 1

    for word in prompt_words:
        tokens.append(FF_BASE_ID + min(ff_val, FF_MAX))
        tokens.extend(enc.encode(word))
        ff_val -= 1

    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    generated_text = prompt
    words_generated = 0

    for _ in range(n_words):
        if ff_val < 0:
            break

        ff_token = torch.tensor([FF_BASE_ID + min(ff_val, FF_MAX)], dtype=torch.long, device=device)
        tokens = torch.cat([tokens, ff_token])

        word_tokens = []
        for _ in range(max_subwords):
            ctx = tokens[-cfg['model']['sequence_length']:]
            logits = model(ctx[None])
            logits = logits[0, -1].float()

            logits /= max(temperature, 1e-5)
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[-1]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token >= FF_BASE_ID:
                break

            # EOT becomes part of the sequence (model expects to see it) but not part of the text
            if next_token == EOT_ID:
                tokens = torch.cat([tokens, torch.tensor([next_token], dtype=torch.long, device=device)])
                continue

            word_tokens.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([next_token], dtype=torch.long, device=device)])

        if word_tokens:
            word = enc.decode(word_tokens)
            if generated_text and not generated_text.endswith((" ", "\n", "(", "[", "'")):
                generated_text += " "
            generated_text += word

        words_generated += 1
        ff_val -= 1

    total_ff_words = len(prompt_words) + words_generated

    # Build scaffolded version (with <ff_N> tokens visible)
    scaffolded = []
    ff_val_scaffold = total_ff_words - 1
    for word in (generated_text.split()):
        scaffolded.append(f"<ff_{ff_val_scaffold}> {word}")
        ff_val_scaffold -= 1
    scaffolded_text = " ".join(scaffolded)

    return generated_text, scaffolded_text, total_ff_words


def main():
    parser = argparse.ArgumentParser(description="FF Countdown GPT Inference")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--words", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--show-tokens", action="store_true",
                        help="Show scaffolded text with <ff_N> tokens")
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpts = sorted([f for f in glob.glob("checkpoints/model*.pt") if "last" not in f])
        if not ckpts:
            print("No checkpoint found in checkpoints/")
            return
        args.checkpoint = ckpts[-1]

    model = load_checkpoint(args.checkpoint)

    for i in range(args.n_samples):
        if args.n_samples > 1:
            print(f"\n--- Sample {i+1}/{args.n_samples} ---")
        text, scaffolded, n_words = generate(model, args.words, args.prompt, args.temperature, args.top_k)
        if args.show_tokens:
            print(f"\n{'='*60}")
            print(scaffolded)
            print(f"{'='*60}")
        print(f"\n{'='*60}")
        print(text)
        print(f"{'='*60}")
        print(f"({n_words} words)")


if __name__ == "__main__":
    main()
