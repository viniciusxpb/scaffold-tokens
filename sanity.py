#!/usr/bin/env python3
"""
Mini sanity training — verifies that the forward, backward, separate loss,
and data loader work correctly with the real shards.

Uses a reduced model (4 layers, 384 dim) to run quickly.
Saves weights before/after, shows decoded input/output.
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import glob
import numpy as np
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
import tiktoken
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# -- Config ------------------------------------------------------------------
VOCAB_SIZE = cfg['model']['vocab_size']
FF_BASE_ID = cfg['tokenizer']['ff_base_id']
EOT_ID = cfg['tokenizer']['eot_id']
SEQ_LEN = 2048
N_STEPS = 30
LR = 1e-3
SHARD_PATTERN_TRAIN = os.path.join(cfg['data']['shards_dir'], "train", "shard_*.bin")
SHARD_PATTERN_VAL = os.path.join(cfg['data']['shards_dir'], "val", "shard_*.bin")

enc = tiktoken.get_encoding("gpt2")

# -- Simple data loader ------------------------------------------------------
def load_shard(path):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == cfg['shards']['magic']
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int64))

def get_batch(tokens, seq_len):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (1,)).item()
    x = tokens[ix : ix + seq_len]
    y = tokens[ix + 1 : ix + seq_len + 1]
    return x.cuda(), y.cuda()

# -- Mini model (no FlexAttention, no compilation) ----------------------------
class MiniGPT(torch.nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, n_embd=384, n_head=6, n_layer=4):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, n_embd)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets):
        x = self.wte(idx)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Total loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Separate loss (no grad)
        with torch.no_grad():
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            ff_mask = flat_targets >= FF_BASE_ID
            word_mask = ~ff_mask
            loss_ff = F.cross_entropy(flat_logits[ff_mask], flat_targets[ff_mask]).item() if ff_mask.any() else 0.0
            loss_word = F.cross_entropy(flat_logits[word_mask], flat_targets[word_mask]).item() if word_mask.any() else 0.0
            n_ff = ff_mask.sum().item()
            n_word = word_mask.sum().item()

        return loss, loss_ff, loss_word, n_ff, n_word, logits


def decode_token(tid):
    """Decode a token ID to a human-readable string."""
    if tid >= FF_BASE_ID:
        return f"<ff_{tid - FF_BASE_ID}>"
    elif tid == EOT_ID:
        return "<EOT>"
    else:
        try:
            return enc.decode([tid])
        except Exception:
            return f"<{tid}>"


def show_prediction(logits, targets, start=0, n=20):
    """Show actual vs predicted tokens."""
    print(f"\n{'Pos':<5} {'Target':<25} {'Predicted':<25} {'Correct':<8}")
    print("-" * 65)
    preds = logits[0, start:start+n].argmax(dim=-1)
    for i in range(n):
        pos = start + i
        t = targets[0, pos].item()
        p = preds[i].item()
        t_str = decode_token(t)
        p_str = decode_token(p)
        ok = "Y" if t == p else ""
        print(f"{pos:<5} {t_str:<25} {p_str:<25} {ok:<8}")


def weight_stats(model):
    """Return dict with the norm of each parameter."""
    stats = {}
    for name, p in model.named_parameters():
        stats[name] = {
            "norm": p.data.norm().item(),
            "mean": p.data.mean().item(),
            "std": p.data.std().item(),
        }
    return stats


def main():
    assert torch.cuda.is_available(), "CUDA required"

    # Load data
    train_files = sorted(glob.glob(SHARD_PATTERN_TRAIN))
    val_files = sorted(glob.glob(SHARD_PATTERN_VAL))
    assert train_files, f"No shards in {SHARD_PATTERN_TRAIN}"
    assert val_files, f"No shards in {SHARD_PATTERN_VAL}"

    print("Loading training shard...")
    train_tokens = load_shard(train_files[0])
    val_tokens = load_shard(val_files[0])
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")

    # Create mini model
    model = MiniGPT().cuda().bfloat16()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nMini model: {n_params/1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -- Weights BEFORE --
    weights_before = weight_stats(model)

    # -- Show input --
    x, y = get_batch(train_tokens, SEQ_LEN)
    print(f"\n{'='*65}")
    print("INPUT SAMPLE (first 40 tokens):")
    print("-" * 65)
    for i in range(40):
        tid = x[i].item()
        print(f"  [{i:4d}] {tid:6d}  {decode_token(tid)}")

    # -- Training --
    print(f"\n{'='*65}")
    print(f"TRAINING {N_STEPS} steps (seq_len={SEQ_LEN})...")
    print(f"{'Step':<6} {'Loss':<10} {'Loss FF':<10} {'Loss Word':<10} {'#FF':<8} {'#Word':<8}")
    print("-" * 55)

    losses = []
    for step in range(N_STEPS):
        x, y = get_batch(train_tokens, SEQ_LEN)
        model.train()
        loss, loss_ff, loss_word, n_ff, n_word, logits = model(x[None], y[None])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"{step:<6} {loss.item():<10.4f} {loss_ff:<10.4f} {loss_word:<10.4f} {n_ff:<8} {n_word:<8}")

    # -- Checks --
    print(f"\n{'='*65}")
    print("CHECKS:")

    # Did loss decrease?
    first_5 = sum(losses[:5]) / 5
    last_5 = sum(losses[-5:]) / 5
    dropped = first_5 - last_5
    print(f"  Mean loss (first 5 steps): {first_5:.4f}")
    print(f"  Mean loss (last 5 steps):  {last_5:.4f}")
    print(f"  Difference: {dropped:.4f} {'(DECREASED)' if dropped > 0 else '(DID NOT DECREASE)'}")

    # Did weights change?
    weights_after = weight_stats(model)
    print(f"\n  Weight changes:")
    changed = 0
    for name in weights_before:
        diff = abs(weights_after[name]["norm"] - weights_before[name]["norm"])
        if diff > 1e-6:
            changed += 1
    print(f"    {changed}/{len(weights_before)} parameters changed in norm")

    # Any NaN/Inf in weights?
    bad = 0
    for name, p in model.named_parameters():
        if torch.isnan(p.data).any() or torch.isinf(p.data).any():
            print(f"    X NaN/Inf in {name}")
            bad += 1
    if bad == 0:
        print(f"    No NaN/Inf in weights")

    # -- Predictions --
    print(f"\n{'='*65}")
    print("PREDICTIONS (last batch):")
    model.eval()
    with torch.no_grad():
        _, _, _, _, _, logits = model(x[None], y[None])

    # Find document start (after an EOT)
    eot_positions = (y == EOT_ID).nonzero(as_tuple=True)[0]
    if len(eot_positions) > 0:
        doc_start = eot_positions[0].item() + 1
        if doc_start + 20 < SEQ_LEN:
            print(f"(document start at position {doc_start})")
            show_prediction(logits, y[None], start=doc_start, n=20)
        else:
            show_prediction(logits, y[None], start=0, n=20)
    else:
        show_prediction(logits, y[None], start=0, n=20)

    # -- Val loss --
    print(f"\n{'='*65}")
    print("VALIDATION:")
    model.eval()
    val_losses = []
    for _ in range(5):
        xv, yv = get_batch(val_tokens, SEQ_LEN)
        with torch.no_grad():
            vl, vl_ff, vl_word, _, _, _ = model(xv[None], yv[None])
            val_losses.append(vl.item())
    print(f"  Val loss: {sum(val_losses)/len(val_losses):.4f}")

    print(f"\n{'='*65}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    print("Sanity check complete.")


if __name__ == "__main__":
    main()
