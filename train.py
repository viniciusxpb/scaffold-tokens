#!/usr/bin/env python3
"""
GPT-2 124M + FF Countdown training.
Based on train_gpt_improved.py from nanoGPT_1GPU_SPEEDRUN.

Changes from original:
- vocab_size: 51264 (BPE + ff tokens + padding)
- sequence_length: 4096 (docs max ~2000 tokens scaffolded)
- Data paths: data/shards/{train,val}
- Checkpoint: keep best N by val_loss
- No FP8 (RTX 3060 SM86)
"""
import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Limit CPU/RAM — let GPU fly
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import torch
torch.set_num_threads(2)
torch.empty(1, device="cuda", requires_grad=True).backward()
# CUDA optimizations
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True

import torch._dynamo as dynamo
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import triton
import triton.language as tl
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# Compile FlexAttention
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(pid, M, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(configs=_get_autotune_configs(), key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"])
@triton.jit
def XXT_kernel(A_ptr, C_ptr, M, K, a_stride_b, a_stride_r, a_stride_c, c_stride_b, c_stride_r, c_stride_c,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
               GROUP_SIZE_M: tl.constexpr, LOWER_UPPER: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c
    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0
    grid = lambda meta: (batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),)
    XXT_kernel[grid](A_ptr=A, C_ptr=out, M=M, K=K, a_stride_b=input_batch_stride, a_stride_r=A.stride(-2),
                     a_stride_c=A.stride(-1), c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1))
    return out

@triton.autotune(configs=_get_autotune_configs(), key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"])
@triton.jit
def ba_plus_cAA_kernel(A_ptr, C_ptr, M, a_stride_b, a_stride_r, a_stride_c, c_stride_b, c_stride_r, c_stride_c,
                       alpha, beta, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr, LOWER_UPPER: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)
    accumulator *= alpha
    accumulator += a_add * beta
    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0
    grid = lambda meta: (batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),)
    ba_plus_cAA_kernel[grid](A_ptr=A, C_ptr=out, M=M, a_stride_b=input_batch_stride, a_stride_r=A.stride(-2),
                             a_stride_c=A.stride(-1), c_stride_b=output_batch_stride, c_stride_r=out.stride(-2),
                             c_stride_c=out.stride(-1), alpha=alpha, beta=beta)
    return out

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=B)
        aX_plus_BX(X, B, X, beta=a, out=C)
        X, C = C, X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon Optimizer with DDP-style gradient distribution (even for single GPU)

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalization."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr)

# -----------------------------------------------------------------------------
# Model Components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

def next_multiple_of_n(v, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()  # Zero init

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Rotary(torch.nn.Module):
    """Standard Rotary Position Embeddings."""
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
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, flex_kernel_options=None):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # Initialize q/k/v with proper init
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_q.weight.uniform_(-bound, bound)
            self.c_k.weight.uniform_(-bound, bound)
            self.c_v.weight.uniform_(-bound, bound)
        # Value residual lambda (simple scalar)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        # Rotary embeddings
        self.rotary = Rotary(dim // n_head)
        # Output projection (zero init)
        self.c_proj = CastedLinear(dim, dim)
        # Gated attention (your architecture)
        self.attn_gate = CastedLinear(12, n_head)
        # FlexAttention kernel options
        self.flex_kernel_options = flex_kernel_options

    def forward(self, x, v1, block_mask, attn_scale=0.1):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=attn_scale,
            kernel_options=self.flex_kernel_options
        )
        y = y.transpose(1, 2).contiguous()  # (B, T, n_head, head_dim)
        # Apply gated attention (your architecture) - gate each head separately
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)  # (B, T, n_head, 1)
        y = y * gate
        y = y.view_as(x)  # (B, T, n_embd)
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        # Initialize c_fc properly
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.weight.uniform_(-bound, bound)
            # c_proj stays zero-init

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # Keep your architecture: skip attention on layers 0 and 7
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.flex_kernel_options) if layer_idx not in [0, 7] else None
        self.mlp = MLP(config.n_embd) if layer_idx != 0 else None
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask, attn_scale):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1, block_mask, attn_scale)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 51264  # BPE 50257 + ff_0..ff_999 + padding to multiple of 64
    n_layer: int = 12
    n_head: int = 6  # head dim 128
    n_embd: int = 768
    flex_kernel_options: dict = None

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # U-net design
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)

        # YOUR ARCHITECTURE: smear_gate and value_embeds
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = nn.ModuleList([nn.Embedding(config.vocab_size, config.n_embd) for _ in range(3)])

        # Learnable scalars for smear and backout
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, idx, target, attn_blocksize):
        # Document boundary masking with progressive window
        docs = (idx == 50256).cumsum(0)

        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & document_mask & window_mask

        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        # Forward pass
        x = self.transformer.wte(idx[None])

        # YOUR ARCHITECTURE: smear gate
        smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)

        x = norm(x)
        x0 = x
        v1 = None

        # YOUR ARCHITECTURE: value embeddings
        ve = [value_embed(idx) for value_embed in self.value_embeds]
        ve_list = [None, ve[1], ve[2]] + [None] * (len(self.transformer.h) - 6) + [ve[0], ve[1], ve[2]]

        # Store outputs for U-Net skip connections
        skip_connections = []
        x_backout = None
        backout_layer = 8

        # Encoder pass
        for i in range(self.num_encoder_layers):
            # Inject value embeddings by updating v1 if applicable
            if ve_list[i] is not None:
                if v1 is None:
                    v1 = ve_list[i][None].view(1, S, self.transformer.h[i].attn.n_head if self.transformer.h[i].attn else 6, -1)
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask, attn_scale=0.1)
            skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # Decoder pass with weighted skip connections
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None:
                if v1 is None:
                    v1 = ve_list[layer_idx][None].view(1, S, 6, -1)
            x, v1 = self.transformer.h[layer_idx](x, v1, x0, block_mask, attn_scale=0.1)
            if layer_idx == backout_layer:
                x_backout = x

        # YOUR ARCHITECTURE: backout lambda
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)
        logits = self.lm_head(x)
        # IMPROVED: tanh logit scaling instead of sigmoid
        logits = logit_scale * torch.tanh(logits / logit_scale)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        # Separated losses for monitoring (no grad impact)
        with torch.no_grad():
            ff_mask = target >= FF_BASE_ID  # ff tokens
            word_mask = ~ff_mask
            loss_ff = F.cross_entropy(logits.view(-1, logits.size(-1))[ff_mask], target.view(-1)[ff_mask]) if ff_mask.any() else torch.tensor(0.0)
            loss_word = F.cross_entropy(logits.view(-1, logits.size(-1))[word_mask], target.view(-1)[word_mask]) if word_mask.any() else torch.tensor(0.0)

        return loss, loss_ff.item(), loss_word.item()

# -----------------------------------------------------------------------------
# Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = torch.frombuffer(f.read(256*4), dtype=torch.int32)
    assert header[0] == 20240520
    assert header[1] == 1
    return int(header[2])

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = torch.frombuffer(f.read(256*4), dtype=torch.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2]
        tokens = torch.frombuffer(f.read(), dtype=torch.uint16)
    assert len(tokens) == ntok
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, T, process_rank=0, num_processes=1):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        batch_size = self.T * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        buf = torch.tensor(buf.numpy().astype('int32'), dtype=torch.long)
        x = buf[:-1]
        y = buf[1:]
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True)

# -----------------------------------------------------------------------------
# Hyperparameters

import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

FF_BASE_ID = cfg['tokenizer']['ff_base_id']
logit_scale = cfg['model']['logit_scale']
attn_window_min = cfg['training']['attention_window_min']
attn_window_max = cfg['training']['attention_window_max']
momentum_min = cfg['training']['momentum_min']
momentum_max = cfg['training']['momentum_max']

@dataclass
class Hyperparameters:
    input_bin: str = os.path.join(cfg['data']['shards_dir'], 'train/shard_*.bin')
    input_val_bin: str = os.path.join(cfg['data']['shards_dir'], 'val/shard_*.bin')
    batch_size: int = cfg['model']['device_batch_size']
    sequence_length: int = cfg['model']['sequence_length']
    num_iterations: int = cfg['training']['num_iterations']
    warmup_iters: int = cfg['training']['warmup_iters']
    cooldown_iters: int = cfg['training']['cooldown_iters']
    weight_decay: float = 0
    val_loss_every: int = cfg['checkpoint']['eval_every_n_steps']
    val_tokens: int = cfg['model']['sequence_length'] * 10
    checkpoint_dir: str = cfg['checkpoint']['dir']
    keep_last_n: int = cfg['checkpoint']['keep_last_n']
    log_every_n_steps: int = cfg['training']['log_every_n_steps']

args = Hyperparameters()
model_config = GPTConfig(
    vocab_size=cfg['model']['vocab_size'],
    n_layer=cfg['model']['n_layer'],
    n_head=cfg['model']['n_head'],
    n_embd=cfg['model']['n_embd'],
    flex_kernel_options={
        "BLOCK_M": 64, "BLOCK_N": 64,
        "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32
    }
)

# Single GPU setup (fake DDP for compatibility)
assert torch.cuda.is_available()
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
device = 'cuda:0'
torch.cuda.set_device(device)
print(f"using device: {device}")

# Logging
run_id = str(uuid.uuid4())
logdir = f'logs/{run_id}/'
os.makedirs(logdir, exist_ok=True)
logfile = f'logs/{run_id}.txt'

def print0(s, logonly=False):
    with open(logfile, "a") as f:
        if not logonly:
            print(s)
        f.write(s+'\n')

with open(logfile, "w") as f:
    f.write(code)
    f.write('='*100 + '\n')

print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

os.makedirs(args.checkpoint_dir, exist_ok=True)
best_val_loss = float('inf')
patience = 0
EARLY_STOP_PATIENCE = cfg['training']['early_stop_patience']

def save_last(step_num):
    """Save 'last' checkpoint (emergency / reference)."""
    path = os.path.join(args.checkpoint_dir, 'model-last.pt')
    torch.save(dict(step=step_num, model=model.state_dict()), path)
    print0(f"Saved last: {path}")

T = args.sequence_length
assert args.val_tokens % T == 0
val_steps = args.val_tokens // T
train_accumulation_steps = args.batch_size

# Load data
train_loader = DistributedDataLoader(args.input_bin, T, 0, 1)
val_loader = DistributedDataLoader(args.input_val_bin, T, 0, 1)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total}")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total}")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

# Create model
model = GPT(model_config)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()

model = torch.compile(model)

# IMPROVED: Separate optimizers with different learning rates
# Like the reference: wte=0.6, lm_head=0.008, Muon=0.05, scalars=0.04
optimizer1 = torch.optim.Adam([model.transformer.wte.weight], lr=cfg['training']['lr_embeddings'], betas=(0.8, 0.95), fused=True)
for ve in model.value_embeds:
    optimizer1.add_param_group({'params': [ve.weight], 'lr': cfg['training']['lr_embeddings']})

optimizer2 = torch.optim.Adam([model.lm_head.weight], lr=cfg['training']['lr_lm_head'], betas=(0.8, 0.95), fused=True)

# Matrix params for Muon
params = list(model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [model.skip_weights, model.smear_lambda, model.backout_lambda]

optimizer3 = Muon(matrix_params, lr=cfg['training']['lr_muon'], momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=cfg['training']['lr_scalars'], betas=(0.8, 0.95), fused=True)

optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

# Learning rate schedule: no warmup, linear cooldown
def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Training loop
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()
current_step = 0

for step in range(args.num_iterations + 1):
  try:
    current_step = step
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # IMPROVED: Progressive attention window (64 to 1792)
    attn_blocksize = torch.tensor(attn_window_min * ((step / args.num_iterations * (attn_window_max - attn_window_min) + attn_window_min) // attn_window_min), dtype=torch.int, device='cuda')

    # Validation
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        val_loss_ff = 0.0
        val_loss_word = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                vl, vl_ff, vl_word = model(x_val, y_val, attn_blocksize=attn_blocksize)
                val_loss += vl
                val_loss_ff += vl_ff
                val_loss_word += vl_word
        val_loss /= val_steps
        val_loss_ff /= val_steps
        val_loss_word /= val_steps
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} val_ff:{val_loss_ff:.4f} val_word:{val_loss_word:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')

        # Save checkpoint only if val_loss improved
        if step > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            ts = int(time.time() * 1000)
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model-{ts}.pt')
            log = dict(step=step, val_loss=float(val_loss), code=code,
                       model=model.state_dict(),
                       optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, checkpoint_path)
            print0(f"Saved: {checkpoint_path}")

            # Keep only last N checkpoints
            ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'model-*.pt')))
            while len(ckpts) > args.keep_last_n:
                os.remove(ckpts.pop(0))
        elif step > 0:
            patience += 1
            print0(f"  No improvement ({patience}/{EARLY_STOP_PATIENCE})")
            if patience >= EARLY_STOP_PATIENCE:
                print0(f"Early stop: val_loss did not improve for {EARLY_STOP_PATIENCE} validations")
                save_last(step)
                break

        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # Training
    model.train()
    train_loss_ff_accum = 0.0
    train_loss_word_accum = 0.0
    for i in range(1, train_accumulation_steps + 1):
        loss, loss_ff, loss_word = model(x, y, attn_blocksize=attn_blocksize)
        x, y = train_loader.next_batch()
        loss.backward()
        train_loss = loss.detach()
        train_loss_ff_accum += loss_ff
        train_loss_word_accum += loss_word
    train_loss_ff_accum /= train_accumulation_steps
    train_loss_word_accum /= train_accumulation_steps

    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps

    # IMPROVED: Muon momentum warmup (0.85 -> 0.95 over 300 steps)
    frac = min(step / cfg['training']['momentum_warmup_steps'], 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * momentum_min + frac * momentum_max

    # Step optimizers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} ff:{train_loss_ff_accum:.4f} word:{train_loss_word_accum:.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

  except KeyboardInterrupt:
    print0(f"\nInterrupted at step {current_step}.")
    save_last(current_step)
    break

print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0(f"Training complete! Final run_id: {run_id}")
