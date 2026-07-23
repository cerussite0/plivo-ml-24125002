"""Run 3 — Scaled architecture.
Changes from Run 2:
  1. n_embd=192 (was 160), n_head=6 (was 4), n_layer=4
  2. block_size=256 (was 128)
  3. RMSNorm instead of LayerNorm (simpler, slightly better)
  4. SwiGLU MLP (more expressive at similar param count)
  5. All Run 2 improvements retained (tie_weights, scaled init)
Target params: ~1.87M (under 2M cap)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256
    block_size = 256      # CHANGED: was 128
    n_layer = 4
    n_head = 6            # CHANGED: was 4
    n_embd = 192          # CHANGED: was 160
    dropout = 0.0
    tie_weights = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — no mean-centering bias."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.proj._is_residual = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    """SwiGLU activation: gate * swish(proj). More expressive than GELU MLP."""
    def __init__(self, cfg):
        super().__init__()
        # SwiGLU uses 8/3 * d instead of 4*d for similar param count
        hidden = int(8 / 3 * cfg.n_embd)
        # Round to multiple of 8 for efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)   # gate
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)   # up
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)   # down
        self.drop = nn.Dropout(cfg.dropout)
        self.w3._is_residual = True

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(lambda m: self._init(m, cfg))

    def _init(self, m, cfg):
        if isinstance(m, nn.Linear):
            std = 0.02
            if hasattr(m, '_is_residual') and m._is_residual:
                std *= (2 * cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
