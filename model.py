"""Run 5 — RoPE + BPE (vocab 1024) + larger block_size.
RoPE eliminates positional embedding parameters, allowing larger BPE vocab.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 1024     # BPE: 256 bytes + 768 merges
    block_size = 512      # RoPE allows larger context for free
    n_layer = 4
    n_head = 6
    n_embd = 192
    dropout = 0.0
    tie_weights = True
    use_rope = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(dim, max_len, theta=10000.0):
    """Precompute RoPE sin/cos tables."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings."""
    # x: (B, n_head, T, head_dim)
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = getattr(cfg, 'use_rope', False)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.proj._is_residual = True
        if self.use_rope:
            cos, sin = precompute_rope(self.head_dim, cfg.block_size)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = ((int(8 / 3 * cfg.n_embd) + 7) // 8) * 8
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)
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
        self.use_rope = getattr(cfg, 'use_rope', False)
        if not self.use_rope:
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
        x = self.tok_emb(idx)
        if not self.use_rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
