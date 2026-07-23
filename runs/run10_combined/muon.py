"""Muon optimizer — Momentum Orthogonalized by Newton-Schulz.

Pure PyTorch implementation (no triton, no distributed, CPU-safe).
Apply ONLY to 2D weight matrices. Use AdamW for embeddings, norms, biases.

Reference: https://kellerjordan.github.io/posts/muon/
From modded-nanogpt by Keller Jordan et al.
"""
import torch
from torch.optim import Optimizer


def newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration to compute the polar factor (orthogonalization) of G.
    Uses a quintic iteration with coefficients optimized for max slope at zero.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()  # Use float32 on CPU (no bfloat16 benefit)
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """Muon optimizer for 2D weight matrices.

    Uses Nesterov momentum followed by Newton-Schulz orthogonalization.
    Only maintains one momentum buffer (more memory-efficient than Adam).

    Args:
        params: Parameters to optimize (should all be 2D)
        lr: Learning rate (typically 0.02, much higher than Adam)
        momentum: Nesterov momentum coefficient (default 0.95)
        ns_steps: Number of Newton-Schulz iterations (default 5)
        weight_decay: Decoupled weight decay (default 0.0)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                # Get or init momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']

                # Nesterov momentum update
                buf.mul_(momentum).add_(g)
                g_nesterov = g.add(buf, alpha=momentum)

                # Newton-Schulz orthogonalization (only for 2D)
                if g_nesterov.ndim == 2:
                    update = newtonschulz5(g_nesterov, steps=ns_steps)
                else:
                    update = g_nesterov

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss
