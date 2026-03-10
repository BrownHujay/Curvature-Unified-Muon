"""
PerHeadMuon — Per-head orthogonalization for multi-head attention weights.

Instead of one NS call on the full (3*n_heads*d_head, d_model) QKV gradient,
split into n_heads*3 slices of (d_head, d_model) and NS each independently.

This gives per-head spectral equalization: each head gets its own
orthogonalized update direction, respecting the multi-head structure
that flat Muon destroys.
"""

import torch

from cum.newton_schulz import newton_schulz_orthogonalize
from cum.utils import aspect_ratio_scale


class PerHeadMuon(torch.optim.Optimizer):
    """
    Muon with per-head orthogonalization for attention weights.

    Takes param groups with optional 'n_slices' key.
    If n_slices > 1, the gradient's first dimension is split into
    n_slices chunks, each orthogonalized independently via NS.

    Usage:
        # Standard 2D params (MLP weights) — normal Muon
        mlp_params = [{'params': mlp_weights}]
        # Multi-head params — per-head NS
        attn_params = [{'params': qkv_weights, 'n_slices': n_heads * 3}]
        # Or just per-QKV (3 slices)
        attn_params = [{'params': qkv_weights, 'n_slices': 3}]

        opt = PerHeadMuon(mlp_params + attn_params, lr=0.02)
    """

    def __init__(self, params, lr=0.02, beta1=0.95, ns_steps=5,
                 eps=1e-7, nesterov=True):
        defaults = dict(lr=lr, beta1=beta1, ns_steps=ns_steps,
                        n_slices=1, eps=eps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            ns_steps = group['ns_steps']
            eps = group['eps']
            nesterov = group['nesterov']
            n_slices = group.get('n_slices', 1)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                mb = state['momentum_buffer']

                mb.mul_(beta1).add_(grad, alpha=1 - beta1)

                if nesterov:
                    u = grad + beta1 * mb
                else:
                    u = mb

                if p.ndim != 2:
                    # Non-2D: just scale (shouldn't happen if used correctly)
                    p.add_(u, alpha=-lr)
                    continue

                m_dim, n_dim = u.shape

                if n_slices <= 1 or m_dim % n_slices != 0:
                    # Standard Muon: single NS on full matrix
                    orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)
                    scale = aspect_ratio_scale(m_dim, n_dim)
                    p.add_(orth, alpha=-lr * scale)
                else:
                    # Per-head: split rows into n_slices chunks, NS each
                    chunk_size = m_dim // n_slices
                    chunks = u.split(chunk_size, dim=0)
                    orth_chunks = []
                    for chunk in chunks:
                        orth_chunk = newton_schulz_orthogonalize(
                            chunk, steps=ns_steps, eps=eps
                        )
                        orth_chunks.append(orth_chunk)
                    orth = torch.cat(orth_chunks, dim=0)
                    # Scale based on per-slice dimensions
                    scale = aspect_ratio_scale(chunk_size, n_dim)
                    p.add_(orth, alpha=-lr * scale)

        return loss
