"""
UniversalMuon — Single optimizer for ALL parameter shapes.

Phase 1 (Series 15): Kill the AdamW crutch.

| Shape | Operation |
|-------|-----------|
| 2D (m x n) | NS orthogonalization + aspect ratio scaling (same as Muon) |
| 1D (d,) | Unit normalization: g/||g|| * scale_1d |
| 0D (scalar) | Raw gradient (SGD-like) |

All shapes use Nesterov momentum with the same beta1.
"""

import torch

from cum.newton_schulz import newton_schulz_orthogonalize
from cum.utils import aspect_ratio_scale


class UniversalMuon(torch.optim.Optimizer):
    """
    Universal Muon optimizer that handles all parameter shapes
    with a single optimizer instance. No auxiliary AdamW needed.

    Args:
        params: All model parameters (no splitting required).
        lr: Learning rate (default 0.02, same as Muon).
        beta1: Nesterov momentum coefficient (default 0.95).
        ns_steps: Newton-Schulz iteration steps for 2D params (default 5).
        scale_1d: Scaling factor for 1D normalized gradients (default 1.0).
        eps: Numerical stability for normalization (default 1e-7).
        nesterov: Use Nesterov momentum (default True). If False, use heavy-ball.
    """

    def __init__(self, params, lr=0.02, beta1=0.95, ns_steps=5,
                 scale_1d=1.0, eps=1e-7, nesterov=True):
        defaults = dict(lr=lr, beta1=beta1, ns_steps=ns_steps,
                        scale_1d=scale_1d, eps=eps, nesterov=nesterov)
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
            scale_1d = group['scale_1d']
            eps = group['eps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                m = state['momentum_buffer']

                # EMA momentum update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Nesterov lookahead (or plain momentum)
                if nesterov:
                    u = grad + beta1 * m
                else:
                    u = m

                # Shape-dependent update
                if p.ndim == 2:
                    # 2D: NS orthogonalization + aspect ratio scaling
                    orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)
                    scale = aspect_ratio_scale(p.shape[0], p.shape[1])
                    p.add_(orth, alpha=-lr * scale)

                elif p.ndim == 1:
                    # 1D: Unit normalization — polar factor of a vector is its direction
                    norm = u.norm()
                    if norm > eps:
                        direction = u / norm
                    else:
                        direction = u
                    p.add_(direction, alpha=-lr * scale_1d)

                elif p.ndim == 0:
                    # Scalar: raw gradient (SGD)
                    p.add_(u, alpha=-lr)

                else:
                    # 3D+ tensors: reshape to 2D, apply NS, reshape back
                    # (Future Phase 3 will do proper tensor decomposition)
                    original_shape = u.shape
                    u_2d = u.reshape(u.shape[0], -1)
                    orth = newton_schulz_orthogonalize(u_2d, steps=ns_steps, eps=eps)
                    scale = aspect_ratio_scale(u_2d.shape[0], u_2d.shape[1])
                    p.add_(orth.reshape(original_shape), alpha=-lr * scale)

        return loss
