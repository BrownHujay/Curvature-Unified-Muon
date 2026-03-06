import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_multi_resolution
from .utils import aspect_ratio_scale


class CUMv5(Optimizer):
    """
    CUM v5: Multi-Resolution Newton-Schulz.

    Key insight: The NS iteration progressively equalizes singular values.
    After step k, the matrix is partially orthogonalized with some SV spread
    remaining. We can blend the fully-converged output (step 5) with a
    partially-converged intermediate (step 2) to get:

      update = (1 - alpha) * NS_full + alpha * NS_partial

    Why this is better than v3's soft NS:
    - v3 blends NS_full with the raw normalized gradient (full noise)
    - v5 blends NS_full with NS_partial (partially denoised, better curvature)
    - The NS_partial intermediate has already had 2 steps of denoising,
      so it's a higher-quality curvature signal

    Cost: Same as Muon (no extra matmuls — just saves one intermediate).
    Memory: One extra matrix clone at the save point.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        ns_save_at: int = 2,
        eps: float = 1e-7,
        nesterov: bool = True,
        ns_blend: float = 0.1,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, ns_save_at=ns_save_at,
            eps=eps, nesterov=nesterov, ns_blend=ns_blend,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            ns_steps = group["ns_steps"]
            ns_save_at = group["ns_save_at"]
            eps = group["eps"]
            nesterov = group["nesterov"]
            ns_blend = group["ns_blend"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                orig_shape = g.shape

                # Handle conv weights: reshape to 2D
                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)

                m_dim, n_dim = g.shape

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)

                state["step"] += 1

                # Phase 1: Momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Multi-Resolution NS
                # Returns both the fully-converged and partially-converged outputs
                orth_full, orth_partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=ns_save_at, eps=eps,
                )

                # Phase 3: Spectral Blend
                # orth_full: all SVs equalized (pure direction, no curvature)
                # orth_partial: SVs partially equalized (direction + curvature)
                # Blend to get: mostly orthogonal + some curvature info
                if ns_blend > 0:
                    # Normalize to same scale before blending
                    full_norm = orth_full.norm()
                    partial_norm = orth_partial.norm()
                    if partial_norm > eps:
                        orth_partial_scaled = orth_partial * (full_norm / partial_norm)
                        orth = (1.0 - ns_blend) * orth_full + ns_blend * orth_partial_scaled
                    else:
                        orth = orth_full
                else:
                    orth = orth_full

                # Phase 4: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
