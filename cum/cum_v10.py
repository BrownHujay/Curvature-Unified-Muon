import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_dampened_multi_resolution
from .utils import aspect_ratio_scale


class CUMv10(Optimizer):
    """
    CUM v10: Dampened NS + Multi-Resolution Blend.

    Combines the two approaches that independently beat Muon:
      - v5's multi-resolution blend (NS₂ + NS₅) → -0.0113 vs Muon
      - v9's dampened late-stage NS → -0.0089 vs Muon

    Key insight from v9: dampening makes the FINAL output retain more curvature.
    Key insight from v5: blending with the NS₂ INTERMEDIATE adds curvature.

    v10 gets curvature from BOTH ends of the blend:
      - The NS₂ intermediate has ~25% original SV spread (same as v5)
      - The dampened NS₅ has ~5-10% original SV spread (vs ~0% in standard NS₅)

    So both the "clean" signal and the "curvature" signal are richer than in v5.
    The blend combines two curvature-rich sources instead of one curvature-rich
    source and one curvature-free source.

    Cost: Same as v5 (one intermediate clone). No additional matmuls beyond v9.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        ns_save_at: int = 2,
        ns_blend: float = 0.15,
        dampen_after: int = 2,
        dampen_factor: float = 0.3,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, ns_save_at=ns_save_at,
            ns_blend=ns_blend, dampen_after=dampen_after,
            dampen_factor=dampen_factor, eps=eps, nesterov=nesterov,
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
            ns_blend = group["ns_blend"]
            dampen_after = group["dampen_after"]
            dampen_factor = group["dampen_factor"]
            eps = group["eps"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                orig_shape = g.shape

                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)

                m_dim, n_dim = g.shape

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)

                state["step"] += 1

                # Phase 1: Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Dampened multi-resolution NS
                orth_full, orth_partial = newton_schulz_dampened_multi_resolution(
                    u, steps=ns_steps, save_at=ns_save_at,
                    dampen_after=dampen_after, dampen_factor=dampen_factor,
                    eps=eps,
                )

                # Phase 3: Spectral blend (same as v5)
                if ns_blend > 0:
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
