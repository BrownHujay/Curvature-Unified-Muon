import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_multi_resolution
from .utils import aspect_ratio_scale


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
    """Blend two matrices with Frobenius norm matching."""
    if weight <= 0:
        return primary
    p_norm = primary.norm()
    s_norm = secondary.norm()
    if s_norm > eps:
        secondary_scaled = secondary * (p_norm / s_norm)
        return (1 - weight) * primary + weight * secondary_scaled
    return primary


class CUM11v3(Optimizer):
    """
    CUM 11v3: Selective Un-Equalization.

    After combined mode (iterate blend + temporal blend), apply LIGHT row-wise
    curvature reweighting. NS equalizes all rows; we restore a small fraction
    of the original curvature signal.

    Curvature proxy: EMA of gradient row-norm-squared (cheap, O(mn)).
    Row scaling: (1-alpha) + alpha * clamp(row_curv / mean_curv, lo, hi)

    Different from failed v11 (second-moment grafting):
    - 5% strength not 100%
    - Row-wise not element-wise (preserves within-row NS structure)
    - Built on combined mode, not raw NS
    - Clamped to prevent extremes

    Cost: Same as combined + one row-norm buffer + O(m) scaling (~0%).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        # Combined mode params
        save_at: int = 2,
        blend: float = 0.15,
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        # Un-equalization params
        uneq_alpha: float = 0.05,
        uneq_beta: float = 0.99,
        uneq_clamp_lo: float = 0.5,
        uneq_clamp_hi: float = 2.0,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps,
            save_at=save_at, blend=blend,
            input_blend_beta=input_blend_beta,
            input_blend_alpha=input_blend_alpha,
            uneq_alpha=uneq_alpha, uneq_beta=uneq_beta,
            uneq_clamp_lo=uneq_clamp_lo, uneq_clamp_hi=uneq_clamp_hi,
            eps=eps, nesterov=nesterov,
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
            save_at = group["save_at"]
            blend_w = group["blend"]
            input_beta = group["input_blend_beta"]
            input_alpha = group["input_blend_alpha"]
            uneq_alpha = group["uneq_alpha"]
            uneq_beta = group["uneq_beta"]
            clamp_lo = group["uneq_clamp_lo"]
            clamp_hi = group["uneq_clamp_hi"]
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

                # Update row-norm curvature proxy (from raw gradient, before momentum)
                row_norms_sq = (g * g).sum(dim=1)
                if "row_curv" not in state:
                    state["row_curv"] = row_norms_sq.clone()
                else:
                    state["row_curv"].mul_(uneq_beta).add_(
                        row_norms_sq, alpha=1 - uneq_beta,
                    )

                # Standard momentum + Nesterov
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Combined mode: iterate blend + temporal blend
                full, partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=save_at, eps=eps,
                )
                iterate_blended = _frobenius_blend(full, partial, blend_w, eps)

                if "denoised_ema" not in state:
                    state["denoised_ema"] = iterate_blended.clone()
                else:
                    state["denoised_ema"].mul_(input_beta).add_(
                        iterate_blended, alpha=1 - input_beta,
                    )
                orth = _frobenius_blend(
                    iterate_blended, state["denoised_ema"], input_alpha, eps,
                )

                # Selective un-equalization: row-wise curvature reweighting
                if uneq_alpha > 0:
                    row_curv = state["row_curv"]
                    mean_curv = row_curv.mean() + eps
                    raw_scale = row_curv / mean_curv
                    clamped_scale = raw_scale.clamp(clamp_lo, clamp_hi)
                    final_scale = (1 - uneq_alpha) + uneq_alpha * clamped_scale
                    orth = orth * final_scale.unsqueeze(1)

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
