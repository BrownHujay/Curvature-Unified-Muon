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


class CUM11v2(Optimizer):
    """
    CUM 11v2: Per-Layer Adaptive Combined Mode.

    Different layers have different gradient spectra. A uniform blend weight
    is suboptimal -- layers where NS causes more change (higher spectral spread)
    lose more curvature and should get MORE blending.

    Measures "change ratio" = ||NS(u_norm) - u_norm|| / ||u_norm|| per parameter.
    Adapts both iterate blend and temporal blend weights:
        adapted_blend = base_blend * (1 + scale * change_ratio)

    Different from failed adaptive_residual: this adapts the AMOUNT of simple
    averaging, not the weighting logic itself. Statistical cancellation preserved.

    Cost: Same as combined + one extra norm computation per step (~0%).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        save_at: int = 2,
        # Base blend weights (adapted per-layer at runtime)
        base_blend: float = 0.15,
        base_input_alpha: float = 0.15,
        # Adaptation strength
        adapt_scale: float = 1.0,
        # Temporal EMA
        input_blend_beta: float = 0.5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, save_at=save_at,
            base_blend=base_blend, base_input_alpha=base_input_alpha,
            adapt_scale=adapt_scale, input_blend_beta=input_blend_beta,
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
            base_blend = group["base_blend"]
            base_input_alpha = group["base_input_alpha"]
            adapt_scale = group["adapt_scale"]
            input_beta = group["input_blend_beta"]
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

                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Get NS output + intermediate
                full, partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=save_at, eps=eps,
                )

                # Compute change ratio: how much did NS change the gradient?
                u_norm = u.norm() + eps
                u_normed = u / u_norm
                change_ratio = (full - u_normed).norm() / (u_normed.norm() + eps)

                # Adapt blend weights
                adapted_blend = min(
                    base_blend * (1 + adapt_scale * change_ratio.item()), 0.5,
                )
                adapted_alpha = min(
                    base_input_alpha * (1 + adapt_scale * change_ratio.item()), 0.5,
                )

                # Within-step iterate blend (adapted)
                iterate_blended = _frobenius_blend(full, partial, adapted_blend, eps)

                # Update denoised EMA
                if "denoised_ema" not in state:
                    state["denoised_ema"] = iterate_blended.clone()
                else:
                    state["denoised_ema"].mul_(input_beta).add_(
                        iterate_blended, alpha=1 - input_beta,
                    )

                # Across-step temporal blend (adapted)
                orth = _frobenius_blend(
                    iterate_blended, state["denoised_ema"], adapted_alpha, eps,
                )

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
