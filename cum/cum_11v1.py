import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


# Default: standard Muon/Zetta NS polynomial
_DEFAULT_A = 3.4445
_DEFAULT_B = -4.7750
_DEFAULT_C = 2.0315


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


def _custom_ns(
    G: Tensor, steps: int, ns_a: float, ns_b: float, ns_c: float,
    eps: float, save_at: int = -1,
) -> tuple:
    """
    Newton-Schulz iteration with arbitrary polynomial coefficients.
    p(sigma) = a*sigma + b*sigma^3 + c*sigma^5

    Returns (final, intermediate_or_None).
    """
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    intermediate = None
    for i in range(steps):
        A = X @ X.T
        B = ns_b * A + ns_c * (A @ A)
        X = ns_a * X + B @ X
        if i + 1 == save_at:
            intermediate = X.T.clone() if transposed else X.clone()

    if transposed:
        X = X.T
    return X, intermediate


class CUM11v1(Optimizer):
    """
    CUM 11v1: Learned Polynomial Coefficients.

    The standard NS polynomial (a=3.4445, b=-4.7750, c=2.0315) was optimized
    for convergence to polar factor (SVs->1.0). But SVs=1.0 is wrong -- 0.88 is
    empirically better.

    The "stable-0.88" polynomial (a=2.0, b=-1.940, c=0.836):
    - p(0.88) ~ 0.88 (IS a fixed point)
    - p'(0.88) ~ 0.0 (super-stable, derivative near zero)
    - Small SVs grow: 0.1 -> 0.198 -> ... -> 0.873 in 5 steps

    Modes:
    - "basic": Custom NS polynomial, no blending
    - "input_blend": Custom NS + temporal EMA blend
    - "combined": Custom NS + iterate blend + temporal blend
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        # Custom polynomial coefficients
        ns_a: float = _DEFAULT_A,
        ns_b: float = _DEFAULT_B,
        ns_c: float = _DEFAULT_C,
        # Mode
        mode: str = "basic",       # "basic", "input_blend", "combined"
        ns_steps: int = 5,
        # Combined mode params (iterate blend)
        save_at: int = 2,
        blend: float = 0.15,
        # Input blend params (temporal)
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1,
            ns_a=ns_a, ns_b=ns_b, ns_c=ns_c,
            mode=mode, ns_steps=ns_steps,
            save_at=save_at, blend=blend,
            input_blend_beta=input_blend_beta,
            input_blend_alpha=input_blend_alpha,
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
            ns_a = group["ns_a"]
            ns_b = group["ns_b"]
            ns_c = group["ns_c"]
            mode = group["mode"]
            ns_steps = group["ns_steps"]
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

                if mode == "basic":
                    orth, _ = _custom_ns(u, ns_steps, ns_a, ns_b, ns_c, eps)

                elif mode == "input_blend":
                    input_beta = group["input_blend_beta"]
                    input_alpha = group["input_blend_alpha"]
                    full, _ = _custom_ns(u, ns_steps, ns_a, ns_b, ns_c, eps)
                    if "denoised_ema" not in state:
                        state["denoised_ema"] = full.clone()
                    else:
                        state["denoised_ema"].mul_(input_beta).add_(full, alpha=1 - input_beta)
                    orth = _frobenius_blend(full, state["denoised_ema"], input_alpha, eps)

                elif mode == "combined":
                    save_at = group["save_at"]
                    blend = group["blend"]
                    input_beta = group["input_blend_beta"]
                    input_alpha = group["input_blend_alpha"]
                    full, partial = _custom_ns(
                        u, ns_steps, ns_a, ns_b, ns_c, eps, save_at=save_at,
                    )
                    if partial is None:
                        partial = full
                    iterate_blended = _frobenius_blend(full, partial, blend, eps)
                    if "denoised_ema" not in state:
                        state["denoised_ema"] = iterate_blended.clone()
                    else:
                        state["denoised_ema"].mul_(input_beta).add_(
                            iterate_blended, alpha=1 - input_beta
                        )
                    orth = _frobenius_blend(
                        iterate_blended, state["denoised_ema"], input_alpha, eps,
                    )

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
