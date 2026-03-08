"""
CUM 12v1: Bifurcation Diagram Sweep.

Parameterize the NS polynomial family by |p'(sigma*)| (oscillation strength)
while keeping sigma* as a fixed point. Sweep from convergent (|p'|<1) through
period-2 oscillation (|p'|~1.58) to strongly oscillating (|p'|>2).

For p(s) = a*s + b*s^3 + c*s^5 with fixed point p(s*)=s* and p'(s*)=deriv:
  a = (3 - deriv)/2 + c * s*^4
  b = (deriv - 1 - 4*c*s*^4) / (2*s*^2)

c is free; we fix c=2.0315 (standard) to keep high-SV behavior consistent.
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
    if weight <= 0:
        return primary
    p_norm = primary.norm()
    s_norm = secondary.norm()
    if s_norm > eps:
        secondary_scaled = secondary * (p_norm / s_norm)
        return (1 - weight) * primary + weight * secondary_scaled
    return primary


def bifurcation_coeffs(deriv: float, sigma_star: float = 0.868, c: float = 2.0315):
    """Compute (a, b, c) from target derivative at fixed point."""
    s2 = sigma_star ** 2
    s4 = sigma_star ** 4
    a = (3 - deriv) / 2 + c * s4
    b = (deriv - 1 - 4 * c * s4) / (2 * s2)
    return a, b, c


def _custom_ns(
    G: Tensor, steps: int, ns_a: float, ns_b: float, ns_c: float,
    eps: float, save_at: int = -1,
) -> tuple:
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


class CUM12v1(Optimizer):
    """
    Bifurcation diagram sweep optimizer.

    Parameterized by `deriv` = p'(sigma*) at the fixed point.
    Standard Muon: deriv ~ -1.58. Sweep from -0.9 to -2.8.

    Supports combined mode (iterate blend + temporal blend).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        deriv: float = -1.58,
        sigma_star: float = 0.868,
        c_coeff: float = 2.0315,
        mode: str = "combined",
        ns_steps: int = 5,
        save_at: int = 2,
        blend: float = 0.15,
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        a, b, c = bifurcation_coeffs(deriv, sigma_star, c_coeff)
        defaults = dict(
            lr=lr, beta1=beta1,
            ns_a=a, ns_b=b, ns_c=c,
            deriv=deriv, mode=mode, ns_steps=ns_steps,
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

                elif mode == "combined":
                    save_at = group["save_at"]
                    blend_w = group["blend"]
                    input_beta = group["input_blend_beta"]
                    input_alpha = group["input_blend_alpha"]
                    full, partial = _custom_ns(
                        u, ns_steps, ns_a, ns_b, ns_c, eps, save_at=save_at,
                    )
                    if partial is None:
                        partial = full
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
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
