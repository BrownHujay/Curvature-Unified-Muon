"""
CUM 13v1: Minimax-Optimal Polynomial from Theory Analysis.

Uses polynomials found by numerical optimization (differential evolution)
that achieve near-perfect SV equalization after 5 iterations.

These polynomials are OUTSIDE the bifurcation family (no fixed point at σ*=0.868).
They represent the theoretical optimum for iterated SV equalization.

Three presets:
  "minimax": min max |p⁵(σ) - 0.88| → a=2.6806, b=-3.6311, c=1.8871
  "minvar":  min Var[p⁵(σ)]          → a=2.2311, b=-3.2137, c=1.9518
  "l2":      min E[|p⁵(σ) - 0.88|²] → a=2.6704, b=-3.6166, c=1.8848

Also supports arbitrary (ns_a, ns_b, ns_c) for custom coefficients.
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


PRESETS = {
    "minimax": (2.6806, -3.6311, 1.8871),
    "minvar": (2.2311, -3.2137, 1.9518),
    "l2": (2.6704, -3.6166, 1.8848),
    "standard": (3.4445, -4.7750, 2.0315),
}


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
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


def _custom_ns_all(
    G: Tensor, steps: int, ns_a: float, ns_b: float, ns_c: float, eps: float,
) -> list:
    """Run custom polynomial NS and return ALL iterates [NS_1, ..., NS_n]."""
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    iterates = []
    for _ in range(steps):
        A = X @ X.T
        B = ns_b * A + ns_c * (A @ A)
        X = ns_a * X + B @ X
        iterates.append(X.T.clone() if transposed else X.clone())
    return iterates


class CUM13v1(Optimizer):
    """
    Minimax-optimal polynomial optimizer.

    Supports three modes:
    - "basic": just custom NS
    - "combined": two-point iterate blend + temporal EMA (like 8v1)
    - "td": TD(λ) multi-iterate blend + temporal EMA (like 12v2 final recipe)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        preset: str = "minimax",
        ns_a: float | None = None,
        ns_b: float | None = None,
        ns_c: float | None = None,
        mode: str = "combined",
        ns_steps: int = 5,
        save_at: int = 2,
        blend: float = 0.15,
        td_lambda: float = 0.5,
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        if ns_a is not None and ns_b is not None and ns_c is not None:
            a, b, c = ns_a, ns_b, ns_c
        else:
            a, b, c = PRESETS[preset]

        defaults = dict(
            lr=lr, beta1=beta1,
            ns_a=a, ns_b=b, ns_c=c,
            mode=mode, ns_steps=ns_steps,
            save_at=save_at, blend=blend,
            td_lambda=td_lambda,
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

            # TD weights (precomputed per step, cheap)
            if mode == "td":
                td_lam = group["td_lambda"]
                raw_w = [td_lam ** (ns_steps - k) for k in range(1, ns_steps + 1)]
                total_w = sum(raw_w)
                td_weights = [w / total_w for w in raw_w]

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

                elif mode == "td":
                    input_beta = group["input_blend_beta"]
                    input_alpha = group["input_blend_alpha"]
                    all_iterates = _custom_ns_all(
                        u, ns_steps, ns_a, ns_b, ns_c, eps,
                    )
                    final = all_iterates[-1]
                    f_norm = final.norm()
                    blended = torch.zeros_like(final)
                    for iterate, w in zip(all_iterates, td_weights):
                        i_norm = iterate.norm()
                        if i_norm > eps:
                            blended.add_(iterate * (f_norm / i_norm), alpha=w)
                        else:
                            blended.add_(final, alpha=w)

                    if "denoised_ema" not in state:
                        state["denoised_ema"] = blended.clone()
                    else:
                        state["denoised_ema"].mul_(input_beta).add_(
                            blended, alpha=1 - input_beta,
                        )
                    orth = _frobenius_blend(
                        blended, state["denoised_ema"], input_alpha, eps,
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
