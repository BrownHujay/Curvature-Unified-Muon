"""
CUM 12v3: Adaptive Oscillation Scheduling.

Start with stronger oscillation (high |p'|, more exploration) and gradually
transition to weaker oscillation (low |p'|, more exploitation) via cosine schedule.
Mirrors learning rate scheduling but for the polynomial dynamics.

Uses same coefficient derivation as 12v1 (bifurcation_coeffs), but interpolates
the target derivative over training.
"""

import math
import torch
from torch.optim.optimizer import Optimizer

from .cum_12v1 import bifurcation_coeffs, _custom_ns, _frobenius_blend
from .utils import aspect_ratio_scale


class CUM12v3(Optimizer):
    """
    Adaptive oscillation scheduling with combined mode.

    Cosine-anneals p'(sigma*) from deriv_start to deriv_end over training.
    More negative deriv = more oscillation (exploration).
    Less negative deriv = less oscillation (exploitation).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        deriv_start: float = -1.8,
        deriv_end: float = -1.0,
        sigma_star: float = 0.868,
        c_coeff: float = 2.0315,
        total_steps: int = 2000,
        mode: str = "combined",
        ns_steps: int = 5,
        save_at: int = 2,
        blend: float = 0.15,
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1,
            deriv_start=deriv_start, deriv_end=deriv_end,
            sigma_star=sigma_star, c_coeff=c_coeff,
            total_steps=total_steps,
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
            deriv_start = group["deriv_start"]
            deriv_end = group["deriv_end"]
            sigma_star = group["sigma_star"]
            c_coeff = group["c_coeff"]
            total = group["total_steps"]
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

                # Cosine schedule: deriv_start -> deriv_end
                progress = min(state["step"] / max(total, 1), 1.0)
                deriv_t = deriv_end + (deriv_start - deriv_end) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
                ns_a, ns_b, ns_c = bifurcation_coeffs(deriv_t, sigma_star, c_coeff)

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
