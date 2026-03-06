import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale
from .newton_schulz import newton_schulz_orthogonalize


class CUM6v6(Optimizer):
    """
    CUM 6v6: MARS — Variance-reduced Muon via scaled stochastic recursive momentum.

    Wraps Muon's NS orthogonalization with MARS variance reduction:
        m_t = beta1 * m_{t-1} + (1-beta1) * (g_t + gamma * beta1 * NS(g_t - g_{t-1}))

    The correction term NS(g_t - g_{t-1}) reduces gradient variance by exploiting
    temporal correlation between consecutive gradients, scaled through the
    preconditioner (NS) to avoid conflict between variance reduction and
    preconditioning.

    Key insight from MARS (Gu et al., 2024; ICML 2025): naive variance reduction
    conflicts with preconditioning because the VR estimator lives in a different
    space than the preconditioned gradient. Scaling the correction through the
    same preconditioner (NS) resolves this.

    Cost: ~2x Muon (one extra NS call per step for the correction term).
    Memory: +1 gradient clone (prev_grad) per parameter.

    Args:
        lr: Learning rate (default: 0.02)
        beta1: Momentum coefficient (default: 0.95)
        gamma: Variance reduction strength (0 = no VR = standard Muon, 1 = full VR)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        nesterov: Use Nesterov momentum (default: True)
        eps: Numerical stability constant (default: 1e-7)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        gamma: float = 0.5,
        ns_steps: int = 5,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, gamma=gamma, ns_steps=ns_steps,
            nesterov=nesterov, eps=eps,
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
            gamma = group["gamma"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            eps = group["eps"]

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
                    # No prev_grad yet — first step will skip VR correction

                state["step"] += 1

                # ----------------------------------------------------------
                # MARS variance reduction
                # ----------------------------------------------------------
                if "prev_grad" in state and gamma > 0:
                    # Compute gradient difference
                    g_prev = state["prev_grad"]
                    g_diff = g - g_prev

                    # Apply NS to the difference (preconditioner-scaled correction)
                    correction = newton_schulz_orthogonalize(g_diff, steps=ns_steps)

                    # Variance-reduced gradient estimate
                    g_vr = g + gamma * beta1 * correction
                else:
                    # First step or gamma=0: standard gradient (no VR)
                    g_vr = g

                # ----------------------------------------------------------
                # Standard Muon momentum on the VR gradient
                # ----------------------------------------------------------
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g_vr, alpha=1 - beta1)

                if nesterov:
                    u = g_vr + beta1 * mb
                else:
                    u = mb.clone()

                # ----------------------------------------------------------
                # NS orthogonalization of the update
                # ----------------------------------------------------------
                orth = newton_schulz_orthogonalize(u, steps=ns_steps)

                # ----------------------------------------------------------
                # Weight update with aspect ratio scaling
                # ----------------------------------------------------------
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

                # ----------------------------------------------------------
                # Save current gradient for next step's VR correction
                # ----------------------------------------------------------
                state["prev_grad"] = g.clone()

        return loss
