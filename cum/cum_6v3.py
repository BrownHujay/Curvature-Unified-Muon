import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM6v3(Optimizer):
    """
    CUM 6v3: PSGD Kron — Lie group learned Kronecker preconditioner.

    Instead of a fixed spectral transformation (NS), learns a Kronecker-factored
    preconditioner P = Q1^T Q1 (x) Q2^T Q2 via Lie group optimization on GL(n).
    The preconditioner approximates the inverse Hessian.

    Q1, Q2 are updated so that Q1 @ grad @ Q2.T has approximately identity
    covariance (whitening objective). The preconditioned update is then:
        update = Q1.T @ (Q1 @ u @ Q2.T) @ Q2

    Q1, Q2 evolve via multiplicative Lie group steps:
        Q1 <- Q1 - precond_lr * (GtG - I) @ Q1 / m
        Q2 <- Q2 - precond_lr * (QtQ - I) @ Q2 / n
    where GtG = Qg @ Qg.T and QtQ = Qg.T @ Qg measure how far the
    preconditioned gradient is from having identity covariance.

    Preconditioner updates run only every `precond_freq` steps to amortize
    the O(m^3 + n^3) cubic cost of the Q updates.

    Reference: PSGD (Xi-Lin Li, 2015-2025)

    Cost: O(m^2*n + m*n^2) per step for preconditioning,
          plus O(m^3 + n^3) every precond_freq steps for Q updates.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        precond_lr: float = 0.1,
        precond_freq: int = 10,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, precond_lr=precond_lr,
            precond_freq=precond_freq, nesterov=nesterov, eps=eps,
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
            precond_lr = group["precond_lr"]
            precond_freq = group["precond_freq"]
            nesterov = group["nesterov"]
            eps = group["eps"]

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
                    # Initialize Q1 (m x m) and Q2 (n x n) as identity
                    state["Q1"] = torch.eye(m_dim, dtype=g.dtype, device=g.device)
                    state["Q2"] = torch.eye(n_dim, dtype=g.dtype, device=g.device)

                state["step"] += 1
                step = state["step"]

                # --- Standard momentum (same as Muon) ---
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                Q1 = state["Q1"]
                Q2 = state["Q2"]

                # --- Update preconditioner every precond_freq steps ---
                if step % precond_freq == 0:
                    # Preconditioned gradient for whitening objective
                    Qg = Q1 @ u @ Q2.T  # (m x n)

                    # Covariance matrices (should be ~I if well-preconditioned)
                    # Normalize by n_dim and m_dim respectively to get
                    # per-element covariance estimates
                    GtG = Qg @ Qg.T / n_dim  # (m x m), row covariance
                    QtQ = Qg.T @ Qg / m_dim  # (n x n), column covariance

                    # Identity targets
                    I_m = torch.eye(m_dim, dtype=g.dtype, device=g.device)
                    I_n = torch.eye(n_dim, dtype=g.dtype, device=g.device)

                    # Lie group gradient descent on GL(n):
                    #   Q <- Q - lr * (cov - I) @ Q / dim
                    # The /dim keeps the learning rate scale-invariant
                    Q1 = Q1 - precond_lr * ((GtG - I_m) @ Q1) / m_dim
                    Q2 = Q2 - precond_lr * ((QtQ - I_n) @ Q2) / n_dim

                    state["Q1"] = Q1
                    state["Q2"] = Q2

                # --- Apply preconditioner ---
                Qu = Q1 @ u @ Q2.T          # preconditioned (m x n)
                update = Q1.T @ Qu @ Q2     # back to original basis (m x n)

                # --- Normalize: preserve scale of original momentum ---
                u_norm = u.norm()
                update_norm = update.norm()
                update = update / (update_norm + eps) * u_norm

                # --- Apply update ---
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(update.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr * scale)

        return loss
