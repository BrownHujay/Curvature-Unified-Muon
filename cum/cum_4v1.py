import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM4v1(Optimizer):
    """
    CUM 4v1: Shampoo (Kronecker-Factored Preconditioning).

    NOT a Muon variant. Completely different algorithm:
    - Muon: polar factor of gradient (NS iteration)
    - Shampoo: natural gradient approximation via Kronecker factors

    Maintains running covariance of gradients from both sides:
      L = EMA(G @ G.T)   (m×m, row covariance)
      R = EMA(G.T @ G)   (n×n, column covariance)

    Preconditions the gradient:
      update = L^{-1/4} @ G @ R^{-1/4}

    This is a cheap approximation to the full natural gradient:
    - Decorrelates output neurons (via L)
    - Decorrelates input neurons (via R)
    - Captures cross-direction correlations that Muon ignores

    Eigendecomposition computed every `precond_freq` steps (amortized cost).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.999,
        precond_freq: int = 10,
        eps: float = 1e-12,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, precond_freq=precond_freq,
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
            beta2 = group["beta2"]
            precond_freq = group["precond_freq"]
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
                    # Kronecker factors (gradient covariance)
                    state["L"] = torch.zeros(m_dim, m_dim)  # row covariance
                    state["R"] = torch.zeros(n_dim, n_dim)  # col covariance
                    # Preconditioners (start as identity)
                    state["L_inv_fourth"] = torch.eye(m_dim)
                    state["R_inv_fourth"] = torch.eye(n_dim)

                state["step"] += 1

                # Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Update Kronecker factors with current gradient
                state["L"].mul_(beta2).addmm_(g, g.t(), alpha=1 - beta2)
                state["R"].mul_(beta2).addmm_(g.t(), g, alpha=1 - beta2)

                # Recompute preconditioners every precond_freq steps
                if state["step"] % precond_freq == 1 or precond_freq == 1:
                    # L^{-1/4} via eigendecomposition
                    L_reg = state["L"] + eps * torch.eye(m_dim)
                    eigvals_L, eigvecs_L = torch.linalg.eigh(L_reg)
                    eigvals_L = eigvals_L.clamp(min=eps)
                    state["L_inv_fourth"] = (
                        eigvecs_L @ torch.diag(eigvals_L.pow(-0.25)) @ eigvecs_L.t()
                    )

                    # R^{-1/4} via eigendecomposition
                    R_reg = state["R"] + eps * torch.eye(n_dim)
                    eigvals_R, eigvecs_R = torch.linalg.eigh(R_reg)
                    eigvals_R = eigvals_R.clamp(min=eps)
                    state["R_inv_fourth"] = (
                        eigvecs_R @ torch.diag(eigvals_R.pow(-0.25)) @ eigvecs_R.t()
                    )

                # Precondition: L^{-1/4} @ u @ R^{-1/4}
                update = state["L_inv_fourth"] @ u @ state["R_inv_fourth"]

                # Normalize to match Muon-like update scale
                target_scale = 0.877 * math.sqrt(min(m_dim, n_dim))
                update = update * (target_scale / (update.norm() + 1e-8))

                # Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(update.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr * scale)

        return loss
