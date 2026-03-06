import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM4v2(Optimizer):
    """
    CUM 4v2: SODA (Spectral Outlier Dampening Algorithm).

    NOT a Muon variant. Novel algorithm:
    - Muon: NS equalizes ALL singular values (aggressive, destroys curvature)
    - SODA: only dampens the TOP-k singular values (outliers), leaves rest alone

    Uses cheap subspace iteration (NOT full SVD, NOT NS) to find the top-k
    singular components. Dampens them: σ → σ^α (α < 1 compresses outliers).
    The bottom (rank - k) singular values are untouched.

    Hypothesis: the top few SVs are noisy outliers that cause overshooting.
    Dampening them while preserving the natural structure of the remaining
    spectrum might be better than NS's aggressive full equalization.

    Cost: O(mnk) per step for subspace iteration (k=4-8, 5 iters).
    Compare: NS is O(m²n * 5_steps). SODA is ~40x cheaper.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        top_k: int = 4,
        dampen_alpha: float = 0.5,
        power_iters: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, top_k=top_k, dampen_alpha=dampen_alpha,
            power_iters=power_iters, eps=eps, nesterov=nesterov,
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
            top_k = group["top_k"]
            dampen_alpha = group["dampen_alpha"]
            power_iters = group["power_iters"]
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
                    # Persistent subspace tracking (n × k)
                    k = min(top_k, min(m_dim, n_dim))
                    state["V"] = torch.randn(n_dim, k)
                    state["V"], _ = torch.linalg.qr(state["V"])

                state["step"] += 1

                # Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                k = state["V"].shape[1]

                # Subspace iteration to track top-k right singular vectors
                V = state["V"]  # (n, k)
                for _ in range(power_iters):
                    # u @ V → (m, k), then u.T @ result → (n, k)
                    AV = u @ V  # project: (m, k)
                    V = u.t() @ AV  # back-project: (n, k)
                    V, _ = torch.linalg.qr(V)  # orthogonalize
                state["V"] = V

                # Compute top-k singular values and left vectors
                AV = u @ V  # (m, k)
                # SVD of the small m×k projection to get accurate SVs
                U_small, S_small, Vh_small = torch.linalg.svd(AV, full_matrices=False)
                # U_small: (m, k), S_small: (k,), Vh_small: (k, k)
                # Right singular vectors in original space
                V_rot = V @ Vh_small.t()  # (n, k)

                # Spectral dampening: σ → σ^α for top-k
                # Compute the dampened reconstruction
                S_dampened = S_small.pow(dampen_alpha)

                # Original top-k contribution: U @ diag(S) @ V.T
                # Dampened top-k contribution: U @ diag(S^α) @ V.T
                # Delta: U @ diag(S^α - S) @ V.T
                # Update = u + delta = u with top-k SVs replaced by dampened versions
                delta_S = S_dampened - S_small  # negative (dampening reduces)
                update = u + U_small @ (delta_S.unsqueeze(1) * V_rot.t())

                # Normalize to match Muon-like update scale
                target_scale = 0.877 * math.sqrt(min(m_dim, n_dim))
                update = update * (target_scale / (update.norm() + eps))

                # Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(update.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr * scale)

        return loss
