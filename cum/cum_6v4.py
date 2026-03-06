import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM6v4(Optimizer):
    """
    CUM 6v4: Dion — Amortized power iteration orthonormalization.

    Instead of NS's 5 iterations of degree-5 polynomial (15 matmuls),
    uses warm-started rank-r power iteration + QR + small SVD.

    At full rank, this gives EXACT orthonormalization (vs NS's approximation).
    At reduced rank (r < min(m,n)), it's much cheaper while keeping most quality.

    Pipeline:
        1. Standard Nesterov momentum (same as Muon)
        2. Power iteration: Z = (u @ u.T)^iters @ Z  (warm-started)
        3. QR factorization: Q, R = qr(Z)  → m x r orthonormal basis
        4. Project + small SVD: B = Q.T @ u, then svd(B)  → r x n
        5. Reconstruct: update = Q @ Ub @ Vhb  → m x n rank-r orthonormal
        6. Warm-start Z from Q @ Ub for next step

    At r = min(m,n), this recovers the exact polar factor U*Vh from full SVD.
    At r < min(m,n), it's a rank-r approximation that converges quickly
    due to warm-starting — typically 1-2 power iters suffice after step 1.

    Reference: Dion (Ahn, Xu et al., Microsoft, 2025)

    Cost:
        Power iteration: 2 * power_iters matmuls of m×n @ n×r = O(mnr * power_iters)
        QR: O(mr^2)
        Small SVD: O(r^2 n) for r×n matrix
        Total: O(mnr * power_iters + mr^2 + r^2 n)
        Full rank r=min(m,n): similar to full SVD
        Reduced rank r=min(m,n)/4: ~4x cheaper than NS's 15 matmuls

    Args:
        lr: Learning rate (default: 0.02)
        beta1: Momentum coefficient (default: 0.95)
        rank: Rank for power iteration. None = full rank = min(m,n).
              r=min(m,n)/4 is a good cheap option.
        power_iters: Number of power iterations per step (1-2 usually enough
                     with warm start, more on first step)
        nesterov: Use Nesterov momentum (default: True)
        eps: Epsilon for numerical stability (default: 1e-7)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        rank=None,
        power_iters: int = 2,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, rank=rank, power_iters=power_iters,
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
            rank = group["rank"]
            power_iters = group["power_iters"]
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

                # Handle tall matrices by transposing so m <= n
                # (power iteration works on the left singular subspace,
                #  which should be the smaller dimension)
                transposed = m_dim > n_dim
                if transposed:
                    g = g.T
                    m_dim, n_dim = n_dim, m_dim

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["Z"] = None  # warm-start matrix, initialized lazily

                state["step"] += 1

                # --- Determine rank ---
                r = rank if rank is not None else min(m_dim, n_dim)
                r = min(r, min(m_dim, n_dim))  # clamp to valid range

                # --- Standard momentum (same as Muon) ---
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # --- Power iteration (warm-started) ---
                Z = state["Z"]

                # Initialize or re-initialize Z if rank changed
                if Z is None or Z.shape[1] != r:
                    Z = torch.randn(m_dim, r, dtype=u.dtype, device=u.device)
                    # Extra power iterations on first step (no warm start)
                    n_iters = max(power_iters, 5)
                else:
                    n_iters = power_iters

                # Subspace iteration: Z <- (u @ u.T)^iters @ Z
                # Written as alternating u @ (u.T @ Z) to avoid forming m×m matrix
                for _ in range(n_iters):
                    Z = u @ (u.T @ Z)  # m×n @ n×r = m×r, then m×m @ m×r = m×r
                    # Periodic re-orthogonalization for numerical stability
                    # (cheap since Z is m×r with r potentially small)
                    Z_norm = Z.norm(dim=0, keepdim=True) + eps
                    Z = Z / Z_norm

                # --- QR factorization: orthonormal basis for column space ---
                Q, R = torch.linalg.qr(Z)  # Q: m×r, R: r×r

                # --- Project into subspace + small SVD ---
                B = Q.T @ u  # r×n: momentum projected into rank-r subspace

                try:
                    Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)  # r×r, r, r×n
                except torch._C._LinAlgError:
                    B = B + eps * torch.randn_like(B)
                    Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)

                # --- Reconstruct orthonormal update ---
                # Q @ Ub gives m×r left singular vectors of the rank-r approx
                # Vhb gives r×n right singular vectors
                # Together: (m×r) @ (r×n) = m×n rank-r orthonormal matrix
                update = (Q @ Ub) @ Vhb  # m×n

                # --- Warm-start Z for next step ---
                # Use Q @ Ub as the warm start: these are the current left
                # singular vectors, which will be close to next step's
                state["Z"] = Q @ Ub  # m×r

                # --- Handle transpose ---
                if transposed:
                    update = update.T
                    m_dim, n_dim = n_dim, m_dim

                # --- Apply update ---
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != update.shape:
                    p.data.add_(update.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr * scale)

        return loss
