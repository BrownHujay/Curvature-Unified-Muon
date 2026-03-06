import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUM2v1(Optimizer):
    """
    CUM 2v1: Randomized Top-k Curvature Recovery.

    Completely different approach from v1-v12. Instead of saving NS
    intermediates (v5) or modifying the NS iteration (v9), we directly
    extract the TOP curvature directions from the gradient using
    randomized linear algebra, then blend them with the NS output.

    Why this is better than v5:
    - v5's NS₂ intermediate retains ~25% of SV spread across ALL SVs
      (both signal and noise SVs get partially preserved)
    - 2v1 retains 100% of the top-k SVs and 0% of the rest
    - This is a SHARPER curvature signal — only the dominant directions

    Algorithm:
      1. u = Nesterov momentum (same as Muon)
      2. Extract top-k curvature from u via randomized projection:
         - Y = u @ Omega  (project into random k-dim subspace)
         - Q = orth(Y)    (orthogonalize — approximate top-k left SVecs)
         - curvature = Q @ (Q^T @ u)  (rank-k curvature component)
      3. orth = NS(u, steps=5)  (standard full orthogonalization)
      4. Blend: update = (1-alpha) * orth + alpha * normalize(curvature)

    Cost: One m×k matmul + QR(m×k) + one k×n matmul. With k=4, this is
    ~4x cheaper than one NS step. Total overhead is negligible.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        rank: int = 4,
        curv_blend: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            rank: Number of top singular components to recover (k).
            curv_blend: How much curvature to blend back (0=pure NS, 1=pure curvature).
        """
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps,
            rank=rank, curv_blend=curv_blend, eps=eps, nesterov=nesterov,
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
            rank = group["rank"]
            curv_blend = group["curv_blend"]
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

                # Phase 1: Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Full NS orthogonalization
                orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                # Phase 3: Randomized top-k curvature extraction
                if curv_blend > 0:
                    k = min(rank, min(m_dim, n_dim))

                    # Random projection to find approximate top-k subspace
                    # Use the smaller dimension for efficiency
                    if m_dim <= n_dim:
                        # Project columns: find top-k left singular vectors
                        omega = torch.randn(n_dim, k, device=g.device, dtype=g.dtype)
                        Y = u @ omega  # (m, k)
                        Q, _ = torch.linalg.qr(Y)  # (m, k) orthonormal
                        # Rank-k curvature component
                        curvature = Q @ (Q.T @ u)  # (m, n)
                    else:
                        # Project rows: find top-k right singular vectors
                        omega = torch.randn(m_dim, k, device=g.device, dtype=g.dtype)
                        Y = u.T @ omega  # (n, k)
                        Q, _ = torch.linalg.qr(Y)  # (n, k) orthonormal
                        curvature = (u @ Q) @ Q.T  # (m, n)

                    # Normalize curvature to match orth's scale
                    curv_norm = curvature.norm()
                    orth_norm = orth.norm()
                    if curv_norm > eps:
                        curvature_scaled = curvature * (orth_norm / curv_norm)
                        orth = (1.0 - curv_blend) * orth + curv_blend * curvature_scaled

                # Phase 4: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
