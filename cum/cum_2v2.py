import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM2v2(Optimizer):
    """
    CUM 2v2: QR Orthogonalization (No Newton-Schulz).

    Replaces NS with QR decomposition for orthogonalization.

    For an (m, n) matrix with m <= n:
      QR(G) gives Q (m, m) orthogonal and R (m, n) upper triangular.
      We use Q @ sign(R_diag) as the "orthogonal direction" and
      reconstruct to (m, n) via Q @ D @ I_{m,n} where D corrects signs.

    Actually, the simplest correct approach:
      Use the polar decomposition via SVD for correctness, but with
      torch.linalg.svd (low-rank) instead of NS iteration.
      U, S, Vh = svd(G) → polar factor = U @ Vh

    This gives us the EXACT polar factor (what NS approximates) in one
    shot. More expensive per call but no iteration. And we can selectively
    preserve some singular values by using U @ diag(f(S)) @ Vh instead
    of U @ Vh.

    Algorithm:
      1. u = Nesterov momentum
      2. U, S, Vh = SVD(u)  (or truncated SVD for speed)
      3. orth = U @ Vh  (exact polar factor — what NS₅ approximates)
      4. Optionally: orth = U @ diag(S^alpha) @ Vh  (partial SV preservation)
         alpha=0 → polar factor (equal SVs), alpha=1 → original gradient

    Cost: One SVD. For (128, 512): ~2-5ms. Comparable to 5 NS steps.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        sv_alpha: float = 0.0,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            sv_alpha: How much original SV structure to preserve.
                      0.0 = exact polar factor (like NS₅)
                      0.1 = slight curvature preservation
                      1.0 = original gradient (just direction-normalized)
        """
        defaults = dict(
            lr=lr, beta1=beta1, sv_alpha=sv_alpha, eps=eps, nesterov=nesterov,
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
            sv_alpha = group["sv_alpha"]
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

                # Phase 2: SVD-based orthogonalization
                U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                # U: (m, k), S: (k,), Vh: (k, n) where k = min(m, n)

                if sv_alpha > 0:
                    # Partial SV preservation: S^alpha interpolates between
                    # equal SVs (alpha=0) and original SVs (alpha=1)
                    S_modified = S.pow(sv_alpha)
                    S_modified = S_modified / (S_modified.mean() + eps)  # normalize
                    orth = U * S_modified.unsqueeze(0) @ Vh
                else:
                    # Pure polar factor: U @ Vh (exact, no approximation)
                    orth = U @ Vh

                # Phase 3: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
