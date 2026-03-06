import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


def _halley_polar(
    M: Tensor,
    iters: int = 3,
    eps: float = 1e-7,
) -> Tensor:
    """
    Compute the polar factor of M using Halley's iteration.

    Halley's method for the matrix sign function has CUBIC convergence:
        X_{k+1} = X_k @ (3I + A) @ inv(I + 3A)
    where A = X_k^T @ X_k (the Gram matrix).

    This converges in 3 iterations to roughly the same quality as NS in 5,
    because cubic^3 = O(eps^27) vs quadratic^5 = O(eps^32).

    For rectangular m x n matrices (m >= n), we work with the thin form:
    A is n x n, so the solve is on an n x n system — O(n^3).
    For our 128x512 matrices after transpose: n=128, so ~2M FLOPs per solve.

    Reference: Halley's iteration is a degree-(2,1) Padé approximant to
    the matrix sign function. See Nakatsukasa & Freund (2016) ZOLO-PD
    for the broader family of rational iterations.

    Args:
        M: Input matrix (m x n)
        iters: Number of Halley iterations (3 recommended)
        eps: Numerical stability constant

    Returns:
        Approximate polar factor of M
    """
    X = M / (M.norm() + eps)

    # Work with the smaller dimension: if m > n, transpose so we have
    # a wide matrix and the Gram matrix is n x n (smaller).
    transpose = M.size(0) > M.size(1)
    if transpose:
        X = X.T

    n = X.size(1)
    I = torch.eye(n, dtype=X.dtype, device=X.device)

    for _ in range(iters):
        A = X.T @ X  # n x n Gram matrix

        # Halley iteration: X_{k+1} = X_k @ (3I + A) @ (I + 3A)^{-1}
        #
        # For numerical stability, instead of computing the explicit inverse,
        # we solve the linear system: (I + 3A) @ Y = (3I + A)
        # Then X_{k+1} = X_k @ Y
        numerator = 3.0 * I + A
        denominator = I + 3.0 * A

        # Solve denominator @ Y = numerator  =>  Y = denominator^{-1} @ numerator
        # torch.linalg.solve(B, A) solves B @ X = A, so returns B^{-1} @ A
        try:
            Y = torch.linalg.solve(denominator, numerator)
        except torch._C._LinAlgError:
            # If singular (shouldn't happen with I + 3A), add regularization
            denominator = denominator + eps * I
            Y = torch.linalg.solve(denominator, numerator)

        X = X @ Y

    if transpose:
        X = X.T

    return X


def _qdwh_polar(
    M: Tensor,
    iters: int = 3,
    eps: float = 1e-7,
) -> Tensor:
    """
    Compute the polar factor of M using the QR-based Dynamically Weighted
    Halley (QDWH) iteration.

    QDWH adaptively chooses between:
    - The QR-based "Cholesky" iteration for well-conditioned cases
    - Halley's iteration with dynamically computed optimal weights

    For simplicity, we implement the core QDWH inner iteration:
        X_{k+1} = (a_k * X_k + b_k * X_k @ inv(X_k^T @ X_k)) / c_k

    where a_k, b_k, c_k are optimally chosen based on an estimate of the
    smallest singular value ell_k, updated each iteration.

    The key insight is that QDWH guarantees convergence in at most 6 iterations
    for ANY condition number, and typically 2-3 for moderate condition numbers.

    Reference: Nakatsukasa, Bai, Gygi (2010) "Optimizing Halley's iteration
    for computing the matrix polar decomposition"

    Args:
        M: Input matrix (m x n)
        iters: Number of QDWH iterations (2-3 recommended)
        eps: Numerical stability constant

    Returns:
        Approximate polar factor of M
    """
    norm_M = M.norm() + eps
    X = M / norm_M

    transpose = M.size(0) > M.size(1)
    if transpose:
        X = X.T

    n = X.size(1)
    I = torch.eye(n, dtype=X.dtype, device=X.device)

    # Estimate the smallest SV of X (after normalization, largest ≈ 1)
    # Use a cheap estimate: ell ≈ norm(X, 'fro')^2 / (n * norm(X, 2))
    # Since we normalized by Frobenius, we use a conservative starting estimate
    ell = 1.0 / (X.shape[0] * X.shape[1]) ** 0.25  # conservative lower bound

    for _ in range(iters):
        # Compute optimal QDWH parameters from current ell estimate
        # From Nakatsukasa et al.: solve d^2 + (4(1-ell^2)/ell^4) * d - 1 = 0
        ell2 = ell * ell
        dd = (4.0 * (1.0 - ell2) / (ell2 * ell2))

        # Quadratic formula: d = (-dd_coeff + sqrt(dd_coeff^2 + 4)) / 2
        # where the equation is d^2 + dd*d - 1 = 0  =>  d = (-dd + sqrt(dd^2+4))/2
        sqrt_disc = math.sqrt(dd * dd + 4.0)
        d = (-dd + sqrt_disc) / 2.0
        d = max(d, eps)  # ensure positive

        # QDWH weights
        sqrt_d = math.sqrt(d)
        a = sqrt_d / ell + 1.0 / (ell * sqrt_d)
        b = (sqrt_d - 1.0 / sqrt_d) / ell
        c = a + b

        # Clamp to avoid numerical issues
        a = max(a, eps)
        b = max(b, eps)
        c = max(c, eps)

        A = X.T @ X  # n x n

        # X_{k+1} = (a * X + b * X @ inv(A)) / c
        # = X @ (a * I + b * inv(A)) / c
        # = X @ (a * A + b * I) @ inv(A) / c
        # More stable: solve (c * A) @ Y = (a * A + b * I)
        lhs = a * A + b * I
        rhs = c * A

        try:
            Y = torch.linalg.solve(rhs, lhs)
        except torch._C._LinAlgError:
            rhs = rhs + eps * I
            Y = torch.linalg.solve(rhs, lhs)

        X = X @ Y

        # Update ell estimate: after one step, new ell = ell * (a + b*ell^2) / c
        ell_new = ell * (a + b * ell2) / c
        ell = min(ell_new, 1.0)  # clamp to [0, 1]

    if transpose:
        X = X.T

    return X


class CUM6v5(Optimizer):
    """
    CUM 6v5: Zolotarev — Rational approximation for fast polar factor.

    Uses rational (Padé-type) iterations for the matrix polar factor, which
    converge EXPONENTIALLY faster than polynomial iterations (NS).

    The key insight: Newton-Schulz is a degree-5 polynomial iteration with
    roughly quadratic convergence. Rational iterations like Halley's method
    achieve cubic convergence by incorporating a matrix inverse/solve step.
    3 iterations of Halley ≈ 5+ iterations of NS in convergence quality.

    For our 128x512 matrices, after transposing to work with the smaller
    dimension (n=128), each solve costs O(128^3) ≈ 2M FLOPs — well within
    budget. The total cost is comparable to NS since we need fewer iterations.

    Modes:
    - "halley": Halley's method (cubic convergence, 3 iters)
        X_{k+1} = X_k @ (3I + A) @ inv(I + 3A)  where A = X_k^T @ X_k
        Clean, simple, no hyperparameters beyond iteration count.
    - "qdwh": QR-based Dynamically Weighted Halley (adaptive, 2-3 iters)
        Optimal rational iteration that adapts weights based on estimated
        condition number. Fastest known convergence for polar decomposition.

    State per parameter: momentum_buffer only (no extra state needed).

    Cost: ~same as NS for 128x512 matrices (fewer iters offset by solve cost).
    Memory: Same as Muon (momentum buffer only).

    Reference: Nakatsukasa & Freund (2016) "Computing fundamental matrix
    decompositions accurately via the matrix sign function in two iterations:
    The power of Zolotarev's functions"
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "halley",
        iters: int = 3,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        if mode not in ("halley", "qdwh"):
            raise ValueError(
                f"Unknown mode: {mode!r}. Choose 'halley' or 'qdwh'."
            )
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode, iters=iters,
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
            mode = group["mode"]
            iters = group["iters"]
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

                state["step"] += 1

                # Phase 1: Momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Rational polar iteration
                if mode == "halley":
                    orth = _halley_polar(u, iters=iters, eps=eps)
                elif mode == "qdwh":
                    orth = _qdwh_polar(u, iters=iters, eps=eps)

                # Phase 3: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
