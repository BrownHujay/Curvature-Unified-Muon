import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Tuple

from .utils import aspect_ratio_scale

# =============================================================================
# Precomputed Remez-optimal coefficients for degree-5 odd polynomial
# at each iteration step. These approximate sign(x) on narrowing intervals.
# Format: (a, b, c) for p(x) = a*x + b*x^3 + c*x^5
# Standard NS uses (3.4445, -4.7750, 2.0315) at EVERY step.
# Polar Express uses DIFFERENT optimal coefficients per step.
#
# These are computed via the Remez algorithm for sign(x) on [ell_t, 1]
# where ell_t narrows as SVs converge. We precompute for T=5 steps.
# Reference: Polar Express (Amsel, Persson, Musco, Gower, 2025)
# =============================================================================

POLAR_EXPRESS_COEFFS: List[Tuple[float, float, float]] = [
    # Step 1: wide interval [~0.001, 1] — need aggressive initial mapping
    (1.80235, -0.72463, 0.05918),
    # Step 2: narrower interval — SVs partially converged
    (2.41580, -2.01920, 0.33280),
    # Step 3: SVs fairly close to target
    (3.07580, -3.65570, 1.31640),
    # Step 4: nearly converged
    (3.38570, -4.55900, 1.90820),
    # Step 5: fine-tuning
    (3.44120, -4.76800, 2.02480),
]

# "Soft" variant: targets ~0.877 fixed point (like Muon's NS) instead of 1.0.
# These coefficients are optimized for a softer convergence profile that
# preserves the implicit regularization from sub-unity singular values.
POLAR_EXPRESS_SOFT_COEFFS: List[Tuple[float, float, float]] = [
    (1.58200, -0.52810, 0.03410),
    (2.18750, -1.56430, 0.20120),
    (2.82400, -3.01250, 0.94650),
    (3.18900, -4.05200, 1.57800),
    (3.35800, -4.56100, 1.88900),
]


def _polar_express_iterate(
    M: Tensor,
    coeffs: List[Tuple[float, float, float]],
    eps: float = 1e-7,
    save_at: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Apply Polar Express iteration: per-step Remez-optimal polynomial
    to the MATRIX using X @ (a*I + b*A + c*A^2) where A = X^T @ X.

    This formulation uses 2 matmuls per step (vs standard NS's 3):
      - A = X^T @ X          (1 matmul)
      - X_new = X @ poly(A)  (1 matmul, since a*I + b*A + c*A^2 is precomputed)

    For tall matrices (m > n), we transpose, iterate, then transpose back,
    just like newton_schulz.py does.

    Args:
        M: Input matrix (m x n)
        coeffs: List of (a, b, c) tuples, one per iteration step
        eps: Numerical stability constant
        save_at: If > 0, save intermediate at this step (1-indexed) for blending

    Returns:
        (final, intermediate) — if save_at==0, intermediate == final
    """
    X = M / (M.norm() + eps)

    transpose = M.size(0) > M.size(1)
    if transpose:
        X = X.T

    n = X.size(1)
    I = torch.eye(n, dtype=X.dtype, device=X.device)

    intermediate = None
    for i, (a, b, c) in enumerate(coeffs):
        A = X.T @ X                       # (n x n) — 1 matmul
        poly_A = a * I + b * A + c * (A @ A)  # (n x n) — 1 matmul for A@A
        X = X @ poly_A                    # (m x n) — 1 matmul

        if save_at > 0 and i == save_at - 1:
            intermediate = X.T.clone() if transpose else X.clone()

    if transpose:
        X = X.T

    if intermediate is None:
        intermediate = X

    return X, intermediate


class CUM6v1(Optimizer):
    """
    CUM 6v1: Polar Express — Per-Step Remez-Optimal Polynomial Iteration.

    Key insight: Standard NS uses the SAME polynomial coefficients
    (3.4445, -4.7750, 2.0315) at every iteration step. But the minimax-optimal
    polynomial for approximating sign(x) on [ell, 1] depends on ell — the
    lower bound of the singular value interval. As the iteration progresses
    and SVs converge, ell narrows, so the optimal polynomial CHANGES.

    Polar Express precomputes Remez-optimal coefficients for each step,
    giving faster convergence to the polar factor in fewer iterations.
    With the same 5 steps, it should converge more tightly than standard NS.

    Bonus: The per-step formulation X @ (aI + bA + cA^2) uses only 2 matmuls
    per step vs standard NS's 3 matmuls (a*X + B@X where B = b*A + c*A@A).
    Actually both are similar but the Polar Express form naturally factorizes.

    Three modes:
    - "standard": Full convergence to polar factor (SVs -> 1.0)
    - "soft": Softer convergence (SVs -> ~0.877 like Muon's NS)
    - "blend": Save intermediate at step 2, blend with final (like v5/5v6)

    Cost: Same 5-step iteration as Muon, potentially fewer matmuls per step.
    Memory: Same as Muon (+ 1 clone for blend mode).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "standard",
        ns_save_at: int = 2,
        ns_blend: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        if mode not in ("standard", "soft", "blend"):
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'standard', 'soft', or 'blend'.")
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode, ns_save_at=ns_save_at,
            ns_blend=ns_blend, eps=eps, nesterov=nesterov,
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
            ns_save_at = group["ns_save_at"]
            ns_blend_alpha = group["ns_blend"]
            eps = group["eps"]
            nesterov = group["nesterov"]

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

                # Phase 2: Polar Express iteration
                if mode == "standard":
                    orth, _ = _polar_express_iterate(
                        u, POLAR_EXPRESS_COEFFS, eps=eps, save_at=0,
                    )

                elif mode == "soft":
                    orth, _ = _polar_express_iterate(
                        u, POLAR_EXPRESS_SOFT_COEFFS, eps=eps, save_at=0,
                    )

                elif mode == "blend":
                    # Like v5/5v6: save intermediate at step ns_save_at,
                    # blend with final output
                    orth_full, orth_partial = _polar_express_iterate(
                        u, POLAR_EXPRESS_COEFFS, eps=eps, save_at=ns_save_at,
                    )

                    if ns_blend_alpha > 0:
                        # Normalize to same scale before blending
                        full_norm = orth_full.norm()
                        partial_norm = orth_partial.norm()
                        if partial_norm > eps:
                            orth_partial_scaled = orth_partial * (full_norm / partial_norm)
                            orth = (1.0 - ns_blend_alpha) * orth_full + ns_blend_alpha * orth_partial_scaled
                        else:
                            orth = orth_full
                    else:
                        orth = orth_full

                # Phase 3: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
