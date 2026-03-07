import torch
from torch import Tensor
from typing import Tuple, Optional


def newton_schulz_orthogonalize(G: Tensor, steps: int = 3, eps: float = 1e-7) -> Tensor:
    """
    Compute approximate polar factor of G via Newton-Schulz iteration.

    Args:
        G: Input matrix (m x n), should be pre-normalized
        steps: Number of NS iterations (CUM default: 3)
        eps: Numerical stability

    Returns:
        Approximate polar factor (m x n) with singular values ≈ 1
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X


def newton_schulz_multi_resolution(
    G: Tensor,
    steps: int = 5,
    save_at: int = 2,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor]:
    """
    NS iteration that returns BOTH the fully-converged output AND a
    partially-converged intermediate from step `save_at`.

    The intermediate retains significant singular value structure (curvature
    info) while being partially denoised by the initial NS steps.

    Cost: Same as standard NS (just saves one intermediate — no extra matmuls).
    Memory: One extra matrix clone at the save point.

    Returns:
        (fully_converged, partially_converged) — both (m x n)
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    intermediate = None
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        if i == save_at - 1:
            intermediate = X.T.clone() if transpose else X.clone()

    if transpose:
        X = X.T

    if intermediate is None:
        intermediate = X  # save_at >= steps, no intermediate

    return X, intermediate


def newton_schulz_triple_resolution(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    NS iteration that returns intermediates at steps 1, 3, and the final (step 5).

    Returns three resolution levels:
      - NS₁: 1 step — heavily curvature-rich, partially noisy
      - NS₃: 3 steps — moderate curvature, well-denoised
      - NS₅: 5 steps — fully orthogonalized, no curvature

    Cost: Same matmuls as standard NS. Two extra clones.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    ns1 = None
    ns3 = None
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        if i == 0:
            ns1 = X.T.clone() if transpose else X.clone()
        if i == 2:
            ns3 = X.T.clone() if transpose else X.clone()

    if transpose:
        X = X.T

    if ns1 is None:
        ns1 = X
    if ns3 is None:
        ns3 = X

    return X, ns3, ns1  # (full, mid, early)


def newton_schulz_n_resolution(
    G: Tensor,
    steps: int = 5,
    save_at: Tuple[int, ...] = (1, 3),
    eps: float = 1e-7,
) -> Tuple[Tensor, ...]:
    """
    Generalized NS iteration that saves intermediates at arbitrary steps.

    Args:
        G: Input matrix (m x n)
        steps: Total NS iterations
        save_at: Tuple of step numbers (1-indexed) at which to save intermediates
        eps: Numerical stability

    Returns:
        (final, intermediate_1, intermediate_2, ...) in order of save_at
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    save_set = set(save_at)
    intermediates = {}
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        if (i + 1) in save_set:  # 1-indexed: step 1 means after first iteration
            intermediates[i + 1] = X.T.clone() if transpose else X.clone()

    if transpose:
        X = X.T

    # Return (final, *intermediates in save_at order)
    result = [X]
    for s in save_at:
        result.append(intermediates.get(s, X))
    return tuple(result)


def newton_schulz_dampened(
    G: Tensor,
    steps: int = 5,
    dampen_after: int = 2,
    dampen_factor: float = 0.5,
    eps: float = 1e-7,
) -> Tensor:
    """
    NS iteration with dampened late-stage steps.

    Standard NS converges aggressively — all singular values equalize by step 5.
    This variant runs standard NS for `dampen_after` steps, then interpolates
    each subsequent NS step with the identity (slowing convergence).

    The result: singular values are PARTIALLY equalized. Large SVs are brought
    down but not to equality with small SVs. This preserves curvature info
    in the final output WITHOUT needing a separate blend step.

    X_{k+1} = (1-d) * NS_step(X_k) + d * X_k   for k >= dampen_after

    Args:
        dampen_after: Step at which to start dampening (0-indexed, default=2)
        dampen_factor: Interpolation weight toward identity (0=standard NS, 1=freeze)
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X_next = a * X + B @ X

        if i >= dampen_after:
            # Interpolate between NS step and current (slow convergence)
            X = (1.0 - dampen_factor) * X_next + dampen_factor * X
        else:
            X = X_next

    if transpose:
        X = X.T

    return X


def newton_schulz_dampened_multi_resolution(
    G: Tensor,
    steps: int = 5,
    save_at: int = 2,
    dampen_after: int = 2,
    dampen_factor: float = 0.3,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor]:
    """
    Combines dampened late-stage NS with multi-resolution intermediate save.

    Steps 1..save_at: standard NS (fast cleanup), saves intermediate at save_at.
    Steps save_at+1..steps: dampened NS (preserve curvature in final output).

    The final output retains more curvature than standard NS₅ (thanks to dampening),
    and we also get the NS₂ intermediate for blending. Both ends of the blend
    are richer in curvature info.

    Returns:
        (dampened_final, intermediate) — both (m x n)
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    intermediate = None
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X_next = a * X + B @ X

        if i >= dampen_after:
            X = (1.0 - dampen_factor) * X_next + dampen_factor * X
        else:
            X = X_next

        if i == save_at - 1:
            intermediate = X.T.clone() if transpose else X.clone()

    if transpose:
        X = X.T

    if intermediate is None:
        intermediate = X

    return X, intermediate
