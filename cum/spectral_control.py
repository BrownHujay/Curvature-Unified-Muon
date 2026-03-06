import torch
from torch import Tensor
from typing import Tuple


def spectral_damping(
    W: Tensor,            # weight matrix (m x n)
    v: Tensor,            # power iteration vector (n,)
    sigma_max: float,
    alpha_damp: float,
) -> Tuple[float, Tensor]:
    """
    One step of power iteration + smooth spectral damping.

    Returns:
        (damping_factor, updated_v)
    """
    # One power iteration step: estimate σ_max(W)
    Wv = W.T @ (W @ v)          # (n,) — computes v' = WᵀWv
    v_new = Wv / (Wv.norm() + 1e-7)
    sigma_est = (W @ v_new).norm()  # ≈ σ_max(W)

    # Smooth damping
    excess = max(0.0, sigma_est.item() - sigma_max)
    damping = 1.0 / (1.0 + alpha_damp * excess)

    return damping, v_new
