import torch
import math
from torch import Tensor


def aspect_ratio_scale(m: int, n: int) -> float:
    """Muon's aspect ratio scaling factor: sqrt(max(1, m/n))."""
    return math.sqrt(max(1, m / n))


def ns_convergence_error(X: Tensor) -> float:
    """
    Measure how close X is to being a scaled orthogonal matrix.

    The NS iteration with Muon/Zetta coefficients converges singular values
    to a common fixed point (~0.877), NOT to 1.0. So we measure the singular
    value spread (σ_max/σ_min - 1) instead of ||XX^T - I||.

    Returns:
        σ_max/σ_min - 1.0 (0.0 = perfect convergence, >0 = spread remaining)
    """
    svs = torch.linalg.svdvals(X)
    return (svs[0] / (svs[-1] + 1e-10) - 1.0).item()


def sv_spread(M: Tensor) -> float:
    """Compute σ_max / σ_min ratio (condition number) of a matrix."""
    svs = torch.linalg.svdvals(M)
    return (svs[0] / (svs[-1] + 1e-10)).item()
