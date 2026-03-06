import torch
from torch import Tensor
from typing import Tuple


def apply_factored_precond(
    u: Tensor,          # momentum (m x n)
    g: Tensor,          # current gradient (m x n)
    row_var: Tensor,    # running row variance (m,)
    col_var: Tensor,    # running column variance (n,)
    beta2: float,
    step: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Update factored second moments and apply preconditioning.

    Returns:
        (preconditioned_u, updated_row_var, updated_col_var)
    """
    # Update EMAs (in-place for memory)
    g_sq = g * g
    row_var.mul_(beta2).add_(g_sq.sum(dim=1), alpha=1 - beta2)
    col_var.mul_(beta2).add_(g_sq.sum(dim=0), alpha=1 - beta2)

    # Bias correction
    bc = 1.0 - beta2 ** step
    row_scale = 1.0 / (row_var.div(bc).sqrt() + eps)  # (m,)
    col_scale = 1.0 / (col_var.div(bc).sqrt() + eps)  # (n,)

    # Apply as outer product scaling: O(mn)
    preconditioned = u * row_scale[:, None] * col_scale[None, :]

    return preconditioned, row_var, col_var
