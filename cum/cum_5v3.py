import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM5v3(Optimizer):
    """
    CUM 5v3: Schatten-p Steepest Descent.

    Key insight from deep research: NS approximates steepest descent on the
    unit Schatten-32 ball. The SV mapping for Schatten-p steepest descent is:

        σ → σ^{1/(p-1)} / normalization

    Properties of the power mapping σ^α where α = 1/(p-1):
    - p=2 (α=1): identity (SGD, no equalization)
    - p=4 (α=1/3): moderate equalization
    - p=8 (α=1/7): strong equalization
    - p=16 (α=1/15): very strong (approaching NS)
    - p=32 (α=1/31): near-NS
    - p=∞ (α=0): full equalization (polar factor)

    vs 5v2's tanh mapping: the power function lifts small SVs MUCH more.
    Example with σ=0.01:
      tanh β=7: 0.01 → 0.07 (barely lifted)
      power p=8: 0.01 → 0.52 (strongly lifted!)

    This is the correct mathematical framework for exploring the
    equalization-vs-identity continuum.

    Cost: Full SVD per step (~40% slower than NS at our scale).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        schatten_p: float = 16.0,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, schatten_p=schatten_p, eps=eps, nesterov=nesterov,
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
            schatten_p = group["schatten_p"]
            eps = group["eps"]
            nesterov = group["nesterov"]

            # SV mapping exponent: α = 1/(p-1)
            alpha = 1.0 / (schatten_p - 1.0)

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

                # Standard momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # SVD of momentum (with robustness)
                try:
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                except torch._C._LinAlgError:
                    u = u + eps * torch.randn_like(u)
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                # Schatten-p SV mapping: σ → σ^α where α = 1/(p-1)
                # Normalize SVs to [0,1] first to avoid numerical issues
                s_max = S[0] + eps
                s_normalized = S / s_max  # [0, 1]

                # Apply power mapping
                S_mapped = s_normalized.pow(alpha)

                # Scale to match NS output magnitude
                # NS output has ||X||_F ≈ 0.877 * sqrt(min(m,n))
                target_scale = 0.877 * math.sqrt(min(m_dim, n_dim))
                current_scale = S_mapped.norm() + eps
                S_final = S_mapped * (target_scale / current_scale)

                # Reconstruct
                orth = U * S_final.unsqueeze(0) @ Vh

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
