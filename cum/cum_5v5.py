import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


# NS polynomial coefficients (from Muon/Zetta)
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315


def _ns_polynomial_scalar(sigma: Tensor, steps: int = 5) -> Tensor:
    """
    Apply NS's polynomial p(σ) = aσ + bσ³ + cσ⁵ iteratively to scalar SVs.

    This should give EXACTLY the same SV mapping as matrix-level NS iteration,
    since NS preserves singular vectors and only maps singular values.
    """
    s = sigma.clone()
    for _ in range(steps):
        s2 = s * s
        s = _NS_A * s + _NS_B * (s * s2) + _NS_C * (s * s2 * s2)
    return s


class CUM5v5(Optimizer):
    """
    CUM 5v5: SVD + NS-Exact Scalar Polynomial (Diagnostic).

    This optimizer computes the SVD, applies NS's EXACT polynomial
    p(σ) = 3.4445σ - 4.775σ³ + 2.0315σ⁵ iteratively to each singular
    value (5 iterations), then reconstructs.

    If NS = scalar SV mapping (which matrix theory says it should be),
    this should give EXACTLY the same result as standard NS iteration.
    Any difference reveals non-scalar effects.

    This is a DIAGNOSTIC optimizer, not intended to beat Muon.
    It verifies our understanding of NS.

    Cost: Full SVD per step (~40% slower than NS).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, eps=eps, nesterov=nesterov,
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

                # Apply the same scaling NS uses: X = G / ||G||_F
                # For SVD: ||G||_F = ||S||_2
                frob = S.norm() + eps
                S_scaled = S / frob

                # Apply NS polynomial to each SV (scalar iteration)
                S_ns = _ns_polynomial_scalar(S_scaled, steps=ns_steps)

                # Reconstruct
                orth = U * S_ns.unsqueeze(0) @ Vh

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
