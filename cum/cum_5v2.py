import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM5v2(Optimizer):
    """
    CUM 5v2: Tunable Soft Equalization via SVD.

    Core idea: NS maps ALL singular values to ~0.877 (hard equalization).
    But maybe full equalization is too aggressive. V5's success shows that
    blending in partial equalization (NS step 2) helps — suggesting the
    optimal spectral filter is BETWEEN identity and full equalization.

    This optimizer applies a tunable sigmoid SV mapping:
        σ → tanh(β * σ/σ_max) / tanh(β)

    Properties of the tanh mapping:
    - β → 0: identity (no processing, like SGD)
    - β ≈ 3: moderate compression (preserves relative ordering)
    - β ≈ 10: strong equalization (near-flat, like NS)
    - β → ∞: hard equalization (all SVs → 1, exact polar factor)

    Unlike NS, this mapping is:
    - Continuously tunable (β is a real number, not integer NS steps)
    - Monotonic (preserves SV ordering exactly)
    - Smooth (no polynomial oscillation near the fixed point)

    Hypothesis: there exists an optimal β ∈ (2, 10) that outperforms
    both SGD (β=0) and NS (β≈∞) by preserving the right amount of
    spectral structure.

    Cost: Full SVD per step (~40% slower than NS at our scale).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        eq_beta: float = 5.0,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, eq_beta=eq_beta, eps=eps, nesterov=nesterov,
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
            eq_beta = group["eq_beta"]
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

                # SVD of momentum
                U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                # Soft equalization: σ → tanh(β * σ/σ_max) / tanh(β)
                s_max = S[0] + eps  # S is sorted descending
                s_normalized = S / s_max  # [0, 1]

                # Apply tanh mapping
                tanh_beta = math.tanh(eq_beta)
                S_mapped = torch.tanh(eq_beta * s_normalized) / tanh_beta

                # Scale to match NS-like output magnitude
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
