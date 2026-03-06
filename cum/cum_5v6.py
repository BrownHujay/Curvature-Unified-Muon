import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


# NS polynomial coefficients (from Muon/Zetta)
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315


def _ns_poly_k(sigma: Tensor, k: int) -> Tensor:
    """Apply NS polynomial k times to scalar SVs."""
    s = sigma.clone()
    for _ in range(k):
        s2 = s * s
        s = _NS_A * s + _NS_B * (s * s2) + _NS_C * (s * s2 * s2)
    return s


class CUM5v6(Optimizer):
    """
    CUM 5v6: SVD + Custom SV Mapping (NS-blend and beyond).

    Since we proved (5v5) that NS ≈ scalar SV mapping, we can replicate
    v5's multi-resolution blend as a pure SVD operation:
        f(σ) = (1-α) * NS₅(σ) + α * NS_k(σ)

    But SVD gives us power that NS matrix iteration doesn't:
    - We can blend ANY two polynomial iterations (not just adjacent steps)
    - We can apply non-polynomial SV mappings
    - We can treat different SVs differently (top vs bottom)

    Mode 'ns_blend': replicates v5 via SVD (diagnostic)
    Mode 'tilt': mostly-flat mapping with slight SV ordering preservation
        f(σ) = NS₅(σ) * (1 + ε * (σ_norm - mean(σ_norm)))
        where σ_norm = σ/σ_max normalized to [0,1]

    Cost: Full SVD per step (~40% slower than NS).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "ns_blend",
        ns_save_at: int = 2,
        ns_blend: float = 0.15,
        tilt_eps: float = 0.1,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode, ns_save_at=ns_save_at,
            ns_blend=ns_blend, tilt_eps=tilt_eps, eps=eps, nesterov=nesterov,
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
            tilt_eps = group["tilt_eps"]
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

                # SVD of momentum (with robustness for ill-conditioned matrices)
                try:
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                except torch._C._LinAlgError:
                    # Add small noise to break degeneracies
                    u = u + eps * torch.randn_like(u)
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                # Scale SVs the same way NS does: divide by Frobenius norm
                frob = S.norm() + eps
                S_scaled = S / frob

                if mode == "ns_blend":
                    # Replicate v5: blend NS_k intermediate with NS_5 final
                    S_intermediate = _ns_poly_k(S_scaled, ns_save_at)
                    S_final = _ns_poly_k(S_scaled, 5)
                    S_out = (1 - ns_blend_alpha) * S_final + ns_blend_alpha * S_intermediate

                elif mode == "tilt":
                    # Mostly-flat NS mapping with slight original SV ordering
                    S_ns = _ns_poly_k(S_scaled, 5)
                    # Compute normalized original SV structure
                    s_max = S_scaled[0] + eps
                    s_norm = S_scaled / s_max  # [0, 1]
                    s_mean = s_norm.mean()
                    # Add tilt: scale NS output based on original SV position
                    S_out = S_ns * (1 + tilt_eps * (s_norm - s_mean))

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Reconstruct
                orth = U * S_out.unsqueeze(0) @ Vh

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
