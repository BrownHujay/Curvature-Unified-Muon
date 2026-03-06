import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale
from .newton_schulz import newton_schulz_orthogonalize, newton_schulz_multi_resolution


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


class CUM6v2(Optimizer):
    """
    CUM 6v2: PolarGrad — Nuclear norm scaled polar factor update.

    Standard Muon computes:
        W -= lr * polar(G)

    This lacks "null-gradient consistency": when gradients are small, the
    polar factor still has SVs near 1, so the update magnitude doesn't
    shrink. PolarGrad fixes this by scaling the update by the nuclear norm:

        W -= lr * ||G||_* * msign(G)

    where ||G||_* = sum of singular values (nuclear norm) and
    msign(G) = polar factor (approximate or exact).

    This ensures that small gradients produce proportionally small updates,
    while preserving the directional structure of the polar factor.

    Reference: PolarGrad (Lau, Long, Su, 2025)

    Modes:
    - "ns": Use Newton-Schulz for polar factor, torch.linalg.svdvals for
            exact nuclear norm. Cheapest option that still gets exact norm.
    - "svd": Full SVD for exact polar factor + exact nuclear norm.
    - "ns_blend": Combine with 5v6-style NS_k/NS_5 blend via SVD,
                  plus nuclear norm scaling. Most expressive mode.

    Cost:
    - "ns": NS iteration + one svdvals call (~20% over Muon)
    - "svd": Full SVD per step (~40% over Muon)
    - "ns_blend": Full SVD per step (~40% over Muon)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "ns",
        ns_steps: int = 5,
        norm_scale: float = 1.0,
        ns_blend: float = 0.15,
        ns_save_at: int = 2,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode, ns_steps=ns_steps,
            norm_scale=norm_scale, ns_blend=ns_blend, ns_save_at=ns_save_at,
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
            ns_steps = group["ns_steps"]
            norm_scale = group["norm_scale"]
            ns_blend_alpha = group["ns_blend"]
            ns_save_at = group["ns_save_at"]
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

                # Phase 1: Standard momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Compute polar factor and nuclear norm
                if mode == "ns":
                    # NS for polar factor, svdvals for exact nuclear norm
                    polar = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                    # Nuclear norm = sum of singular values
                    svs = torch.linalg.svdvals(u)
                    nuclear_norm = svs.sum()

                    # Scale polar factor by nuclear norm
                    update = nuclear_norm * norm_scale * polar

                elif mode == "svd":
                    # Full SVD: exact polar factor + exact nuclear norm
                    try:
                        U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                    except torch._C._LinAlgError:
                        u = u + eps * torch.randn_like(u)
                        U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                    nuclear_norm = S.sum()
                    polar = U @ Vh

                    # Scale polar factor by nuclear norm
                    update = nuclear_norm * norm_scale * polar

                elif mode == "ns_blend":
                    # SVD-based NS blend (like 5v6) + nuclear norm scaling
                    try:
                        U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                    except torch._C._LinAlgError:
                        u = u + eps * torch.randn_like(u)
                        U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                    # Nuclear norm from raw singular values
                    nuclear_norm = S.sum()

                    # Scale SVs the same way NS does: divide by Frobenius norm
                    frob = S.norm() + eps
                    S_scaled = S / frob

                    # Blend NS_k intermediate with NS_5 final (like 5v6)
                    S_intermediate = _ns_poly_k(S_scaled, ns_save_at)
                    S_final = _ns_poly_k(S_scaled, 5)
                    S_out = (1 - ns_blend_alpha) * S_final + ns_blend_alpha * S_intermediate

                    # Reconstruct blended polar-ish factor
                    blended = U * S_out.unsqueeze(0) @ Vh

                    # Scale by nuclear norm
                    update = nuclear_norm * norm_scale * blended

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Phase 3: Weight update with aspect ratio scaling
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(update.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr * scale)

        return loss
