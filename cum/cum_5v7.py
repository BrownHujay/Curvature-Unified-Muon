import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


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


class CUM5v7(Optimizer):
    """
    CUM 5v7: Split-Spectrum NS — Different Iteration Counts per SV.

    Novel idea: instead of applying the same number of NS polynomial
    iterations to ALL singular values, split the spectrum and apply
    different iteration counts to different SV ranges.

    Mode 'top_preserve':
        Top-k SVs: fewer NS iterations (preserve gradient direction info)
        Bottom SVs: full NS iterations (equalize noise)

    Mode 'adaptive_per_sv':
        Each SV gets iterations proportional to how far it is from the
        target (0.877). SVs already near 0.877 get fewer iterations.

    Mode 'schedule':
        Blend parameter changes over training via cosine schedule.
        Early: high blend (more curvature preserved)
        Late: low blend (more equalization for convergence)

    Cost: Full SVD + scalar polynomial evaluation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "top_preserve",
        ns_top_steps: int = 3,
        ns_bot_steps: int = 5,
        split_frac: float = 0.25,
        blend_start: float = 0.35,
        blend_end: float = 0.10,
        total_steps: int = 2000,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode,
            ns_top_steps=ns_top_steps, ns_bot_steps=ns_bot_steps,
            split_frac=split_frac,
            blend_start=blend_start, blend_end=blend_end,
            total_steps=total_steps, eps=eps, nesterov=nesterov,
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

                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                try:
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                except torch._C._LinAlgError:
                    u = u + eps * torch.randn_like(u)
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                frob = S.norm() + eps
                S_scaled = S / frob

                if mode == "top_preserve":
                    # Top-k SVs get fewer iterations (preserve structure)
                    # Bottom SVs get full iterations (equalize)
                    ns_top = group["ns_top_steps"]
                    ns_bot = group["ns_bot_steps"]
                    k = max(1, int(len(S) * group["split_frac"]))

                    S_top = _ns_poly_k(S_scaled[:k], ns_top)
                    S_bot = _ns_poly_k(S_scaled[k:], ns_bot)
                    S_out = torch.cat([S_top, S_bot])

                elif mode == "schedule":
                    # Cosine blend schedule: high blend early, low blend late
                    t = state["step"]
                    total = group["total_steps"]
                    b_start = group["blend_start"]
                    b_end = group["blend_end"]

                    progress = min(t / total, 1.0)
                    blend = b_end + 0.5 * (b_start - b_end) * (1 + math.cos(math.pi * progress))

                    S_ns2 = _ns_poly_k(S_scaled, 2)
                    S_ns5 = _ns_poly_k(S_scaled, 5)
                    S_out = (1 - blend) * S_ns5 + blend * S_ns2

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                orth = U * S_out.unsqueeze(0) @ Vh

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
