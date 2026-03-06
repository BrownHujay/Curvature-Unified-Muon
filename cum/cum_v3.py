import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUMv3(Optimizer):
    """
    CUM v3: Muon + Soft Newton-Schulz + Cautious Masking.

    Key insight: NS orthogonalization equalizes ALL singular values,
    destroying information about which directions matter more. The
    gradient's singular value structure encodes useful curvature info
    (high SV = large activation AND large error = important direction).

    Soft NS: Blend NS output with normalized momentum to partially
    preserve the gradient's singular value structure:
        update = (1 - ns_blend) * NS(u) + ns_blend * normalize(u)
    - ns_blend=0: pure Muon (full orthogonalization)
    - ns_blend>0: partial orthogonalization, preserves some curvature

    Cautious masking: Zero out momentum entries that disagree with
    the current gradient sign before NS, giving NS a cleaner signal.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
        cautious: bool = False,
        ns_blend: float = 0.0,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, eps=eps,
            nesterov=nesterov, cautious=cautious, ns_blend=ns_blend,
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
            cautious = group["cautious"]
            ns_blend = group["ns_blend"]

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

                # Phase 1: Momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Cautious masking (optional)
                # Zero out entries where momentum disagrees with gradient
                if cautious:
                    mask = (u * g > 0).float()
                    # Rescale to preserve norm (so effective LR doesn't change)
                    orig_norm = u.norm()
                    u = u * mask
                    masked_norm = u.norm()
                    if masked_norm > eps:
                        u = u * (orig_norm / masked_norm)

                # Phase 3: NS orthogonalization
                orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                # Phase 4: Soft NS — blend with normalized momentum
                # This preserves some of the gradient's singular value structure
                if ns_blend > 0:
                    orth_norm = orth.norm()
                    u_norm = u.norm()
                    if u_norm > eps:
                        u_normalized = u * (orth_norm / u_norm)
                        orth = (1 - ns_blend) * orth + ns_blend * u_normalized

                # Phase 5: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
