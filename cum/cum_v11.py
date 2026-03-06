import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_multi_resolution
from .utils import aspect_ratio_scale


class CUMv11(Optimizer):
    """
    CUM v11: Second-Moment Grafting onto Multi-Resolution NS.

    Every approach so far modified the NS step or blended NS intermediates.
    v11 takes a completely different angle: use Adam's second-moment estimate
    to REWEIGHT the NS output element-wise.

    NS gives us the best possible DIRECTION (all SVs equalized).
    Adam's second moment gives us the best possible per-element MAGNITUDE.
    Grafting combines: direction from NS, magnitude from Adam's v_t.

    Formula:
      orth = multi_resolution_NS_blend(u)  (v5 style, best direction)
      v_t = β₂ * v_{t-1} + (1-β₂) * g²  (Adam second moment)
      scale = 1 / (sqrt(v_t) + ε)         (Adam-style scaling)
      scale = scale / scale.mean()         (normalize to preserve NS magnitude)
      update = orth * scale                (graft magnitude onto direction)

    Why this should work:
    - NS equalizes all SVs → all parameters get equal update magnitude
    - But some parameters SHOULD get larger updates (high gradient variance)
    - Adam knows which parameters need larger updates (via second moment)
    - Grafting gives each parameter its "natural" step size while keeping
      NS's superior direction

    The normalization (scale/scale.mean()) is critical: without it, the
    Adam scaling would completely change the update magnitude. With it,
    we preserve NS's overall magnitude while redistributing it based on
    Adam's per-element signal.

    Cost: One extra buffer per weight (second moment, same as Adam's v_t).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.999,
        ns_steps: int = 5,
        ns_save_at: int = 2,
        ns_blend: float = 0.15,
        graft_strength: float = 0.3,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            beta2: EMA decay for second moment estimate.
            graft_strength: How much to apply the Adam scaling (0=pure NS, 1=full graft).
                           Interpolates between uniform NS magnitude and Adam-reweighted.
        """
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, ns_steps=ns_steps,
            ns_save_at=ns_save_at, ns_blend=ns_blend,
            graft_strength=graft_strength, eps=eps, nesterov=nesterov,
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
            beta2 = group["beta2"]
            ns_steps = group["ns_steps"]
            ns_save_at = group["ns_save_at"]
            ns_blend = group["ns_blend"]
            graft_strength = group["graft_strength"]
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
                    state["v"] = torch.zeros_like(g)

                state["step"] += 1

                # Phase 1: Momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Multi-Resolution NS (v5 style)
                orth_full, orth_partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=ns_save_at, eps=eps,
                )

                # Phase 3: Spectral blend (v5 style)
                if ns_blend > 0:
                    full_norm = orth_full.norm()
                    partial_norm = orth_partial.norm()
                    if partial_norm > eps:
                        orth_partial_scaled = orth_partial * (full_norm / partial_norm)
                        orth = (1.0 - ns_blend) * orth_full + ns_blend * orth_partial_scaled
                    else:
                        orth = orth_full
                else:
                    orth = orth_full

                # Phase 4: Second-moment grafting
                if graft_strength > 0:
                    # Update second moment estimate (Adam-style)
                    v = state["v"]
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    # Bias correction
                    v_corrected = v / (1 - beta2 ** state["step"])

                    # Adam-style per-element scaling
                    adam_scale = 1.0 / (v_corrected.sqrt() + eps)

                    # Normalize: preserve NS's overall magnitude
                    adam_scale = adam_scale / (adam_scale.mean() + eps)

                    # Interpolate between uniform (NS) and Adam-reweighted
                    effective_scale = (1.0 - graft_strength) + graft_strength * adam_scale

                    orth = orth * effective_scale

                # Phase 5: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
