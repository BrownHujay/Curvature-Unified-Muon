import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_multi_resolution
from .utils import aspect_ratio_scale


class CUMv12(Optimizer):
    """
    CUM v12: Gradient Difference Momentum + Multi-Resolution NS.

    Every version so far uses the same momentum scheme as Muon:
      m = β₁*m + (1-β₁)*g       (EMA of gradients)
      u = g + β₁*m               (Nesterov lookahead)

    v12 adds a SECOND momentum buffer that tracks gradient DIFFERENCES:
      m_diff = β_diff * m_diff + (1-β_diff) * (g - g_prev)

    The gradient difference (g_t - g_{t-1}) approximates the directional
    second derivative of the loss — it's a cheap curvature signal from
    the loss landscape itself, not from the NS iteration.

    NS input becomes:
      u = g + β₁*m + α_diff * m_diff

    Why this should work:
    - Standard momentum tracks WHERE the gradient points (first order)
    - Gradient difference tracks HOW the gradient is CHANGING (second order)
    - When gradient accelerates in some direction → positive curvature →
      m_diff amplifies that direction → NS gets a curvature-biased input
    - When gradient decelerates → negative curvature → m_diff dampens →
      NS input focuses on the stable directions

    This is a different curvature source than NS intermediates (v5):
    - v5's curvature comes from the NS iteration's SV structure
    - v12's curvature comes from the loss landscape's temporal structure
    - They capture different information and should compose well

    Combined with v5's multi-resolution blend for maximum curvature recovery.

    Cost: One extra matrix buffer (m_diff) + one extra (g_prev). Two additions.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta_diff: float = 0.9,
        alpha_diff: float = 0.1,
        ns_steps: int = 5,
        ns_save_at: int = 2,
        ns_blend: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            beta_diff: EMA decay for gradient difference momentum.
            alpha_diff: Weight of gradient difference term in NS input.
        """
        defaults = dict(
            lr=lr, beta1=beta1, beta_diff=beta_diff, alpha_diff=alpha_diff,
            ns_steps=ns_steps, ns_save_at=ns_save_at, ns_blend=ns_blend,
            eps=eps, nesterov=nesterov,
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
            beta_diff = group["beta_diff"]
            alpha_diff = group["alpha_diff"]
            ns_steps = group["ns_steps"]
            ns_save_at = group["ns_save_at"]
            ns_blend = group["ns_blend"]
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
                    state["grad_prev"] = torch.zeros_like(g)
                    state["diff_momentum"] = torch.zeros_like(g)

                state["step"] += 1

                # Phase 1: Standard momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                # Phase 2: Gradient difference momentum
                grad_diff = g - state["grad_prev"]
                dm = state["diff_momentum"]
                dm.mul_(beta_diff).add_(grad_diff, alpha=1 - beta_diff)

                # Save current gradient for next step
                state["grad_prev"].copy_(g)

                # Phase 3: Construct NS input
                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Add gradient difference signal (skip step 1 — no valid diff yet)
                if state["step"] > 1 and alpha_diff > 0:
                    u = u + alpha_diff * dm

                # Phase 4: Multi-Resolution NS (v5 style)
                orth_full, orth_partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=ns_save_at, eps=eps,
                )

                # Phase 5: Spectral blend
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

                # Phase 6: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
