import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_multi_resolution
from .utils import aspect_ratio_scale


class CUMv7(Optimizer):
    """
    CUM v7: Orthogonal Feedback Loop.

    Mathematical innovation: Feed back a fraction of the PREVIOUS step's
    orthogonalized update into the current NS input. This creates temporal
    coherence in the orthogonalized space.

    Standard Muon pipeline:
      u = g + β₁ * momentum
      update = NS(u)

    v7 pipeline:
      u = g + β₁ * momentum + β_feedback * prev_orth
      update = NS(u)  (also blended with NS₂ intermediate à la v5)

    Why this should work:
    - The prev_orth term biases NS toward the previous orthogonalized direction
    - This creates smoother trajectories in the orthogonal optimization space
    - Without feedback, consecutive NS outputs can jump around because NS is
      sensitive to the input's SV structure (which changes every step)
    - With feedback, we get "orthogonal momentum" — persistence in the
      orthogonalized direction, not just the raw gradient direction

    The feedback also composes with multi-resolution NS (v5) for curvature
    preservation.

    Cost: One extra matrix buffer per weight (same size as momentum buffer).
    One extra addition per step.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        ns_save_at: int = 2,
        ns_blend: float = 0.15,
        beta_feedback: float = 0.1,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            ns_blend: Blend factor for multi-resolution NS (v5-style).
            beta_feedback: Weight of previous orth output in NS input.
                           Higher = more temporal smoothing in orth space.
        """
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, ns_save_at=ns_save_at,
            ns_blend=ns_blend, beta_feedback=beta_feedback,
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
            ns_steps = group["ns_steps"]
            ns_save_at = group["ns_save_at"]
            ns_blend = group["ns_blend"]
            beta_feedback = group["beta_feedback"]
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
                    state["prev_orth"] = torch.zeros_like(g)

                state["step"] += 1

                # Phase 1: Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Orthogonal feedback — inject previous orth direction
                if state["step"] > 1 and beta_feedback > 0:
                    u = u + beta_feedback * state["prev_orth"]

                # Phase 3: Multi-Resolution NS (v5-style)
                orth_full, orth_partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=ns_save_at, eps=eps,
                )

                # Phase 4: Spectral blend
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

                # Phase 5: Save for next step's feedback
                state["prev_orth"] = orth.clone()

                # Phase 6: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
