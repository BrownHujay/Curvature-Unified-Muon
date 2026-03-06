import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_dampened
from .utils import aspect_ratio_scale


class CUMv9(Optimizer):
    """
    CUM v9: Dampened Late-Stage NS (Curvature-Preserving Iteration).

    Mathematical innovation: Instead of modifying the NS OUTPUT (post-hoc
    blending like v5/v8), modify the NS ITERATION ITSELF to preserve curvature.

    Standard NS runs 5 identical iteration steps. Each step aggressively
    pushes singular values toward equality. By step 5, virtually all original
    SV spread is destroyed.

    v9 modifies the last 3 steps (after step 2): instead of applying the
    full NS update, it interpolates between the NS update and the current
    state:

      X_{k+1} = (1-d) * NS_step(X_k) + d * X_k   for k >= 2

    This SLOWS DOWN convergence for the late steps, so the final output
    has partially equalized singular values — NOT fully equalized like NS₅,
    but NOT as spread as NS₂.

    The key insight: v5 blends NS₅ (no curvature) with NS₂ (lots of curvature)
    externally. v9 achieves a similar effect by modifying the iteration itself,
    which should give a smoother interpolation between curvature preservation
    and orthogonalization.

    Why this might be better than v5:
    - v5's blend is LINEAR between two snapshots. The intermediate SV
      distribution jumps discontinuously at the save point.
    - v9's dampening creates a SMOOTH trajectory through SV space.
      Each dampened step finds its own equilibrium.
    - The dampened iteration naturally adapts to each matrix's SV structure,
      since the interpolation affects large and small SVs differently.

    Cost: ZERO extra memory. Same matmul count. Just a different iteration.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        dampen_after: int = 2,
        dampen_factor: float = 0.3,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            dampen_after: NS step at which dampening begins (1-indexed).
                          Default 2: first 2 steps are standard NS (fast cleanup),
                          last 3 steps are dampened (preserve curvature).
            dampen_factor: How much to dampen (0=standard NS, 1=freeze after dampen_after).
                          0.3 means each late step is 70% NS update + 30% current state.
        """
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps,
            dampen_after=dampen_after, dampen_factor=dampen_factor,
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
            dampen_after = group["dampen_after"]
            dampen_factor = group["dampen_factor"]
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

                # Phase 1: Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Dampened NS
                orth = newton_schulz_dampened(
                    u, steps=ns_steps,
                    dampen_after=dampen_after,
                    dampen_factor=dampen_factor,
                    eps=eps,
                )

                # Phase 3: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
