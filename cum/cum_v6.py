import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_multi_resolution
from .utils import aspect_ratio_scale


class CUMv6(Optimizer):
    """
    CUM v6: Adaptive Multi-Resolution NS.

    Mathematical innovation: The blend between NS_full and NS_partial is
    determined AUTOMATICALLY per layer per step, based on the spectral
    divergence between the two NS stages.

    Spectral divergence = 1 - cos_sim(NS_full, NS_partial)

    When divergence is HIGH:
      - NS changed the matrix a lot between steps 2→5
      - The gradient was ill-conditioned (wide SV spread)
      - NS destroyed a lot of curvature info
      - → Use MORE blend to recover curvature

    When divergence is LOW:
      - NS barely changed between steps 2→5
      - The gradient was already well-conditioned
      - NS didn't destroy much info
      - → Use LESS blend (curvature is redundant)

    This gives each layer the right amount of curvature preservation
    based on its actual SV structure at each step. Attention Q/K/V
    projections (often ill-conditioned) get more curvature. FFN weights
    (often well-conditioned) get less.

    Additional innovation: EMA smoothing of the blend parameter to prevent
    oscillation between steps. The effective blend is:
      blend_t = beta_blend * blend_{t-1} + (1-beta_blend) * divergence * blend_scale

    Cost: One extra cosine similarity per weight matrix (~O(mn), negligible).
    Memory: One scalar (blend EMA) per weight matrix.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        ns_save_at: int = 2,
        eps: float = 1e-7,
        nesterov: bool = True,
        blend_scale: float = 0.5,
        blend_max: float = 0.3,
        beta_blend: float = 0.9,
    ):
        """
        Args:
            blend_scale: Multiplier for divergence → blend conversion.
                         Higher = more aggressive curvature preservation.
            blend_max: Maximum blend value (caps adaptive blend).
            beta_blend: EMA decay for blend smoothing.
        """
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, ns_save_at=ns_save_at,
            eps=eps, nesterov=nesterov,
            blend_scale=blend_scale, blend_max=blend_max, beta_blend=beta_blend,
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
            eps = group["eps"]
            nesterov = group["nesterov"]
            blend_scale = group["blend_scale"]
            blend_max = group["blend_max"]
            beta_blend = group["beta_blend"]

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
                    state["blend_ema"] = 0.0

                state["step"] += 1

                # Phase 1: Momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Multi-Resolution NS
                orth_full, orth_partial = newton_schulz_multi_resolution(
                    u, steps=ns_steps, save_at=ns_save_at, eps=eps,
                )

                # Phase 3: Compute spectral divergence
                # How much did NS change between the save point and convergence?
                full_norm = orth_full.norm()
                partial_norm = orth_partial.norm()

                if full_norm > eps and partial_norm > eps:
                    cos_sim = (orth_full * orth_partial).sum() / (full_norm * partial_norm)
                    divergence = (1.0 - cos_sim).clamp(min=0.0).item()
                else:
                    divergence = 0.0

                # Phase 4: Adaptive blend via EMA-smoothed divergence
                raw_blend = min(divergence * blend_scale, blend_max)
                state["blend_ema"] = beta_blend * state["blend_ema"] + (1 - beta_blend) * raw_blend
                effective_blend = state["blend_ema"]

                # Phase 5: Apply blend
                if effective_blend > 1e-6:
                    orth_partial_scaled = orth_partial * (full_norm / (partial_norm + eps))
                    orth = (1.0 - effective_blend) * orth_full + effective_blend * orth_partial_scaled
                else:
                    orth = orth_full

                # Phase 6: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
