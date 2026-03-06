import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUMv4(Optimizer):
    """
    CUM v4: Three stacked innovations on top of Muon.

    1. Gradient Centralization: center gradient rows before momentum.
       Removes common-mode bias, acts as implicit regularization.
       Each row g_i → g_i - mean(g_i). This zeros the row means,
       making the gradient "zero-centered" and reducing the projection
       onto the all-ones direction (which is typically not useful for
       learned feature extractors).

    2. Soft NS: After orthogonalization, blend with normalized pre-NS
       momentum to preserve some of the gradient's singular value
       structure. NS equalizes ALL singular values — but the top SVs
       carry real signal about which directions matter. Preserving
       10% of this structure (ns_blend=0.1) gives better convergence.

    3. Coherence-Adaptive Step Size: Per-layer learning rate modulation
       based on temporal gradient coherence. When the current gradient
       aligns with the momentum direction (high cosine similarity),
       the loss landscape is locally smooth → safe to take bigger steps.
       When they disagree, the landscape is rough → be conservative.
       Cost: one dot product per weight matrix.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
        ns_blend: float = 0.1,
        coherence_alpha: float = 0.3,
        centralize: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, eps=eps,
            nesterov=nesterov, ns_blend=ns_blend,
            coherence_alpha=coherence_alpha, centralize=centralize,
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
            ns_blend = group["ns_blend"]
            coherence_alpha = group["coherence_alpha"]
            centralize = group["centralize"]

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

                # === Innovation 1: Gradient Centralization ===
                # Center each row: removes common-mode, reduces rank-1 bias
                if centralize:
                    g = g - g.mean(dim=1, keepdim=True)

                # === Momentum (same as Muon) ===
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # === Innovation 3: Coherence-Adaptive Step Size ===
                # Cosine similarity between current gradient and momentum
                step_scale = 1.0
                if coherence_alpha > 0:
                    g_norm = g.norm()
                    mb_norm = mb.norm()
                    if g_norm > eps and mb_norm > eps:
                        cos_sim = (g * mb).sum() / (g_norm * mb_norm)
                        coherence = cos_sim.clamp(min=0.0, max=1.0)
                        step_scale = 1.0 + coherence_alpha * coherence.item()

                # === NS Orthogonalization (same as Muon) ===
                orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                # === Innovation 2: Soft NS ===
                # Blend with normalized momentum to preserve curvature structure
                if ns_blend > 0:
                    orth_norm = orth.norm()
                    u_norm = u.norm()
                    if u_norm > eps:
                        u_normalized = u * (orth_norm / u_norm)
                        orth = (1.0 - ns_blend) * orth + ns_blend * u_normalized

                # === Weight Update ===
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * step_scale * scale)
                else:
                    p.add_(orth, alpha=-lr * step_scale * scale)

        return loss
