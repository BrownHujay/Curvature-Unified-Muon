import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUMv2(Optimizer):
    """
    CUM v2: Muon + post-NS adaptive row/column scaling.

    Key insight: NS orthogonalization normalizes the update direction.
    Curvature info should scale the step size AFTER NS, not distort
    the gradient direction BEFORE NS.

    Phase 1: Nesterov momentum (same as Muon)
    Phase 2: NS orthogonalization (same as Muon, 5 steps)
    Phase 3: Adaptive row/column scaling from gradient curvature
    Phase 4: Weight update
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.99,
        ns_steps: int = 5,
        eps: float = 1e-7,
        precond_alpha: float = 0.5,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, ns_steps=ns_steps,
            eps=eps, precond_alpha=precond_alpha, nesterov=nesterov,
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
            eps = group["eps"]
            alpha = group["precond_alpha"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                m_dim, n_dim = g.shape[0], g.view(g.shape[0], -1).shape[1]

                # Handle conv weights: reshape to 2D
                orig_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)
                    p_2d = p.data.view(p.shape[0], -1)
                else:
                    p_2d = p.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p_2d)
                    state["row_var"] = torch.zeros(m_dim, device=p.device, dtype=p.dtype)
                    state["col_var"] = torch.zeros(n_dim, device=p.device, dtype=p.dtype)

                state["step"] += 1
                step_t = state["step"]

                # Phase 1: Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: NS orthogonalization (same as Muon)
                orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                # Phase 3: Adaptive row/column scaling
                if alpha > 0:
                    # Track gradient variance per row and column
                    g_sq = g * g
                    state["row_var"].mul_(beta2).add_(g_sq.mean(dim=1), alpha=1 - beta2)
                    state["col_var"].mul_(beta2).add_(g_sq.mean(dim=0), alpha=1 - beta2)

                    # Bias correction
                    bc = 1.0 - beta2 ** step_t
                    row_v = state["row_var"] / bc
                    col_v = state["col_var"] / bc

                    # Relative scaling: normalize so mean scale ≈ 1.0
                    # High-variance rows/cols get scaled DOWN (smaller steps)
                    # Low-variance rows/cols get scaled UP (larger steps)
                    row_mean = row_v.mean() + eps
                    col_mean = col_v.mean() + eps
                    row_scale = (row_mean / (row_v + eps)).sqrt().pow(alpha)
                    col_scale = (col_mean / (col_v + eps)).sqrt().pow(alpha)

                    orth = orth * row_scale[:, None] * col_scale[None, :]

                # Phase 4: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if g.ndim != orig_shape:
                    # Reshape orth back to original shape for conv weights
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
