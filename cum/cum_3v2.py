import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUM3v2(Optimizer):
    """
    CUM 3v2: Cayley Retraction Direction.

    Fundamentally different from Muon's approach:
    - Muon asks: "what's the closest orthogonal matrix to the gradient?"
    - Cayley asks: "what ROTATION of the current WEIGHTS best follows the gradient?"

    The Cayley direction uses the weight matrix itself to compute a
    Stiefel-manifold-aware update:
      A = momentum @ W.T - W @ momentum.T   (skew-symmetric generator)
      direction = A @ W                       (tangent vector)

    This direction naturally:
    - Scales with the misalignment between gradient and weights (zero when aligned)
    - Respects the rotational geometry of the weight space
    - Uses weight structure information that Muon completely ignores

    Optionally followed by NS cleanup (ns_steps > 0) for normalization.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 3,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, eps=eps, nesterov=nesterov,
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

                # Momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Get the weight matrix
                W = p.data
                if W.ndim > 2:
                    W = W.view(W.shape[0], -1)

                # Cayley direction: skew-symmetric generator + tangent vector
                # Efficient computation avoiding large m×m intermediate when m > n
                if m_dim <= n_dim:
                    # A = u @ W.T - W @ u.T  (m×m, small)
                    A = u @ W.t() - W @ u.t()
                    direction = A @ W  # m×n
                else:
                    # Avoid m×m intermediate: expand A @ W directly
                    # A @ W = u @ (W.T @ W) - W @ (u.T @ W)
                    WtW = W.t() @ W  # n×n
                    utW = u.t() @ W  # n×n
                    direction = u @ WtW - W @ utW  # m×n

                if ns_steps > 0:
                    # NS cleanup for normalization and further orthogonalization
                    orth = newton_schulz_orthogonalize(direction, steps=ns_steps, eps=eps)
                else:
                    # Pure Cayley: Frobenius-normalize, scaled to match NS output magnitude
                    import math
                    target_scale = 0.877 * math.sqrt(min(m_dim, n_dim))
                    orth = direction * (target_scale / (direction.norm() + eps))

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
