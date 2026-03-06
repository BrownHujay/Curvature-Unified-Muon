import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

from .newton_schulz import newton_schulz_orthogonalize
from .factored_precond import apply_factored_precond
from .spectral_control import spectral_damping
from .utils import aspect_ratio_scale


class CUM(Optimizer):
    """
    CUM: Curvature-Unified Muon optimizer.

    Muon + factored curvature preconditioning + smooth spectral control.
    Only for 2D hidden layer weights. Use CUMWithAuxAdam for full models.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        ns_steps: int = 3,
        eps: float = 1e-7,
        sigma_max: float = 30.0,
        alpha_damp: float = 0.1,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay,
            ns_steps=ns_steps, eps=eps, sigma_max=sigma_max,
            alpha_damp=alpha_damp, nesterov=nesterov,
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
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            sigma_max = group["sigma_max"]
            alpha_damp = group["alpha_damp"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                assert p.ndim == 2, f"CUM only supports 2D params, got {p.ndim}D"

                g = p.grad
                m_dim, n_dim = p.shape

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["row_var"] = torch.zeros(m_dim, device=p.device, dtype=p.dtype)
                    state["col_var"] = torch.zeros(n_dim, device=p.device, dtype=p.dtype)
                    v = torch.randn(n_dim, device=p.device, dtype=p.dtype)
                    state["power_iter_v"] = v / (v.norm() + 1e-7)

                state["step"] += 1
                step = state["step"]

                # Phase 1: Gradient & Momentum
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * momentum_buffer
                else:
                    u = momentum_buffer.clone()

                # Phase 2: Factored Preconditioning
                u_precond, state["row_var"], state["col_var"] = apply_factored_precond(
                    u, g, state["row_var"], state["col_var"],
                    beta2, step, eps,
                )

                # Phase 3: Newton-Schulz Orthogonalization
                orth = newton_schulz_orthogonalize(u_precond, steps=ns_steps, eps=eps)

                # Phase 4: Spectral Norm Control
                damping_factor, state["power_iter_v"] = spectral_damping(
                    p, state["power_iter_v"], sigma_max, alpha_damp,
                )

                # Phase 5: Weight Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                p.mul_(1 - weight_decay)  # weight decay
                p.add_(orth, alpha=-lr * damping_factor * scale)  # orthogonal update

        return loss
