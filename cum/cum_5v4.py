import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM5v4(Optimizer):
    """
    CUM 5v4: Adaptive Schatten-p with Per-Layer Condition-Based p.

    Extension of 5v3: instead of fixed p for all layers, adapt p based on
    the gradient momentum's condition number (σ_max/σ_min).

    Intuition: layers with high condition number (spread SVs) need aggressive
    equalization (high p). Layers with low condition number (already equalized)
    can use less equalization (lower p), preserving useful spectral structure.

    p_layer = p_min + (p_max - p_min) * sigmoid(k * (log_cond - log_cond_target))

    Where log_cond = log(σ_max/σ_min) is the log condition number.
    This gives a smooth mapping from condition number to p.

    Cost: Full SVD per step (~40% slower than NS at our scale).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        p_min: float = 4.0,
        p_max: float = 64.0,
        cond_target: float = 10.0,
        cond_steepness: float = 2.0,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, p_min=p_min, p_max=p_max,
            cond_target=cond_target, cond_steepness=cond_steepness,
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
            p_min = group["p_min"]
            p_max = group["p_max"]
            cond_target = group["cond_target"]
            cond_steepness = group["cond_steepness"]
            eps = group["eps"]
            nesterov = group["nesterov"]

            log_cond_target = math.log(cond_target)

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

                # Standard momentum (same as Muon)
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # SVD of momentum
                U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                # Compute condition number and adaptive p
                s_max = S[0] + eps
                s_min = S[-1] + eps
                log_cond = math.log(s_max / s_min)

                # Sigmoid mapping: high condition → high p (more equalization)
                sigmoid_arg = cond_steepness * (log_cond - log_cond_target)
                sigmoid_val = 1.0 / (1.0 + math.exp(-sigmoid_arg))
                schatten_p = p_min + (p_max - p_min) * sigmoid_val

                # SV mapping: σ → σ^{1/(p-1)}
                alpha = 1.0 / (schatten_p - 1.0)
                s_normalized = S / s_max
                S_mapped = s_normalized.pow(alpha)

                # Scale to match NS output
                target_scale = 0.877 * math.sqrt(min(m_dim, n_dim))
                current_scale = S_mapped.norm() + eps
                S_final = S_mapped * (target_scale / current_scale)

                # Reconstruct
                orth = U * S_final.unsqueeze(0) @ Vh

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
