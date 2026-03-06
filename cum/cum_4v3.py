import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM4v3(Optimizer):
    """
    CUM 4v3: Weight-Geometry-Aware Spectral Update (WGASU).

    Core idea: The gradient G tells us the Euclidean direction. But
    what matters for the function isn't Euclidean distance in weight space —
    it's how the OUTPUT changes. A weight matrix W maps x → Wx, so the
    effect of a perturbation ΔW on the output is ΔW @ x, which depends
    on W's singular structure.

    Algorithm:
    1. Track W's top-k singular subspace cheaply (subspace iteration on W itself)
    2. Decompose the gradient into components along W's singular directions
    3. Reweight: components along W's SMALL singular directions get AMPLIFIED
       (these directions have outsized effect on learning but are underrepresented)
    4. Components along W's LARGE singular directions get DAMPENED
       (these are already well-learned, updating them further risks overshoot)

    This is NOT Muon (which ignores W entirely).
    This is NOT Shampoo (which uses gradient covariance, not weight structure).
    This is NOT natural gradient (which uses Fisher information).

    It's a novel approach: use the WEIGHT's spectrum to reweight the GRADIENT's
    spectral components. Small SV directions of W = high leverage → amplify.
    Large SV directions of W = low leverage → dampen.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        top_k: int = 8,
        reweight_strength: float = 0.5,
        subspace_iters: int = 3,
        subspace_freq: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, top_k=top_k, reweight_strength=reweight_strength,
            subspace_iters=subspace_iters, subspace_freq=subspace_freq,
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
            top_k = group["top_k"]
            strength = group["reweight_strength"]
            sub_iters = group["subspace_iters"]
            sub_freq = group["subspace_freq"]
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
                k = min(top_k, min(m_dim, n_dim))

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)
                    # Subspace of W's right singular vectors
                    state["W_V"] = torch.randn(n_dim, k)
                    state["W_V"], _ = torch.linalg.qr(state["W_V"])
                    state["W_S"] = torch.ones(k)  # singular values of W

                state["step"] += 1

                # Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Track W's top-k singular subspace (update every sub_freq steps)
                W = p.data
                if W.ndim > 2:
                    W = W.view(W.shape[0], -1)

                if state["step"] % sub_freq == 1 or sub_freq == 1:
                    V = state["W_V"]
                    for _ in range(sub_iters):
                        AV = W @ V          # (m, k)
                        V = W.t() @ AV      # (n, k)
                        V, _ = torch.linalg.qr(V)
                    state["W_V"] = V

                    # Get singular values
                    AV = W @ V  # (m, k)
                    U_small, S_small, Vh_small = torch.linalg.svd(
                        AV, full_matrices=False
                    )
                    state["W_S"] = S_small
                    state["W_U"] = U_small  # (m, k)
                    state["W_V_rot"] = V @ Vh_small.t()  # (n, k), rotated

                if "W_U" not in state:
                    # First step before subspace is computed — fall back to raw update
                    update = u
                else:
                    W_U = state["W_U"]    # (m, k) - left SVs of W
                    W_V = state["W_V_rot"]  # (n, k) - right SVs of W
                    W_S = state["W_S"]    # (k,) - singular values of W

                    # Project gradient onto W's singular directions
                    # coeff[i] = U_i.T @ u @ V_i (projection of u onto i-th SV direction of W)
                    proj = W_U.t() @ u @ W_V  # (k, k) — projections

                    # We care about the diagonal: how much gradient lies along
                    # each of W's singular directions
                    diag_proj = torch.diag(proj)  # (k,)

                    # Reweight: inverse-proportional to W's singular values
                    # Large SV of W → this direction is already strong → dampen gradient
                    # Small SV of W → this direction is weak → amplify gradient
                    # Weight: (1/σ_W)^strength, normalized
                    inv_weights = (1.0 / (W_S + eps)).pow(strength)
                    inv_weights = inv_weights / (inv_weights.mean() + eps)

                    # Reconstruct reweighted component
                    # Original component: sum_i diag_proj[i] * U_i @ V_i.T
                    # Reweighted: sum_i diag_proj[i] * inv_weights[i] * U_i @ V_i.T
                    # Delta = sum_i diag_proj[i] * (inv_weights[i] - 1) * U_i @ V_i.T
                    scale_delta = diag_proj * (inv_weights - 1.0)  # (k,)

                    # Apply: update = u + sum_i scale_delta[i] * U_i @ V_i.T
                    update = u + (W_U * scale_delta.unsqueeze(0)) @ W_V.t()

                # Normalize to match Muon-like update scale
                target_scale = 0.877 * math.sqrt(min(m_dim, n_dim))
                update = update * (target_scale / (update.norm() + eps))

                # Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(update.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr * scale)

        return loss
