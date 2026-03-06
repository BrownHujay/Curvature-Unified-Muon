import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM7v1(Optimizer):
    """
    CUM 7v1: SVD + Custom SV Mapping Functions.

    Deep Research V2 identified that the optimal SV mapping should:
    1. Equalize SVs above a noise threshold (exploit all informative directions)
    2. Apply sub-unity contraction (~0.88, not 1.0) for spectral norm regularization
    3. Be MONOTONE (unlike the oscillating NS blend)

    Modes:
    - "huber": f(σ) = min(σ^α, c)
        α controls equalization (α→0 = full equalize, α=1 = SGD)
        c controls sub-unity cap (0.88 = NS-like contraction)
        Monotone, 2-param, theoretically grounded in robust statistics + Schatten-p

    - "power": f(σ) = c * σ^α / max(σ^α)
        Pure Schatten-p descent scaled to cap c. α = (p-1) for Schatten-p.
        α=0 → sign (Schatten-∞), α=1 → identity (Schatten-2/SGD)

    - "huber_smooth": f(σ) = c * tanh(σ^α / c)
        Smooth approximation to Huber. Differentiable everywhere.

    - "firm": f(σ) = SCAD-like firm thresholding
        Three regimes: linear (small σ), transition, flat (large σ)
        Provides noise suppression + equalization + sub-unity cap

    - "scheduled_power": f(σ) = σ^{α(t)} where α(t) cosine-anneals
        from α_start (less equalization early) to α_end (more equalization late)
        Motivated by gradient spectra changing during training.

    Cost: Same as 5v6 — one SVD per step (~5x slower than NS on GPU).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "huber",
        alpha: float = 0.3,
        cap: float = 0.88,
        # For scheduled modes
        alpha_start: float = 0.5,
        alpha_end: float = 0.1,
        total_steps: int = 2000,
        # For firm thresholding
        threshold: float = 0.1,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode, alpha=alpha, cap=cap,
            alpha_start=alpha_start, alpha_end=alpha_end, total_steps=total_steps,
            threshold=threshold, nesterov=nesterov, eps=eps,
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
            mode = group["mode"]
            alpha = group["alpha"]
            cap = group["cap"]
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

                # Standard Muon momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # SVD
                try:
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                except torch._C._LinAlgError:
                    u = u + eps * torch.randn_like(u)
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                # Normalize SVs to [0, 1]
                S_norm = S / (S[0] + eps)

                # Apply SV mapping based on mode
                if mode == "huber":
                    # f(σ) = min(σ^α, c)
                    S_mapped = torch.clamp(S_norm.pow(alpha), max=cap)

                elif mode == "power":
                    # f(σ) = c * σ^α / max(σ^α)
                    S_pow = S_norm.pow(alpha)
                    S_mapped = cap * S_pow / (S_pow[0] + eps)

                elif mode == "huber_smooth":
                    # f(σ) = c * tanh(σ^α / c)
                    S_mapped = cap * torch.tanh(S_norm.pow(alpha) / cap)

                elif mode == "firm":
                    # SCAD-like firm thresholding
                    # Below threshold: linear (preserve but shrink)
                    # Above threshold but below 1: transition to cap
                    # At 1: equals cap
                    thresh = group["threshold"]
                    S_mapped = torch.where(
                        S_norm <= thresh,
                        cap * S_norm / (thresh + eps),  # linear ramp
                        torch.where(
                            S_norm <= 1.0,
                            cap - (cap - cap * S_norm) * (1.0 - S_norm) / (1.0 - thresh + eps),
                            torch.full_like(S_norm, cap),
                        ),
                    )
                    # Simplified: just use smooth transition
                    # Actually let's do the cleaner version:
                    # Linear below threshold, smooth saturation to cap above
                    t = torch.clamp((S_norm - thresh) / (1.0 - thresh + eps), 0, 1)
                    S_mapped = cap * thresh * (1 - t) / (S_norm + eps) * S_norm + cap * t

                elif mode == "scheduled_power":
                    # Cosine anneal alpha from alpha_start to alpha_end
                    t = state["step"]
                    total = group["total_steps"]
                    progress = min(t / total, 1.0)
                    current_alpha = group["alpha_end"] + 0.5 * (
                        group["alpha_start"] - group["alpha_end"]
                    ) * (1 + math.cos(math.pi * progress))
                    S_pow = S_norm.pow(current_alpha)
                    S_mapped = cap * S_pow / (S_pow[0] + eps)

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Reconstruct
                orth = U * S_mapped.unsqueeze(0) @ Vh

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
