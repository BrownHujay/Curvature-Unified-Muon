import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


# NS polynomial coefficients (from Muon/Zetta)
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315


def _ns_poly_k(sigma: Tensor, k: int) -> Tensor:
    """Apply NS polynomial k times to scalar SVs."""
    s = sigma.clone()
    for _ in range(k):
        s2 = s * s
        s = _NS_A * s + _NS_B * (s * s2) + _NS_C * (s * s2 * s2)
    return s


class CUM6v8(Optimizer):
    """
    CUM 6v8: Optimal SV Shrinkage — Donoho-Gavish denoising of gradient SVs.

    Instead of NS's polynomial mapping, applies the information-theoretically
    optimal shrinkage from random matrix theory:

    For SV sigma_i above threshold (1 + sqrt(beta)):
        eta*(sigma_i) = sqrt((sigma_i^2 - beta - 1)^2 - 4*beta) / sigma_i
    For sigma_i below threshold:
        eta*(sigma_i) = 0  (pure noise, zero it out)

    where beta = m/n is the aspect ratio.

    The noise level sigma is estimated from the bottom quartile of singular
    values, following the Marchenko-Pastur framework.

    Modes:
    - "hard": Apply hard threshold + optimal shrinkage (zeros noise SVs)
    - "soft": Apply soft shrinkage (shrink all SVs, don't zero any)
    - "blend": Blend optimal-shrinkage output with NS5 output (combines denoising + equalization)

    Cost: Full SVD per step (~40% slower than NS).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "hard",
        noise_est: str = "median",
        ns_blend: float = 0.5,
        nesterov: bool = True,
        eps: float = 1e-7,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode, noise_est=noise_est,
            ns_blend=ns_blend, nesterov=nesterov, eps=eps,
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
            noise_est = group["noise_est"]
            ns_blend_alpha = group["ns_blend"]
            nesterov = group["nesterov"]
            eps = group["eps"]

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

                # SVD of momentum (with robustness for ill-conditioned matrices)
                try:
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
                except torch._C._LinAlgError:
                    # Add small noise to break degeneracies
                    u = u + eps * torch.randn_like(u)
                    U, S, Vh = torch.linalg.svd(u, full_matrices=False)

                n_sv = len(S)
                # beta for Marchenko-Pastur: ratio of smaller to larger dimension
                beta_ratio = min(m_dim, n_dim) / max(m_dim, n_dim)

                # --- Noise estimation ---
                if noise_est == "median":
                    # Median-based estimator (Donoho-Gavish):
                    # Under pure noise, median(SV) / mu_mp gives sigma
                    # mu_mp ≈ sqrt(2 * beta_ratio) is an approximation to the
                    # median of the Marchenko-Pastur distribution
                    mu_mp = math.sqrt(2 * beta_ratio) if beta_ratio > 0.01 else 1.0
                    noise_std = (S.median() / mu_mp).item() / math.sqrt(max(m_dim, n_dim))
                elif noise_est == "quartile":
                    # Estimate noise from the bottom quartile of SVs
                    noise_svs = S[3 * n_sv // 4:]
                    if len(noise_svs) > 0:
                        noise_std = noise_svs.mean().item() / math.sqrt(max(m_dim, n_dim))
                    else:
                        noise_std = S[-1].item() / math.sqrt(max(m_dim, n_dim))
                else:
                    raise ValueError(f"Unknown noise_est: {noise_est}")

                # BBP (Baik-Ben Arous-Peche) transition threshold
                threshold = noise_std * (1 + math.sqrt(beta_ratio))

                # --- Apply shrinkage based on mode ---
                if mode == "hard":
                    # Donoho-Gavish optimal hard shrinkage:
                    # For SVs above threshold, apply:
                    #   eta*(sigma) = sqrt((sigma^2 - beta*noise^2 - noise^2)^2 - 4*beta*noise^4) / sigma
                    # For SVs below threshold: zero them out
                    noise_var = noise_std ** 2
                    mask = S > threshold

                    inner = (S ** 2 - beta_ratio * noise_var - noise_var) ** 2 - 4 * beta_ratio * (noise_var ** 2)
                    inner = torch.clamp(inner, min=0.0)
                    S_shrunk = torch.where(
                        mask,
                        torch.sqrt(inner) / (S + eps),
                        torch.zeros_like(S),
                    )

                    # Normalize so total energy is preserved
                    shrunk_norm = S_shrunk.norm() + eps
                    S_out = S_shrunk / shrunk_norm

                elif mode == "soft":
                    # Soft thresholding: shrink all SVs toward zero but
                    # keep a small floor to avoid zeroing directions entirely
                    S_shrunk = torch.clamp(S - threshold, min=0.01 * S[0].item())
                    # Normalize
                    shrunk_norm = S_shrunk.norm() + eps
                    S_out = S_shrunk / shrunk_norm

                elif mode == "blend":
                    # Blend optimal shrinkage with NS5 polynomial mapping
                    # This combines the denoising benefit of shrinkage with
                    # the equalization benefit of NS

                    # Compute shrunk SVs (hard shrinkage)
                    noise_var = noise_std ** 2
                    mask = S > threshold
                    inner = (S ** 2 - beta_ratio * noise_var - noise_var) ** 2 - 4 * beta_ratio * (noise_var ** 2)
                    inner = torch.clamp(inner, min=0.0)
                    S_shrunk = torch.where(
                        mask,
                        torch.sqrt(inner) / (S + eps),
                        torch.zeros_like(S),
                    )
                    # Normalize shrunk SVs
                    shrunk_norm = S_shrunk.norm() + eps
                    S_shrunk_normed = S_shrunk / shrunk_norm

                    # Compute NS5 polynomial SVs (same normalization as CUM5v6)
                    frob = S.norm() + eps
                    S_scaled = S / frob
                    S_ns5 = _ns_poly_k(S_scaled, 5)

                    # Blend
                    S_out = (1 - ns_blend_alpha) * S_ns5 + ns_blend_alpha * S_shrunk_normed

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Reconstruct
                orth = U * S_out.unsqueeze(0) @ Vh

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
