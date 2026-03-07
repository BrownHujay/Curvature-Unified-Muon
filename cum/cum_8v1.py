import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import (
    newton_schulz_multi_resolution,
    newton_schulz_n_resolution,
    newton_schulz_orthogonalize,
)
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


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
    """Blend two matrices with Frobenius norm matching."""
    if weight <= 0:
        return primary
    p_norm = primary.norm()
    s_norm = secondary.norm()
    if s_norm > eps:
        secondary_scaled = secondary * (p_norm / s_norm)
        return (1 - weight) * primary + weight * secondary_scaled
    return primary


class CUM8v1(Optimizer):
    """
    CUM 8v1: Generalized NS Iterate Blending Explorer.

    Systematically explores the space of NS iterate blending strategies:

    Modes (matrix path):
    - "two_point": (1-b)*NS_n + b*scale(NS_k) — generalized v5
    - "three_point": (1-a-b)*NS_n + b*scale(NS_j) + a*scale(NS_i) — three iterates
    - "input_blend": EMA of past NS outputs blended into current NS output
    - "combined": iterate blend + input blend together (within-step + across-step)
    - "adaptive_residual": base + adaptive corrections from curvature & temporal signals
    - "cosine_gated": corrections only applied when curvature & temporal signals agree
    - "scheduled_three": three-point early, two-point late (anneal NS₁ weight to 0)

    Modes (SVD path):
    - "two_point": Same formula but via SVD scalar polynomial
    - "three_point": Same formula but via SVD scalar polynomial
    - "sv_blend": Geometric or arithmetic mean of SV vectors from two iterates

    Cost: Matrix modes = same as Muon. SVD modes = ~5x slower (full SVD per step).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        method: str = "matrix",       # "matrix" or "svd"
        mode: str = "two_point",      # blending strategy
        # NS iteration control
        ns_steps: int = 5,            # total NS steps (can go to 7, 8)
        # Two-point blend
        save_at: int = 2,             # which iterate to blend
        blend: float = 0.15,          # weight of intermediate
        # Three-point blend
        save_at_a: int = 1,           # first intermediate
        save_at_b: int = 3,           # second intermediate
        blend_a: float = 0.05,        # weight of first
        blend_b: float = 0.10,        # weight of second
        # SV-space blend (SVD only)
        sv_blend_mode: str = "arithmetic",  # "arithmetic" or "geometric"
        # Input blend (matrix only)
        input_blend_beta: float = 0.5,   # EMA decay for denoised momentum
        input_blend_alpha: float = 0.15, # blend strength into NS output
        # Adaptive residual params
        alpha_curv: float = 0.15,     # base curvature correction weight
        beta_temp: float = 0.15,      # base temporal correction weight
        # Scheduled three-point params
        total_steps: int = 2000,      # for scheduling
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, method=method, mode=mode,
            ns_steps=ns_steps, save_at=save_at, blend=blend,
            save_at_a=save_at_a, save_at_b=save_at_b,
            blend_a=blend_a, blend_b=blend_b,
            sv_blend_mode=sv_blend_mode,
            input_blend_beta=input_blend_beta, input_blend_alpha=input_blend_alpha,
            alpha_curv=alpha_curv, beta_temp=beta_temp,
            total_steps=total_steps,
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
            method = group["method"]
            mode = group["mode"]
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

                # Standard Muon momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                if method == "matrix":
                    orth = self._matrix_step(u, group, state, eps)
                elif method == "svd":
                    orth = self._svd_step(u, group, eps)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss

    def _matrix_step(self, u: Tensor, group: dict, state: dict, eps: float) -> Tensor:
        mode = group["mode"]
        ns_steps = group["ns_steps"]

        if mode == "two_point":
            save_at = group["save_at"]
            blend = group["blend"]
            full, partial = newton_schulz_multi_resolution(
                u, steps=ns_steps, save_at=save_at, eps=eps,
            )
            return _frobenius_blend(full, partial, blend, eps)

        elif mode == "three_point":
            save_at_a = group["save_at_a"]
            save_at_b = group["save_at_b"]
            blend_a = group["blend_a"]
            blend_b = group["blend_b"]
            final, inter_a, inter_b = newton_schulz_n_resolution(
                u, steps=ns_steps, save_at=(save_at_a, save_at_b), eps=eps,
            )
            # Three-point blend: scale each intermediate to final's norm
            f_norm = final.norm()
            a_norm = inter_a.norm()
            b_norm = inter_b.norm()
            result = (1 - blend_a - blend_b) * final
            if a_norm > eps:
                result = result + blend_a * inter_a * (f_norm / a_norm)
            if b_norm > eps:
                result = result + blend_b * inter_b * (f_norm / b_norm)
            return result

        elif mode == "input_blend":
            input_beta = group["input_blend_beta"]
            input_alpha = group["input_blend_alpha"]
            full = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)
            # Update denoised EMA of NS outputs
            if "denoised_ema" not in state:
                state["denoised_ema"] = full.clone()
            else:
                state["denoised_ema"].mul_(input_beta).add_(full, alpha=1 - input_beta)
            return _frobenius_blend(full, state["denoised_ema"], input_alpha, eps)

        elif mode == "combined":
            # Iterate blend (within-step) + input blend (across-step)
            save_at = group["save_at"]
            blend = group["blend"]
            input_beta = group["input_blend_beta"]
            input_alpha = group["input_blend_alpha"]
            # Get NS₂ and NS₅
            full, partial = newton_schulz_multi_resolution(
                u, steps=ns_steps, save_at=save_at, eps=eps,
            )
            # Within-step: iterate blend
            iterate_blended = _frobenius_blend(full, partial, blend, eps)
            # Update EMA of blended outputs
            if "denoised_ema" not in state:
                state["denoised_ema"] = iterate_blended.clone()
            else:
                state["denoised_ema"].mul_(input_beta).add_(
                    iterate_blended, alpha=1 - input_beta
                )
            # Across-step: temporal blend
            return _frobenius_blend(iterate_blended, state["denoised_ema"], input_alpha, eps)

        elif mode == "adaptive_residual":
            # base + adaptive corrections from curvature & temporal signals
            save_at = group["save_at"]
            alpha_c = group["alpha_curv"]
            beta_t = group["beta_temp"]
            input_beta = group["input_blend_beta"]
            full, partial = newton_schulz_multi_resolution(
                u, steps=ns_steps, save_at=save_at, eps=eps,
            )
            # Update EMA
            if "denoised_ema" not in state:
                state["denoised_ema"] = full.clone()
            else:
                state["denoised_ema"].mul_(input_beta).add_(full, alpha=1 - input_beta)
            # Compute residuals
            base_norm = full.norm() + eps
            # Scale partial to match full norm before computing residual
            p_norm = partial.norm()
            partial_scaled = partial * (base_norm / (p_norm + eps)) if p_norm > eps else partial
            delta_curv = partial_scaled - full  # curvature destroyed by NS steps 3-5
            # Scale EMA to match full norm before computing residual
            e_norm = state["denoised_ema"].norm()
            ema_scaled = state["denoised_ema"] * (base_norm / (e_norm + eps)) if e_norm > eps else state["denoised_ema"]
            delta_temp = ema_scaled - full      # temporal consistency correction
            # Adaptive weights: trust corrections proportional to their relative magnitude
            curv_ratio = delta_curv.norm() / base_norm
            temp_ratio = delta_temp.norm() / base_norm
            w_curv = alpha_c * curv_ratio       # more curvature = more correction
            w_temp = beta_t / (1.0 + temp_ratio)  # more instability = less trust
            return full + w_curv * delta_curv + w_temp * delta_temp

        elif mode == "cosine_gated":
            # Corrections gated by agreement between curvature & temporal signals
            save_at = group["save_at"]
            alpha_c = group["alpha_curv"]
            beta_t = group["beta_temp"]
            input_beta = group["input_blend_beta"]
            full, partial = newton_schulz_multi_resolution(
                u, steps=ns_steps, save_at=save_at, eps=eps,
            )
            # Update EMA
            if "denoised_ema" not in state:
                state["denoised_ema"] = full.clone()
            else:
                state["denoised_ema"].mul_(input_beta).add_(full, alpha=1 - input_beta)
            # Compute residuals (norm-matched)
            base_norm = full.norm() + eps
            p_norm = partial.norm()
            partial_scaled = partial * (base_norm / (p_norm + eps)) if p_norm > eps else partial
            delta_curv = partial_scaled - full
            e_norm = state["denoised_ema"].norm()
            ema_scaled = state["denoised_ema"] * (base_norm / (e_norm + eps)) if e_norm > eps else state["denoised_ema"]
            delta_temp = ema_scaled - full
            # Gate: only inject when curvature and temporal corrections agree
            dc_norm = delta_curv.norm() + eps
            dt_norm = delta_temp.norm() + eps
            cos_sim = (delta_curv * delta_temp).sum() / (dc_norm * dt_norm)
            gate = cos_sim.clamp(min=0.0)  # only positive agreement
            return full + gate * (alpha_c * delta_curv + beta_t * delta_temp)

        elif mode == "scheduled_three":
            # Three-point early (fast convergence), two-point late (clean finish)
            # NS₁ weight anneals from blend_a to 0 over training
            save_at_a = group["save_at_a"]
            save_at_b = group["save_at_b"]
            blend_a_max = group["blend_a"]
            blend_b = group["blend_b"]
            total = group["total_steps"]
            final, inter_a, inter_b = newton_schulz_n_resolution(
                u, steps=ns_steps, save_at=(save_at_a, save_at_b), eps=eps,
            )
            # Cosine anneal blend_a from blend_a_max to 0
            progress = min(state["step"] / max(total, 1), 1.0)
            blend_a = blend_a_max * 0.5 * (1 + math.cos(math.pi * progress))
            # Three-point blend with scheduled weights
            f_norm = final.norm()
            a_norm = inter_a.norm()
            b_norm = inter_b.norm()
            result = (1 - blend_a - blend_b) * final
            if a_norm > eps:
                result = result + blend_a * inter_a * (f_norm / a_norm)
            if b_norm > eps:
                result = result + blend_b * inter_b * (f_norm / b_norm)
            return result

        else:
            raise ValueError(f"Unknown matrix mode: {mode}")

    def _svd_step(self, u: Tensor, group: dict, eps: float) -> Tensor:
        mode = group["mode"]
        ns_steps = group["ns_steps"]

        # SVD with fallback
        try:
            U, S, Vh = torch.linalg.svd(u, full_matrices=False)
        except torch._C._LinAlgError:
            u = u + eps * torch.randn_like(u)
            U, S, Vh = torch.linalg.svd(u, full_matrices=False)

        frob = S.norm() + eps
        S_scaled = S / frob

        if mode == "two_point":
            save_at = group["save_at"]
            blend = group["blend"]
            S_k = _ns_poly_k(S_scaled, save_at)
            S_n = _ns_poly_k(S_scaled, ns_steps)
            S_out = (1 - blend) * S_n + blend * S_k

        elif mode == "three_point":
            save_at_a = group["save_at_a"]
            save_at_b = group["save_at_b"]
            blend_a = group["blend_a"]
            blend_b = group["blend_b"]
            S_a = _ns_poly_k(S_scaled, save_at_a)
            S_b = _ns_poly_k(S_scaled, save_at_b)
            S_n = _ns_poly_k(S_scaled, ns_steps)
            S_out = (1 - blend_a - blend_b) * S_n + blend_b * S_b + blend_a * S_a

        elif mode == "sv_blend":
            save_at = group["save_at"]
            blend = group["blend"]
            sv_mode = group["sv_blend_mode"]
            S_k = _ns_poly_k(S_scaled, save_at)
            S_n = _ns_poly_k(S_scaled, ns_steps)
            if sv_mode == "arithmetic":
                S_out = (1 - blend) * S_n + blend * S_k
            elif sv_mode == "geometric":
                # Geometric mean: S_n^(1-b) * S_k^b
                # Clamp to avoid issues with negative/zero values
                S_n_safe = S_n.clamp(min=eps)
                S_k_safe = S_k.clamp(min=eps)
                S_out = S_n_safe.pow(1 - blend) * S_k_safe.pow(blend)
            else:
                raise ValueError(f"Unknown sv_blend_mode: {sv_mode}")

        else:
            raise ValueError(f"Unknown SVD mode: {mode}")

        # Reconstruct
        orth = U * S_out.unsqueeze(0) @ Vh
        return orth
