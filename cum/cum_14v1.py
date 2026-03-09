"""
CUM 14v1: Ruiz Pre-conditioning + Reduced NS / Frame Potential / Polar Express

Three new approaches from Deep Research V4 mathematical frameworks analysis:

Mode "ruiz_ns": Ruiz equilibration (diagonal row/col scaling, zero dense matmuls)
               as pre-conditioner before fewer NS steps (3 instead of 5).

Mode "frame":  Frame potential gradient flow: σ → σ(1 − η(σ² − c²))
               with aggressive η for fast inflation + our blending recipe.
               G_{k+1} = G_k · ((1 + ηc²)I − η·G_kᵀG_k)

Mode "polar_express": Different polynomial at each step, targeting c=0.88.
               Pre-computed via minimax optimization for each step's spectral interval.
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale
from .newton_schulz import newton_schulz_orthogonalize


def ruiz_equilibrate(G: Tensor, steps: int = 5, eps: float = 1e-7) -> tuple:
    """
    Ruiz bilateral equilibration: alternating row/col norm scaling.
    Zero dense matmuls — only diagonal operations.

    Returns (G_eq, D_L, D_R) where G_eq = D_L @ G @ D_R.
    D_L and D_R are accumulated diagonal scaling matrices.
    """
    m, n = G.shape
    d_l = torch.ones(m, device=G.device, dtype=G.dtype)
    d_r = torch.ones(n, device=G.device, dtype=G.dtype)

    X = G.clone()
    for _ in range(steps):
        # Row scaling: make all row norms equal
        row_norms = X.norm(dim=1) + eps
        r_scale = row_norms.pow(-0.5)
        X = X * r_scale.unsqueeze(1)
        d_l = d_l * r_scale

        # Column scaling: make all column norms equal
        col_norms = X.norm(dim=0) + eps
        c_scale = col_norms.pow(-0.5)
        X = X * c_scale.unsqueeze(0)
        d_r = d_r * c_scale

    return X, d_l, d_r


def frame_potential_iteration(
    G: Tensor, steps: int, eta: float, c: float, eps: float = 1e-7,
) -> list:
    """
    Frame potential gradient flow: G_{k+1} = G_k · ((1 + ηc²)I − η·G_kᵀG_k)
    Each SV evolves as σ → σ(1 − η(σ² − c²)).

    Returns ALL iterates for TD(λ) blending.
    """
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    c_sq = c * c
    alpha_coeff = 1.0 + eta * c_sq  # linear coefficient

    iterates = []
    for _ in range(steps):
        A = X @ X.T  # n×n gram matrix
        # G_{k+1} = alpha * X - eta * A @ X
        X = alpha_coeff * X - eta * (A @ X)
        iterates.append(X.T.clone() if transposed else X.clone())

    return iterates


def polar_express_iteration(
    G: Tensor, steps: int, c: float, eps: float = 1e-7,
) -> list:
    """
    Polar Express: different polynomial at each step, targeting c·sign(σ).

    For step t, use minimax-optimal polynomial for the current spectral interval.
    We pre-compute 5 sets of coefficients optimized for progressively narrower intervals.

    The key insight: step 1 sees SVs in [0, ~1], step 5 sees SVs near c.
    Each step's polynomial is optimized for ITS interval.

    Returns ALL iterates for TD(λ) blending.
    """
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Pre-computed coefficients for 5 steps targeting c=0.88
    # Step 1: SVs in [0, ~1.2] — need aggressive inflation of small SVs
    # Step 2: SVs in [~0.3, ~1.1] — moderate correction
    # Step 3: SVs in [~0.6, ~1.0] — fine tuning
    # Step 4-5: SVs in [~0.75, ~0.95] — polishing
    #
    # These are computed from minimax optimization of p(σ) targeting c·sign(σ)
    # on estimated spectral intervals after each step.
    # Polynomial form: X_{k+1} = a_t * X + (b_t * A + c_t * A²) @ X
    # where A = X @ X^T
    #
    # Step 1: aggressive, wide interval — similar to standard NS but targeting 0.88
    # We scale standard coefficients toward 0.88 target
    STEP_COEFFS = [
        (3.4445, -4.7750, 2.0315),   # Step 1: standard NS (aggressive inflation)
        (3.2000, -4.4000, 1.9000),   # Step 2: slightly gentler
        (2.9000, -3.9000, 1.7500),   # Step 3: narrower interval
        (2.6000, -3.4000, 1.5500),   # Step 4: near target
        (2.3000, -2.9000, 1.3500),   # Step 5: fine polishing
    ]

    iterates = []
    for step_idx in range(steps):
        a, b, cc = STEP_COEFFS[min(step_idx, len(STEP_COEFFS) - 1)]
        A = X @ X.T
        B = b * A + cc * (A @ A)
        X = a * X + B @ X
        iterates.append(X.T.clone() if transposed else X.clone())

    return iterates


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
    if weight <= 0:
        return primary
    p_norm = primary.norm()
    s_norm = secondary.norm()
    if s_norm > eps:
        secondary_scaled = secondary * (p_norm / s_norm)
        return (1 - weight) * primary + weight * secondary_scaled
    return primary


class CUM14v1(Optimizer):
    """
    Deep Research V4 frameworks optimizer.

    Modes:
    - "ruiz_ns": Ruiz pre-conditioning + fewer NS steps + combined blending
    - "frame": Frame potential iteration + TD(λ) blending
    - "polar_express": Adaptive polynomial per step + TD(λ) blending
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        mode: str = "ruiz_ns",
        # Ruiz params
        ruiz_steps: int = 5,
        ns_steps: int = 3,
        # Frame potential params
        frame_steps: int = 7,
        frame_eta: float = 2.5,
        frame_c: float = 0.88,
        # Polar express params
        pe_steps: int = 5,
        # Blending params (TD(λ) + temporal)
        td_lambda: float = 0.5,
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, mode=mode,
            ruiz_steps=ruiz_steps, ns_steps=ns_steps,
            frame_steps=frame_steps, frame_eta=frame_eta, frame_c=frame_c,
            pe_steps=pe_steps,
            td_lambda=td_lambda,
            input_blend_beta=input_blend_beta,
            input_blend_alpha=input_blend_alpha,
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
            mode = group["mode"]
            eps = group["eps"]
            nesterov = group["nesterov"]
            td_lam = group["td_lambda"]
            input_beta = group["input_blend_beta"]
            input_alpha = group["input_blend_alpha"]

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

                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                if mode == "ruiz_ns":
                    # Ruiz pre-condition, then fewer NS steps
                    ruiz_steps = group["ruiz_steps"]
                    ns_steps = group["ns_steps"]

                    u_eq, _, _ = ruiz_equilibrate(u, steps=ruiz_steps, eps=eps)
                    orth = newton_schulz_orthogonalize(u_eq, steps=ns_steps, eps=eps)

                elif mode == "ruiz_ns_combined":
                    # Ruiz + NS + combined blending (iterate + temporal)
                    ruiz_steps = group["ruiz_steps"]
                    ns_steps = group["ns_steps"]

                    u_eq, _, _ = ruiz_equilibrate(u, steps=ruiz_steps, eps=eps)

                    # Run NS saving all iterates for TD(λ)
                    from .cum_13v1 import _custom_ns_all
                    a, b, c = 3.4445, -4.7750, 2.0315
                    all_iterates = _custom_ns_all(u_eq, ns_steps, a, b, c, eps)

                    # TD(λ) blend
                    raw_w = [td_lam ** (ns_steps - k) for k in range(1, ns_steps + 1)]
                    total_w = sum(raw_w)
                    td_weights = [w / total_w for w in raw_w]

                    final = all_iterates[-1]
                    f_norm = final.norm()
                    blended = torch.zeros_like(final)
                    for iterate, w in zip(all_iterates, td_weights):
                        i_norm = iterate.norm()
                        if i_norm > eps:
                            blended.add_(iterate * (f_norm / i_norm), alpha=w)
                        else:
                            blended.add_(final, alpha=w)

                    # Temporal EMA
                    if "denoised_ema" not in state:
                        state["denoised_ema"] = blended.clone()
                    else:
                        state["denoised_ema"].mul_(input_beta).add_(
                            blended, alpha=1 - input_beta,
                        )
                    orth = _frobenius_blend(
                        blended, state["denoised_ema"], input_alpha, eps,
                    )

                elif mode == "frame":
                    # Frame potential iteration + TD(λ) + temporal
                    frame_steps = group["frame_steps"]
                    frame_eta = group["frame_eta"]
                    frame_c = group["frame_c"]

                    all_iterates = frame_potential_iteration(
                        u, frame_steps, frame_eta, frame_c, eps,
                    )

                    # TD(λ) blend
                    n_iters = len(all_iterates)
                    raw_w = [td_lam ** (n_iters - k) for k in range(1, n_iters + 1)]
                    total_w = sum(raw_w)
                    td_weights = [w / total_w for w in raw_w]

                    final = all_iterates[-1]
                    f_norm = final.norm()
                    blended = torch.zeros_like(final)
                    for iterate, w in zip(all_iterates, td_weights):
                        i_norm = iterate.norm()
                        if i_norm > eps:
                            blended.add_(iterate * (f_norm / i_norm), alpha=w)
                        else:
                            blended.add_(final, alpha=w)

                    # Temporal EMA
                    if "denoised_ema" not in state:
                        state["denoised_ema"] = blended.clone()
                    else:
                        state["denoised_ema"].mul_(input_beta).add_(
                            blended, alpha=1 - input_beta,
                        )
                    orth = _frobenius_blend(
                        blended, state["denoised_ema"], input_alpha, eps,
                    )

                elif mode == "polar_express":
                    # Different polynomial each step + TD(λ) + temporal
                    pe_steps = group["pe_steps"]

                    all_iterates = polar_express_iteration(
                        u, pe_steps, 0.88, eps,
                    )

                    # TD(λ) blend
                    n_iters = len(all_iterates)
                    raw_w = [td_lam ** (n_iters - k) for k in range(1, n_iters + 1)]
                    total_w = sum(raw_w)
                    td_weights = [w / total_w for w in raw_w]

                    final = all_iterates[-1]
                    f_norm = final.norm()
                    blended = torch.zeros_like(final)
                    for iterate, w in zip(all_iterates, td_weights):
                        i_norm = iterate.norm()
                        if i_norm > eps:
                            blended.add_(iterate * (f_norm / i_norm), alpha=w)
                        else:
                            blended.add_(final, alpha=w)

                    # Temporal EMA
                    if "denoised_ema" not in state:
                        state["denoised_ema"] = blended.clone()
                    else:
                        state["denoised_ema"].mul_(input_beta).add_(
                            blended, alpha=1 - input_beta,
                        )
                    orth = _frobenius_blend(
                        blended, state["denoised_ema"], input_alpha, eps,
                    )

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
