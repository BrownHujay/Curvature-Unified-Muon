import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


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


class CUM9v1(Optimizer):
    """
    CUM 9v1: Dual-Momentum Muon.

    Tests whether intermediate iterate blending generalizes beyond NS.
    Two momentum buffers with different time constants:
    - Fast (β=0.80): recent gradients, curvature-rich
    - Slow (β=0.95): stable direction, denoised

    Modes:
    - "pre_ns": Blend fast+slow momentum BEFORE NS. One NS pass (cheap).
    - "post_ns": NS each momentum separately, blend outputs. Two NS passes (2x cost).

    If pre_ns works, the principle is: blending two different "views" of the
    gradient (at different temporal scales) improves training, even before NS.
    If post_ns works but pre_ns doesn't, NS washes out input differences
    but preserves output differences.

    Cost: pre_ns = Muon + 1 extra momentum buffer. post_ns = 2x Muon + 1 extra buffer.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta_fast: float = 0.80,
        beta_slow: float = 0.95,
        blend: float = 0.15,          # weight of fast momentum
        mode: str = "pre_ns",         # "pre_ns" or "post_ns"
        ns_steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta_fast=beta_fast, beta_slow=beta_slow,
            blend=blend, mode=mode, ns_steps=ns_steps,
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
            beta_fast = group["beta_fast"]
            beta_slow = group["beta_slow"]
            blend = group["blend"]
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
                    state["momentum_fast"] = torch.zeros_like(g)
                    state["momentum_slow"] = torch.zeros_like(g)

                state["step"] += 1

                # Update both momentum buffers
                mb_fast = state["momentum_fast"]
                mb_slow = state["momentum_slow"]
                mb_fast.mul_(beta_fast).add_(g, alpha=1 - beta_fast)
                mb_slow.mul_(beta_slow).add_(g, alpha=1 - beta_slow)

                if nesterov:
                    u_fast = g + beta_fast * mb_fast
                    u_slow = g + beta_slow * mb_slow
                else:
                    u_fast = mb_fast.clone()
                    u_slow = mb_slow.clone()

                if mode == "pre_ns":
                    # Blend momenta before NS
                    u_blended = _frobenius_blend(u_slow, u_fast, blend, eps)
                    orth = newton_schulz_orthogonalize(u_blended, steps=ns_steps, eps=eps)

                elif mode == "post_ns":
                    # NS each momentum separately, then blend
                    orth_slow = newton_schulz_orthogonalize(u_slow, steps=ns_steps, eps=eps)
                    orth_fast = newton_schulz_orthogonalize(u_fast, steps=ns_steps, eps=eps)
                    orth = _frobenius_blend(orth_slow, orth_fast, blend, eps)

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
