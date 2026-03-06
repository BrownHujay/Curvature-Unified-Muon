import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUM5v1(Optimizer):
    """
    CUM 5v1: Lie Algebra Momentum + NS Spectral Processing.

    The key insight from differential geometry: for matrix optimization,
    the Lie algebra so(m) (skew-symmetric matrices) provides a FLAT space
    where momentum can be accumulated without vector transport.

    Standard Muon: element-wise EMA of gradients → NS
    This: project gradient into Lie algebra → EMA in flat Lie algebra →
          compute tangent direction → NS

    Why this might work where 3v2 (Cayley) failed:
    - 3v2 computed A@W from a SINGLE step's momentum. No persistent state.
    - 5v1 accumulates Ω̄ across steps. The EMA in Lie algebra gives a
      stable, consistent rotational direction that incorporates weight structure.
    - The Lie algebra EMA encodes "which rotation of the weights has been
      consistently useful" — information that element-wise momentum doesn't capture.
    - NS handles the spectral processing (we're not trying to replace NS).

    Cost: ~20% more than Muon (3 extra matmuls for Lie algebra projection per step).
    Memory: One m×m skew-symmetric buffer per weight (Ω̄).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
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
                    # Lie algebra momentum buffer (skew-symmetric, m×m or n×n)
                    # Use the smaller dimension for efficiency
                    if m_dim <= n_dim:
                        state["omega_bar"] = torch.zeros(m_dim, m_dim)
                    else:
                        state["omega_bar"] = torch.zeros(n_dim, n_dim)

                state["step"] += 1

                W = p.data
                if W.ndim > 2:
                    W = W.view(W.shape[0], -1)

                # Project gradient into Lie algebra: Ω = 0.5*(g@W.T - W@g.T)
                # This is the skew-symmetric part (rotational component)
                if m_dim <= n_dim:
                    # Work in m×m space (smaller)
                    A = g @ W.t()  # m×m
                    omega = 0.5 * (A - A.t())  # skew-symmetric

                    # EMA in Lie algebra (flat space — no vector transport needed!)
                    omega_bar = state["omega_bar"]
                    omega_bar.mul_(beta1).add_(omega, alpha=1 - beta1)

                    # Nesterov-like: use current + momentum
                    if nesterov:
                        omega_nesterov = omega + beta1 * omega_bar
                    else:
                        omega_nesterov = omega_bar.clone()

                    # Tangent direction: Ω̄ @ W (m×m @ m×n → m×n)
                    u = omega_nesterov @ W
                else:
                    # Work in n×n space (smaller when m > n)
                    A = g.t() @ W  # n×n... wait, g.T is n×m, W is m×n, so g.T@W is n×n
                    # Actually: Ω = 0.5*(g@W.T - W@g.T) is m×m
                    # For m > n, better to work with the dual: Ω' = 0.5*(g.T@W - W.T@g) which is n×n
                    # Then direction = W @ Ω' (m×n @ n×n → m×n)
                    # Note: W@Ω' ≠ Ω@W in general, but both are valid tangent directions
                    A = W.t() @ g  # n×m @ ... wait
                    # g is m×n, W is m×n
                    # g.T @ W = n×m @ m×n = n×n ✓
                    # W.T @ g = n×m @ m×n = n×n ✓
                    A = g.t() @ W  # n×n
                    omega = 0.5 * (A - A.t())  # skew-symmetric n×n

                    omega_bar = state["omega_bar"]
                    omega_bar.mul_(beta1).add_(omega, alpha=1 - beta1)

                    if nesterov:
                        omega_nesterov = omega + beta1 * omega_bar
                    else:
                        omega_nesterov = omega_bar.clone()

                    # Direction: W @ Ω' (m×n @ n×n → m×n)
                    u = W @ omega_nesterov

                # Apply NS for spectral processing
                orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
