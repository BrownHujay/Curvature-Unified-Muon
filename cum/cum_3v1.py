import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .utils import aspect_ratio_scale


class CUM3v1(Optimizer):
    """
    CUM 3v1: Warm-Started Newton-Schulz.

    Exploits temporal coherence: the polar factor of consecutive momentum
    vectors changes slowly between optimizer steps. Instead of computing NS
    from scratch (5 steps from normalized gradient), we blend the PREVIOUS
    step's NS output with the new gradient and run fewer NS steps.

    This is NOT the same as blending two NS snapshots (v5). It fundamentally
    changes the NS TRAJECTORY by starting from a different point in matrix
    space — one that's already partially orthogonal.

    Cold start (step 1): standard NS, 5 steps.
    Warm start (step 2+): blend prev_orth with new u, NS from there, fewer steps.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        warm_steps: int = 3,
        inject_rate: float = 0.5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps, warm_steps=warm_steps,
            inject_rate=inject_rate, eps=eps, nesterov=nesterov,
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
            warm_steps = group["warm_steps"]
            inject_rate = group["inject_rate"]
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

                # NS coefficients
                a, b, c = 3.4445, -4.7750, 2.0315

                if "prev_orth" in state:
                    # WARM START: inject new gradient info into previous polar factor
                    u_norm = u / (u.norm() + eps)
                    prev = state["prev_orth"]
                    X = (1.0 - inject_rate) * prev + inject_rate * u_norm
                    X = X / (X.norm() + eps)
                    steps_to_run = warm_steps
                else:
                    # COLD START: standard NS initialization
                    X = u / (u.norm() + eps)
                    steps_to_run = ns_steps

                transpose = m_dim > n_dim
                if transpose:
                    X = X.T

                for _ in range(steps_to_run):
                    A = X @ X.T
                    B = b * A + c * (A @ A)
                    X = a * X + B @ X

                if transpose:
                    X = X.T

                orth = X
                state["prev_orth"] = orth.clone()

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
