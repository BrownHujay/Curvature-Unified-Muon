import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_orthogonalize
from .utils import aspect_ratio_scale


class CUM3v3(Optimizer):
    """
    CUM 3v3: Directional Momentum.

    Standard Muon momentum averages gradient ELEMENTS (mixing direction and magnitude):
      mb = β*mb + (1-β)*g
    Large-magnitude gradients dominate the average, potentially injecting noise.

    Directional momentum separates direction from magnitude:
      dir_ema = β_dir * dir_ema + (1-β_dir) * (g / ||g||)
      mag_ema = β_mag * mag_ema + (1-β_mag) * ||g||

    All gradient steps contribute equally to the direction regardless of magnitude.
    This prevents large-magnitude noise from hijacking the momentum direction.

    Since NS normalizes everything anyway, only the DIRECTION of the input matters.
    This optimizer feeds a higher-quality direction into NS.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta_dir: float = 0.95,
        beta_mag: float = 0.95,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta_dir=beta_dir, beta_mag=beta_mag,
            ns_steps=ns_steps, eps=eps, nesterov=nesterov,
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
            beta_dir = group["beta_dir"]
            beta_mag = group["beta_mag"]
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
                    state["dir_buffer"] = torch.zeros_like(g)
                    state["mag_buffer"] = 0.0

                state["step"] += 1

                # Separate gradient into direction and magnitude
                g_norm = g.norm()
                g_dir = g / (g_norm + eps)

                # Direction momentum (unit gradient EMA)
                db = state["dir_buffer"]
                db.mul_(beta_dir).add_(g_dir, alpha=1 - beta_dir)

                # Magnitude momentum (scalar EMA)
                state["mag_buffer"] = (
                    beta_mag * state["mag_buffer"]
                    + (1 - beta_mag) * g_norm.item()
                )

                # Construct input for NS
                if nesterov:
                    u = g_dir + beta_dir * db
                else:
                    u = db.clone()

                # NS orthogonalization (magnitude doesn't matter — NS normalizes)
                orth = newton_schulz_orthogonalize(u, steps=ns_steps, eps=eps)

                # Update
                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
