"""
Smoothed optimizers: temporal averaging of update directions.

Tests whether the temporal averaging principle (from CUM combined mode)
generalizes beyond NS-based optimizers.

SmoothedAdam: AdamW + EMA of update directions. If smooth_alpha=0, this is AdamW.
"""

import torch
from torch.optim.optimizer import Optimizer


class SmoothedAdam(Optimizer):
    """
    AdamW with temporal averaging of update directions.

    Standard AdamW: update = m_hat / (sqrt(v_hat) + eps)
    SmoothedAdam:   ema_t = beta_s * ema_{t-1} + (1-beta_s) * update_t
                    final = (1-alpha_s) * update_t + alpha_s * ema_t

    Uses decoupled weight decay (AdamW style).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        smooth_beta: float = 0.5,
        smooth_alpha: float = 0.15,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            smooth_beta=smooth_beta, smooth_alpha=smooth_alpha,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            smooth_beta = group["smooth_beta"]
            smooth_alpha = group["smooth_alpha"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                m_hat = m / bc1
                v_hat = v / bc2

                update = m_hat / (v_hat.sqrt() + eps)

                if smooth_alpha > 0:
                    if "update_ema" not in state:
                        state["update_ema"] = update.clone()
                    else:
                        state["update_ema"].mul_(smooth_beta).add_(
                            update, alpha=1 - smooth_beta,
                        )
                    update = (1 - smooth_alpha) * update + smooth_alpha * state["update_ema"]

                p.add_(update, alpha=-lr)

        return loss
