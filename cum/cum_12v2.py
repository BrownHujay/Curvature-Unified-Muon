"""
CUM 12v2: TD(lambda) Blending.

Instead of blending just NS_2 + NS_5, exponentially weight ALL NS iterates:
  w_k = lambda^(n-k)  (unnormalized, then normalized to sum=1)

lambda=0: pure NS_n (standard Muon)
lambda=0.3: mostly NS_5 with light mixing from NS_4/3
lambda=0.7: broader spread across iterates
lambda=1.0: uniform average of all iterates

Previous three-point (uniform NS_1+NS_3+NS_5) failed because NS_1 is too noisy.
TD(lambda) naturally downweights early (noisy) iterates via exponential decay.

Optionally stacks temporal EMA on top (combined-style).
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_n_resolution
from .utils import aspect_ratio_scale


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
    if weight <= 0:
        return primary
    p_norm = primary.norm()
    s_norm = secondary.norm()
    if s_norm > eps:
        secondary_scaled = secondary * (p_norm / s_norm)
        return (1 - weight) * primary + weight * secondary_scaled
    return primary


class CUM12v2(Optimizer):
    """
    TD(lambda) exponentially-weighted NS iterate blending.

    Saves all NS intermediates and blends with geometric decay weights.
    Optionally adds temporal EMA (across-step) on top.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        td_lambda: float = 0.5,
        ns_steps: int = 5,
        use_temporal: bool = True,
        input_blend_beta: float = 0.5,
        input_blend_alpha: float = 0.15,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr, beta1=beta1,
            td_lambda=td_lambda, ns_steps=ns_steps,
            use_temporal=use_temporal,
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
            td_lambda = group["td_lambda"]
            ns_steps = group["ns_steps"]
            use_temporal = group["use_temporal"]
            eps = group["eps"]
            nesterov = group["nesterov"]

            # Precompute TD weights: w_k = lambda^(n-k), normalized
            raw_weights = [td_lambda ** (ns_steps - k) for k in range(1, ns_steps + 1)]
            total_w = sum(raw_weights)
            weights = [w / total_w for w in raw_weights]

            # save_at for all intermediates except the last (which is 'final')
            save_at = tuple(range(1, ns_steps))  # (1, 2, ..., n-1)

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

                # Get all NS intermediates: (NS_n, NS_1, NS_2, ..., NS_{n-1})
                all_results = newton_schulz_n_resolution(
                    u, steps=ns_steps, save_at=save_at, eps=eps,
                )
                # Reorder to [NS_1, NS_2, ..., NS_n]
                final = all_results[0]
                intermediates = list(all_results[1:])  # [NS_1, ..., NS_{n-1}]
                all_iterates = intermediates + [final]  # [NS_1, ..., NS_n]

                # Norm-match all to final's norm, then weighted sum
                f_norm = final.norm()
                blended = torch.zeros_like(final)
                for k, (iterate, w) in enumerate(zip(all_iterates, weights)):
                    i_norm = iterate.norm()
                    if i_norm > eps:
                        blended.add_(iterate * (f_norm / i_norm), alpha=w)
                    else:
                        blended.add_(final, alpha=w)

                # Optional temporal EMA on top
                if use_temporal:
                    input_beta = group["input_blend_beta"]
                    input_alpha = group["input_blend_alpha"]
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
                    orth = blended

                scale = aspect_ratio_scale(m_dim, n_dim)
                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
