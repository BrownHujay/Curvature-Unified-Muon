import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .newton_schulz import newton_schulz_triple_resolution
from .utils import aspect_ratio_scale


class CUMv8(Optimizer):
    """
    CUM v8: Multi-Scale Curvature Blend.

    Mathematical innovation: Instead of blending TWO NS resolutions (v5),
    blend THREE: NS₁, NS₃, and NS₅. This captures curvature at multiple
    scales simultaneously.

    NS₁ (1 step): ~60% of original SV spread retained. Rich curvature,
                   some noise remains.
    NS₃ (3 steps): ~5% of original SV spread retained. Moderate curvature,
                    well-denoised.
    NS₅ (5 steps): ~0% of original SV spread. Fully orthogonalized.

    Formula:
      update = (1 - w₁ - w₃) * NS₅ + w₃ * scale(NS₃) + w₁ * scale(NS₁)

    Why this should work:
    - v5 showed that NS₂ intermediate (b=0.15) gives -0.0113 over Muon
    - But save@3 was much weaker (-0.0038) — too little curvature left
    - And save@1 was being tested (noisier but richer)
    - Three-point blend lets us get NS₁'s rich curvature with a SMALL weight
      (filtering its noise) while NS₃ provides the stable curvature backbone

    This is like a multi-scale decomposition:
    - NS₅ = base signal (clean direction)
    - NS₃ - NS₅ = medium-frequency curvature
    - NS₁ - NS₃ = high-frequency curvature (noisier)

    Cost: Two extra clones during NS (no extra matmuls). Same NS step count.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        ns_steps: int = 5,
        w1: float = 0.05,
        w3: float = 0.10,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        """
        Args:
            w1: Weight for NS₁ intermediate (high curvature, some noise).
            w3: Weight for NS₃ intermediate (moderate curvature, denoised).
            Total blend = w1 + w3. Weight for NS₅ = 1 - w1 - w3.
        """
        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps,
            w1=w1, w3=w3, eps=eps, nesterov=nesterov,
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
            w1 = group["w1"]
            w3 = group["w3"]
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

                # Phase 1: Momentum
                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                # Phase 2: Triple-Resolution NS
                ns5, ns3, ns1 = newton_schulz_triple_resolution(
                    u, steps=ns_steps, eps=eps,
                )

                # Phase 3: Multi-scale blend
                full_norm = ns5.norm()

                if w1 > 0 or w3 > 0:
                    w5 = 1.0 - w1 - w3
                    orth = w5 * ns5

                    # Blend NS₃
                    if w3 > 0:
                        ns3_norm = ns3.norm()
                        if ns3_norm > eps:
                            orth = orth + w3 * ns3 * (full_norm / ns3_norm)

                    # Blend NS₁
                    if w1 > 0:
                        ns1_norm = ns1.norm()
                        if ns1_norm > eps:
                            orth = orth + w1 * ns1 * (full_norm / ns1_norm)
                else:
                    orth = ns5

                # Phase 4: Weight update
                scale = aspect_ratio_scale(m_dim, n_dim)

                if orig_shape != g.shape:
                    p.data.add_(orth.view(orig_shape), alpha=-lr * scale)
                else:
                    p.add_(orth, alpha=-lr * scale)

        return loss
