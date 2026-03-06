# Beyond Muon: novel mathematical frameworks for matrix-aware neural network optimization

**The most promising path forward is not fixing Newton-Schulz — it's recognizing that NS accidentally implements Schatten-32 steepest descent and designing the spectral processing you actually want.** Muon's polar factor approximation is one point in a rich design space connecting Schatten-p norms, polynomial spectral filters, Lie algebra dynamics, and Frank-Wolfe optimization. The finding that SVs ≈ 0.877 beat exact SVs = 1.0 is explained by a convergence of three theoretical forces: James-Stein-type shrinkage, early-stopping implicit regularization, and gradient noise robustness. Below are 12 genuinely different mathematical directions, ranked by likelihood of beating Muon, with equations, costs, and implementation details.

---

## Why 0.877 beats 1.0: three forces converging

The empirical puzzle — approximate orthogonalization outperforming exact — is now well-understood theoretically. Three independent mechanisms explain it, and their convergence suggests the optimal spectral processing is neither identity nor full equalization but something specific in between.

**Schatten-p interpolation.** Franz Louis Cesista (2025) demonstrated that Muon's NS iteration with standard coefficients does not implement Schatten-∞ steepest descent (perfect equalization). The residual SV variance matches **Schatten-32 steepest descent**, where the update direction under Schatten-p norm for gradient $G = U\Sigma V^\top$ is:

$$\Delta^* = -U \cdot \operatorname{diag}\!\left(\frac{\sigma_i^{p-1}}{\|\sigma\|_p^{p-1}}\right) V^\top$$

At $p = 2$ this recovers standard SGD (proportional to $\sigma_i$); at $p = \infty$ it gives all SVs = 1 (exact polar factor). At $p = 32$, singular values are nearly equalized but retain slight ordering — a "soft equalization" that preserves some gradient magnitude information. The optimal $p$ likely depends on batch size and training duration: larger batches and longer training favor higher $p$ (less noise to regularize against).

**James-Stein shrinkage.** In dimensions ≥ 3, shrinkage estimators always dominate unbiased estimators (the James-Stein paradox). Mapping SVs to 0.877 instead of 1.0 is a **12.3% shrinkage** that reduces total risk. From the Donoho-Gavish optimal shrinkage framework, the optimal spectral shrinker for noisy matrices maps observed singular values below a threshold to zero and shrinks those above — never mapping to the raw observed value. The NS polynomial's convergence profile naturally implements differential shrinkage: small input SVs converge more slowly to the fixed point, receiving more relative shrinkage than large SVs.

**Early-stopping regularization.** NS is an iterative method converging to the sign function. Stopping at 5 iterations of degree-5 polynomials produces a smooth spectral filter $g_5(\sigma)$ rather than the discontinuous $\operatorname{sign}(\sigma)$. Classical results from inverse problems (Landweber iteration, spectral filtering theory) establish that early stopping at iteration $k$ is equivalent to Tikhonov regularization with parameter $\lambda \sim 1/k$. The mid-iteration blend improvement (saving step 2, blending 15% into final output) works because it creates a richer polynomial spectral filter — effectively a **degree-10 polynomial** combining steps 2 and 5, with a more carefully shaped SV mapping profile than either alone.

The reason **weight decay kills Muon** is now understood via the Lion-K framework (Chen et al., 2025): Muon with decoupled weight decay implicitly solves $\min_X F(X)$ subject to $\|X\|_{\text{op}} \leq 1/\lambda$. Weight decay doesn't regularize the Frobenius norm of weights (as with Adam) — it constrains the **spectral norm**, which may be too aggressive or mismatched with the training objective at typical $\lambda$ values.

---

## Direction 1: optimal polynomial spectral filters via the Remez algorithm

The Polar Express framework (Amsel, Persson, Musco, Gower, 2025; arXiv:2505.16932) is the most immediately actionable improvement. Rather than using NS's fixed polynomial $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ at every iteration, it computes a **different minimax-optimal polynomial at each step** by solving:

$$p_t^* = \arg\min_{p \in \mathcal{P}_d^{\text{odd}}} \max_{\sigma \in [\ell_t, u_t]} |p(\sigma) - \operatorname{sign}(\sigma)|$$

where $[\ell_t, u_t]$ is the SV interval at iteration $t$. The coefficients are found via the Remez algorithm (equioscillation theorem) and precomputed offline in float64. **Already demonstrated to consistently improve Muon's validation loss** on GPT-2 training across learning rates.

But here's the key insight for the researcher: the Remez machinery applies to **any target function**, not just $\operatorname{sign}(\sigma)$. You can design the optimal degree-5 odd polynomial approximation to any desired SV mapping:

- **Soft equalization:** $g(\sigma) = \tanh(\beta\sigma)/\tanh(\beta)$ for tunable sharpness $\beta$
- **Power compression:** $g(\sigma) = \sigma^\alpha$ for $\alpha \in (0,1)$, reducing dynamic range while preserving ordering
- **Schatten-p target:** $g(\sigma) = \sigma^{p-1}/\|\sigma\|_p^{p-1}$ for specific $p$

**Cost:** Same as NS — only matrix-matrix multiplications. With $d=5, T=5$: ~10 matmuls total (vs NS's 15). **Faster and better.**

```python
# Polar Express: precomputed optimal coefficients (from Remez algorithm offline)
OPTIMAL_COEFFS = [  # T=5, d=5, ell=1e-3
    (1.80235, -0.72463, 0.05918),  # iteration 1 (different from NS!)
    (2.42117, -2.14325, 0.38914),  # iteration 2
    # ... precomputed for each step
]

def polar_express(M, coeffs=OPTIMAL_COEFFS):
    X = M / (M.norm() + 1e-2)  # initial scaling
    for (a1, a3, a5) in coeffs:
        A = X.T @ X
        X = X @ (a1 * I + a3 * A + a5 * A @ A)  # 2 matmuls per step
    return X
```

**Why it might beat NS:** Provably optimal convergence rate in its class. The per-step adaptation means early iterations handle the bulk spectrum efficiently while later iterations refine the near-converged region. NS uses the same suboptimal polynomial at every step.

---

## Direction 2: explicit Schatten-p steepest descent with tunable p

Instead of relying on NS to accidentally produce a particular Schatten-p behavior, implement the Schatten-p steepest descent direction directly. This requires computing singular values (but not necessarily full SVD — see cost analysis below).

$$\Delta^{(p)} = -U \cdot \operatorname{diag}\!\left(\sigma_i^{p-1}\right) \cdot V^\top \cdot \|\sigma\|_p^{1-p}$$

For large $p$ (e.g., 16–64), this nearly equalizes SVs while preserving slight ordering. The normalization $\|\sigma\|_p^{1-p}$ ensures unit dual norm.

**Implementation via polynomial approximation (avoiding SVD):** The function $\sigma \mapsto \sigma^{p-1}$ for even $p-1$ can be expressed as a matrix polynomial. For $p = 5$: $\sigma^4 = (\sigma^2)^2$, applied as $G(G^\top G)(G^\top G)$. For general $p$, use Chebyshev polynomial approximation of $\sigma^{p-1}$ on the SV interval.

```python
def schatten_p_descent(G, p=32):
    """Steepest descent direction under Schatten-p norm."""
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    S_transformed = S.pow(p - 1)
    S_transformed = S_transformed / S_transformed.norm(p=p/(p-1))  # dual norm
    return U @ torch.diag(S_transformed) @ Vh

# SVD-free version via polynomial approximation (for large p):
def schatten_p_poly(G, p=32, degree=8):
    """Approximate Schatten-p descent via polynomial in G^T G."""
    # Chebyshev approximation of sigma^{p-1} on estimated SV interval
    coeffs = chebyshev_fit(lambda s: s**((p-1)/2), degree, interval=[0, est_smax])
    A = G.T @ G  # Gram matrix
    result = coeffs[0] * G
    Ak = A.clone()
    for c in coeffs[1:]:
        result = result + c * G @ Ak
        Ak = Ak @ A
    return result / result.norm()  # normalize
```

**Cost:** Full SVD version: $O(mn \cdot \min(m,n))$ — expensive but exact. Polynomial version: $(d/2)$ matmuls for degree-$d$ polynomial, comparable to NS. **Sweep $p \in \{4, 8, 16, 32, 64, 128\}$ to find the optimum for your setup.**

---

## Direction 3: Lie algebra momentum with Cayley retraction

This is the most **fundamentally different** framework from Muon. Instead of: (1) element-wise EMA momentum → (2) NS orthogonalization, it does: (1) project gradient into the Lie algebra → (2) EMA in Lie algebra (a flat vector space — no transport needed) → (3) retract via Cayley transform.

The critical insight from Brasselet et al. (2023) and Lezcano-Casado (2019): for homogeneous spaces like the Stiefel and orthogonal groups, the Lie algebra provides a **global tangent space representation**. All tangent spaces can be identified with one common vector space (skew-symmetric matrices), eliminating the need for vector transport entirely.

**Mathematical formulation:**

$$\Omega_t = \frac{1}{2}(G_t W_t^\top - W_t G_t^\top) \quad \text{(skew-symmetric projection into } \mathfrak{so}(m)\text{)}$$

$$\bar{\Omega}_t = \beta \bar{\Omega}_{t-1} + (1-\beta)\Omega_t \quad \text{(EMA in flat Lie algebra — no transport!)}$$

$$W_{t+1} = \operatorname{Cayley}(\eta \bar{\Omega}_t) \cdot W_t = (I + \tfrac{\eta}{2}\bar{\Omega}_t)^{-1}(I - \tfrac{\eta}{2}\bar{\Omega}_t) W_t$$

The Cayley transform maps skew-symmetric → orthogonal exactly, and the iterative version avoids the matrix inverse:

```python
def lie_algebra_optimizer_step(W, G, Omega_bar, beta, lr):
    # 1. Project gradient into Lie algebra (skew-symmetric)
    A = G @ W.T
    Omega = 0.5 * (A - A.T)  # skew-symmetric
    
    # 2. EMA in Lie algebra (flat space, no vector transport!)
    Omega_bar = beta * Omega_bar + (1 - beta) * Omega
    
    # 3. Iterative Cayley retraction (no matrix inverse)
    scaled = lr * Omega_bar
    Y = W.clone()
    for _ in range(4):  # 4 fixed-point iterations
        Y = W - 0.5 * scaled @ (W + Y)
    
    return Y, Omega_bar
```

**Cost:** 4 matmuls for iterative Cayley + 2 matmuls for Lie algebra projection = **6 matmuls total** (vs NS's 15). **2.5× cheaper than Muon.**

**Why it might beat NS:** (a) Momentum is handled in a geometrically correct space without the Euclidean-then-orthogonalize pattern that destroys pre-NS modifications. (b) The Cayley transform is mathematically exact — no polynomial approximation error. (c) The reason Cayley retraction previously "failed" may be that it was used as a drop-in NS replacement without restructuring momentum to use the Lie algebra. This approach restructures the entire optimizer around the manifold geometry.

**Key caveat:** This naturally produces orthogonal updates (SVs = 1.0 exactly). To get the beneficial ~0.877 shrinkage, add a scaling factor or blend with the gradient: $\Delta W = \alpha \cdot \text{Cayley update} + (1-\alpha) \cdot G / \|G\|_F$ for $\alpha \approx 0.85$.

---

## Direction 4: PolarGrad with nuclear norm scaling

PolarGrad (Lau, Long, Su, 2025; arXiv:2505.21799) identifies a critical theoretical gap in Muon: it lacks **null-gradient consistency** — when gradients are small, Muon's update magnitude doesn't shrink because the polar factor always has SVs near 1. PolarGrad fixes this:

$$W_{t+1} = W_t - \eta \cdot \|G_t\|_* \cdot \operatorname{msign}(G_t)$$

where $\|G_t\|_* = \sum_i \sigma_i(G_t)$ is the nuclear norm and $\operatorname{msign}(G_t) = UV^\top$ is the matrix sign (polar factor). The nuclear norm acts as an **adaptive step size** that scales the update by the total gradient magnitude.

```python
def polargrad_step(W, G, lr):
    # Compute polar factor (via NS, Polar Express, or any method)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    polar = U @ Vh  # matrix sign
    nuclear_norm = S.sum()  # sum of singular values
    W = W - lr * nuclear_norm * polar
    return W
```

**Cost:** Same as Muon + one SVD for the nuclear norm (or approximate it cheaply from NS's intermediate values).

**Why it might beat Muon:** PolarGrad achieves **linear convergence** for strongly convex problems where Muon does not. The nuclear norm scaling provides gradient-magnitude-adaptive step sizes without element-wise second moments (avoiding Adam's overhead). It's the theoretically "correct" version of Muon.

---

## Direction 5: PSGD Kron — Lie group preconditioner estimation

PSGD (Xi-Lin Li, 2015–2025; arXiv:1512.04202) is a genuinely different second-order optimizer that learns a **Kronecker-factored preconditioner via Lie group optimization**. Instead of computing a polar factor, it fits a preconditioner $P = Q_1^\top Q_1 \otimes Q_2^\top Q_2$ by minimizing:

$$\min_{Q_1, Q_2} \;\mathbb{E}\!\left[\delta g^\top (Q_1^\top Q_1 \otimes Q_2^\top Q_2)\, \delta g + \delta\theta^\top (Q_1^\top Q_1 \otimes Q_2^\top Q_2)^{-1}\, \delta\theta\right]$$

This makes $P \approx H^{-1}$ (inverse Hessian). The Kronecker factors $Q_1, Q_2$ are updated via multiplicative steps on the general linear group $GL(n)$, ensuring $P$ remains positive definite without projection.

```python
def psgd_kron_step(W, G, Q1, Q2, momentum, beta, lr, precond_lr=0.1):
    # 1. Momentum
    momentum = beta * momentum + (1 - beta) * G
    
    # 2. Update preconditioner (probabilistically, e.g., 10% of steps)
    if should_update_precond():
        # Whitening-based: fit Q so that Q@g has identity covariance
        Qg = Q1 @ momentum @ Q2.T  # preconditioned gradient
        # Lie group gradient step on Q1, Q2
        Q1 = Q1 - precond_lr * (Qg @ Qg.T - I) @ Q1 / G.shape[0]
        Q2 = Q2 - precond_lr * (Qg.T @ Qg - I) @ Q2 / G.shape[1]
    
    # 3. Apply preconditioner
    update = Q1.T @ (Q1 @ momentum @ Q2.T) @ Q2
    
    # 4. Update
    W = W - lr * update
    return W, Q1, Q2, momentum
```

**Cost:** $O(m^2 n + mn^2)$ per step for Kron application. For 128×128: ~$2 \times 128^3 \approx 4M$ FLOPs. Preconditioner update adds $O(m^3 + n^3)$ but runs only 3-10% of steps. **Comparable to NS.**

**Why it might beat Muon:** PSGD learns the curvature of the loss landscape rather than applying a fixed spectral transformation. It finds **flatter minima** (better generalization) and doesn't rely on weight decay. The Lie group parameterization ensures numerical stability without the polynomial approximation issues of NS. The "What Really Matters in Matrix-Whitening Optimizers" meta-analysis (2025) found PSGD Kron competitive with both Muon and SOAP.

---

## Direction 6: Dion — amortized power iteration for distributed orthonormalization

Dion (Ahn, Xu et al., Microsoft, 2025; arXiv:2504.05295) replaces NS with **rank-$r$ power iteration plus error feedback**, which is both cheaper and more communication-efficient:

$$Z_t = M_t (M_t^\top Z_{t-1}) \quad \text{(one power iteration, warm-started)}$$
$$Q_t, R_t = \operatorname{QR}(Z_t) \quad \text{(small } m \times r \text{ QR)}$$
$$B_t = Q_t^\top M_t \quad \text{(project momentum to rank-}r\text{)}$$
$$\text{update}_t = Q_t U_B V_B^\top \quad \text{(from small SVD of } B_t \text{)}$$
$$M_t \leftarrow M_t - Q_t B_t \quad \text{(error feedback: keep unexplained variance)}$$

At full rank ($r = \min(m,n)$), Dion **outperforms Muon** at larger batch sizes, hypothesized to be because QR gives exact orthonormalization while NS introduces approximation noise. At reduced rank ($r = d/4$), it remains competitive while drastically reducing communication in distributed settings.

**Cost:** $O(mnr + mr^2)$ per step. At full rank, comparable to NS. At $r = d/4$ for 128-dim: **4× cheaper.**

---

## Direction 7: Zolotarev rational approximations for 2-iteration convergence

Zolotarev (1877) proved that the best degree-$(m,n)$ rational approximation to $\operatorname{sign}(x)$ achieves error decaying as $4\exp(-n\pi^2/2\mu)$ — **exponentially faster** than any polynomial. Nakatsukasa & Freund (2016, SIAM Review) showed that high-degree Zolotarev functions decompose as compositions of low-degree ones: a degree-17 Zolotarev function (machine-precision accuracy for $\kappa \leq 10^{16}$) factors as two degree-$\sqrt{17}$ applications.

**The ZOLO-PD algorithm converges in exactly 2 iterations** regardless of conditioning, compared to NS's 5-50 iterations.

$$X_{k+1} = X_k \cdot r(X_k^\top X_k) \quad \text{where } r \text{ is optimal Zolotarev rational function}$$

Each application of $r$ requires solving shifted linear systems $(X_k^\top X_k + c_j I)^{-1}$ for several shifts $c_j$, implementable via QR or conjugate gradient.

**Cost:** ~$43n^3$ total FLOPs for ill-conditioned matrices (vs NS's ~$30n^3$ for 5 degree-5 steps). Slightly more expensive but provides **machine-precision** polar factor.

**Why it might beat NS:** If the 0.877 vs 1.0 question is actually about the **SV mapping profile** (not the target value), then Zolotarev's sharper sigmoid-like transition could provide a different and potentially better spectral filter than NS's blunt polynomial. Moreover, the rational function can be "softened" by modifying the Zolotarev parameters to target a custom SV mapping.

**Limitation:** Requires matrix inversions (not bfloat16-friendly). Best suited for float32 experiments or hybrid-precision schemes. For 128×128, the inverse costs ~$2n^3 \approx 4M$ FLOPs — feasible.

---

## Direction 8: MARS — variance reduction meets preconditioning

MARS (Gu et al., 2024; arXiv:2411.10438, ICML 2025) takes an orthogonal approach: instead of better spectral processing, it reduces the **variance** of the preconditioned gradient estimate via scaled stochastic recursive momentum:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \gamma \beta_1 \cdot \mathcal{P}(g_t - g_{t-1}))$$

where $\mathcal{P}$ is any preconditioner (Adam, Muon, Shampoo) and $\gamma$ controls variance reduction strength. The key insight: variance reduction and preconditioning **conflict** unless the variance-reduced estimator is scaled by the preconditioner.

**MARS consistently outperforms both AdamW and Muon on GPT-2 models** (Small through XL). It's a meta-framework that wraps around any base optimizer.

```python
def mars_step(W, G, G_prev, m, v, beta1, beta2, gamma, lr):
    # Variance-reduced gradient estimate
    correction = gamma * beta1 * precondition(G - G_prev)  # P(g_t - g_{t-1})
    m = beta1 * m + (1 - beta1) * (G + correction)
    v = beta2 * v + (1 - beta2) * G**2  # second moment (if using Adam base)
    update = m / (v.sqrt() + eps)  # or apply Muon's NS to m instead
    W = W - lr * update
    return W, m, v
```

**Cost:** ~2× the base optimizer (one extra gradient evaluation stored, one extra preconditioner application). **When wrapping Muon: ~2× Muon cost.**

---

## Direction 9: warm-started NS from previous step's polar factor

A surprisingly simple idea that exploits temporal coherence: since momentum $M_t$ changes slowly between steps, the polar factor $U_t V_t^\top$ is close to $U_{t-1} V_{t-1}^\top$. Using the previous polar factor as the NS starting point should require **far fewer iterations** to converge.

$$X_0^{(t)} = U_{t-1} V_{t-1}^\top \quad \text{(warm start instead of scaled } M_t\text{)}$$

If $\|U_t V_t^\top - U_{t-1}V_{t-1}^\top\|$ is small (which it will be with momentum $\beta \geq 0.9$), even **2 NS iterations** from this warm start should suffice, reducing cost from 15 matmuls to 6.

```python
def warmstarted_ns(M, prev_polar, ns_coeffs, num_steps=2):
    X = prev_polar  # warm start
    for a, b, c in ns_coeffs[:num_steps]:
        A = X.T @ X
        X = X @ (a * I + b * A + c * A @ A)
    return X
```

**Cost:** 2 steps × 2 matmuls = **4 matmuls** (vs 15 for standard Muon). **3.75× cheaper.**

**Why this is interesting:** The warm start changes the SV mapping profile — starting near SVs = 1 means 2 NS steps barely modify the values, giving SVs ≈ 0.98–1.0 rather than 0.877. This tests whether the optimal SV target depends on how "fresh" the direction is. You could tune the number of warm-started steps (1–3) to find the optimal spectral profile.

---

## Direction 10: optimal singular value shrinkage from random matrix theory

The Donoho-Gavish framework provides **closed-form optimal shrinkage** for noisy matrices under the spiked model. For a matrix $G = S + N$ where $S$ is signal and $N \sim \mathcal{N}(0, \sigma^2/n)$, the optimal shrinker under operator norm loss is:

$$\eta^*(\sigma_i) = \begin{cases} \frac{1}{\sigma_i}\sqrt{(\sigma_i^2 - \beta - 1)^2 - 4\beta} & \text{if } \sigma_i > 1 + \sqrt{\beta} \\ 0 & \text{otherwise}\end{cases}$$

where $\beta = m/n$ is the aspect ratio. This is the **Baik-Ben Arous-Péché (BBP) transition**: singular values below $1 + \sqrt{\beta}$ are pure noise and should be zeroed; those above should be shrunk by a specific factor.

```python
def optimal_shrinkage_update(G, beta_ratio, noise_std):
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    threshold = noise_std * (1 + beta_ratio**0.5)
    # Optimal Donoho-Gavish shrinker
    mask = S > threshold
    S_shrunk = torch.where(
        mask,
        torch.sqrt(torch.clamp((S**2 - beta_ratio - 1)**2 - 4*beta_ratio, min=0)) / S,
        torch.zeros_like(S)
    )
    return U @ torch.diag(S_shrunk) @ Vh
```

**Why it might beat NS:** NS applies the same polynomial regardless of which SVs are signal vs noise. Optimal shrinkage zeros noise components and shrinks signal components by the theoretically optimal amount. The challenge is estimating the noise level $\sigma$ from stochastic gradient batches.

---

## Direction 11: the Procrustes-momentum connection and orthogonal tracking

The regularized orthogonal Procrustes problem has a clean closed-form that reveals Muon's structure:

$$Q^* = \arg\min_{Q^\top Q = I} \|Q - G\|_F + \lambda\|Q - U_{t-1}\|_F = \operatorname{polar}\!\left(\frac{G + \lambda U_{t-1}}{1+\lambda}\right)$$

This is exactly Muon's momentum-then-orthogonalize pattern. But there's a **genuinely different** variant: the **weighted orthogonal Procrustes** where different singular directions get different weights:

$$Q^* = \arg\min_{Q^\top Q = I} \sum_i w_i \|Q e_i - G e_i\|^2$$

With learned or adaptive weights $w_i$, this allows the optimizer to prioritize certain gradient directions over others during orthogonalization — a form of **direction-aware spectral processing** that NS cannot do.

---

## Direction 12: subspace tracking via GROUSE geodesic updates

GROUSE (Balzano, Nowak, Recht, 2010) performs incremental gradient descent on the Grassmannian via rank-1 geodesic updates, tracking a $k$-dimensional subspace at $O(nk)$ cost per step:

$$U_{t+1} = U_t + \frac{\sin(\sigma\eta)}{\|r\|} r \left(\frac{\cos(\sigma\eta)}{\|p\|} p - \frac{w}{\|w\|}\right)^\top$$

where $r = G - U_t(U_t^\top G)$ is the residual, $w = U_t^\top G$ is the projection, $p = U_t w$, and $\sigma$ controls the geodesic step. This is a **proper geodesic update** on $\operatorname{Gr}(n,k)$, not an approximation.

**Cost:** $O(nk)$ per column update. For tracking a rank-64 subspace in 128-dim: ~16K FLOPs. **Orders of magnitude cheaper than NS.** The question is whether a low-rank update suffices — empirically, Dion shows rank $d/4$ is competitive.

---

## How these directions address each research question

**Q1 (Matrix-aware optimization beyond polar factor):** Directions 3 (Lie algebra), 5 (PSGD Kron), and 12 (GROUSE) operate on fundamentally different mathematical objects — skew-symmetric matrices, Kronecker preconditioners, and Grassmannian subspaces respectively. Direction 2 (Schatten-p) generalizes the polar factor to a continuous family.

**Q2 (Why SV equalization helps):** The Schatten-p interpolation, James-Stein shrinkage, and early-stopping regularization frameworks collectively explain that NS implements an effective Schatten-32 descent with beneficial implicit regularization. The optimal SV target is below 1.0 by an amount that depends on the gradient signal-to-noise ratio.

**Q3 (Information-theoretic perspective):** NS is lossy compression of the gradient that preserves all singular vector pairs (direction information) while compressing singular values (magnitude information). The ~0.877 mapping is a rate-distortion optimal point that retains some magnitude information while mostly equalizing. Direction 10 (optimal shrinkage) provides the information-theoretically optimal compression under specific noise models.

**Q4 (Novel mathematical ingredients):** Matrix means (Direction 3's Lie algebra EMA), Procrustes (Direction 11), optimal transport on SVs (Direction 10's shrinkage), polynomial filtering (Direction 1), Zolotarev rational functions (Direction 7), and matrix sign function methods (covered in Direction 7) are all addressed with equations and pseudocode.

**Q5 (Cross-disciplinary methods):** Control theory contributed the Lyapunov stability insight that Muon's Frank-Wolfe structure ensures bounded iterates. Signal processing contributed GROUSE (Direction 12) and subspace tracking. Quantum computing contributed the Lie algebra / exponential map framework (Direction 3). PSGD's Lie group preconditioner (Direction 5) draws from differential geometry.

**Q6 (Scaling considerations):** At scale, Dion (Direction 6) is purpose-built for distributed Muon. Moonshot AI's Moonlight paper showed Muon scales to 16B with careful per-parameter update scaling ($\sqrt{\max(m,n)} \times 0.2$). Kimi K2 scaled to 1T parameters with MuonClip for attention stability. At larger dimensions, the Schatten-p optimal likely shifts toward higher $p$ (more equalization) because the effective rank of gradient matrices grows.

---

## Recommended experimental priority

The following ordering maximizes the probability of finding an improvement while minimizing wasted experiments:

- **Experiment 1 — Polar Express drop-in** (Direction 1): Lowest risk, already proven. Swap NS coefficients for Remez-optimal per-step coefficients. Expected: -0.005 to -0.015 val_loss improvement. Time: 1 hour to implement.
- **Experiment 2 — Schatten-p sweep** (Direction 2): Implement explicit Schatten-p for $p \in \{8, 16, 32, 64, 128\}$. The optimal $p$ directly answers "what spectral processing does the model want?" Use SVD-based implementation first for correctness, then switch to polynomial approximation.
- **Experiment 3 — Warm-started NS** (Direction 9): Trivially easy to implement (cache previous polar factor). Test 1, 2, 3 warm-started NS steps. Expected: significant speedup with minimal quality loss.
- **Experiment 4 — MARS wrapping Muon** (Direction 8): Variance reduction is orthogonal to all other improvements. Can be combined with any spectral method. Expected: consistent improvement based on published GPT-2 results.
- **Experiment 5 — Lie algebra optimizer** (Direction 3): Most fundamentally different architecture. The restructured momentum handling may unlock improvements that no NS modification can reach. Test with and without SV shrinkage blending.
- **Experiment 6 — PSGD Kron** (Direction 5): Existing PyTorch implementation at github.com/lixilinx/psgd_torch. Test as complete Muon replacement. If it works at 128-dim, the learned curvature information provides something no Muon variant can match.
- **Experiment 7 — Optimal SV shrinkage** (Direction 10): Estimate gradient noise level from batch variance of singular values, apply Donoho-Gavish shrinkage instead of NS. Tests whether adaptive, noise-aware spectral processing outperforms fixed polynomials.

Each subsequent experiment should be informed by results from earlier ones — particularly the Schatten-p sweep, which will reveal the shape of the optimal spectral processing function and guide all other directions.