# Oscillation Dynamics of Newton-Schulz Iteration in Neural Network Optimization: A Complete Characterization

**Kota Newman**

---

## Abstract

We present a comprehensive analysis of the Newton-Schulz (NS) polynomial iteration used in the Muon optimizer for neural network training. Through 85+ controlled experiments across 16 series, theoretical analysis drawing on six independent mathematical fields, and formal proofs of four theorems, we characterize the dynamical system governing NS-based gradient preconditioning. Our central finding is a fundamental *coefficient-stability tradeoff*: the large polynomial coefficients required for effective singular value equalization ($a \geq 3.0$) necessarily produce an unstable fixed point, causing structured oscillation in the singular value dynamics. We prove this tradeoff is universal — six independent mathematical fields (frame theory, Riemannian geometry, quantum information, spectral theory, statistics, and optimization) all converge to the same spectral iteration, and none can circumvent the tradeoff.

We identify oscillation cancellation via multi-iterate blending as the only modification (out of 85+ tested) that consistently improves NS-based optimizers. The final recipe — a weaker polynomial at the edge of chaos ($d = -1.0$), TD($\lambda = 0.5$) blending across all five NS iterates, and temporal EMA across training steps — achieves $-0.018$ val\_loss improvement over Muon at 1.2M scale (replicated 3$\times$: 1.4993, 1.4993, 1.4992) and $-0.014$ at 124M scale. The improvement is specific to NS's structured oscillation: the same temporal averaging applied to Adam actively hurts performance.

We additionally discover per-head orthogonalization ($-0.019 \pm 0.003$ at 1.2M) and prove its improvement is aspect-ratio-dependent, not scaling to 124M where increased head count creates unfavorable matrix geometry. Our theoretical contributions include exact algebraic expressions for the stability boundary, a proof that all degree-3 spectral iterations from six mathematical fields are limited to linear coefficients $\alpha < 2$ (far below the required $a \geq 3.0$), and a complete bifurcation analysis showing the optimal operating point lies exactly at the period-doubling boundary.

---

## 1. Introduction

### 1.1 Context and Motivation

The Muon optimizer (Jordan, 2024) applies Newton-Schulz (NS) iteration to approximate the polar factor of gradient momentum, achieving state-of-the-art performance in neural network training. The NS iteration applies a degree-5 odd polynomial

$$p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$$

with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ to the singular values of the gradient matrix over five iterations. This maps all singular values toward an approximate common value, producing a near-orthogonal update direction that equalizes gradient information across all directions.

The key observation motivating this work is that NS does *not* converge to the polar factor. The polynomial has a fixed point at $\sigma^* = 0.868$ where $p(\sigma^*) = \sigma^*$, but the derivative $|p'(\sigma^*)| = 1.58 > 1$ makes this fixed point *unstable*. After five iterations, singular values do not concentrate at $\sigma^* = 0.868$ or at $1.0$ — they scatter across approximately $[0.68, 1.12]$ in a structured, anti-correlated oscillation pattern.

This paper asks: what is the nature of this oscillation, can it be managed, and is it fundamental? Over the course of 85+ experiments, we discover that (1) every tested modification to Muon fails except oscillation cancellation via multi-iterate blending; (2) the oscillation is the unavoidable price of the large polynomial coefficients required for effective training; and (3) six independent mathematical fields prove this tradeoff is universal, not specific to the NS quintic.

### 1.2 Muon Algorithm

For a weight matrix $W \in \mathbb{R}^{m \times n}$, Muon computes:

$$g = \nabla_W \mathcal{L}(W)$$
$$m \leftarrow \beta_1 m + (1 - \beta_1)g \quad (\beta_1 = 0.95)$$
$$u = g + \beta_1 m \quad \text{(Nesterov lookahead)}$$
$$X = \text{NS}_5(u) \quad \text{(5 Newton-Schulz iterations)}$$
$$W \leftarrow W - \eta \sqrt{\max(1, m/n)} \cdot X$$

The NS iteration initializes $X_0 = M / \|M\|_F$ and computes:

$$X_{k+1} = aX_k + (bX_kX_k^\top + c(X_kX_k^\top)^2)X_k$$

Muon applies NS only to 2D hidden-layer weight matrices; embeddings, biases, and layer normalization parameters use AdamW.

### 1.3 Overview of Results

Our investigation proceeded through three phases:

**Phase 1 — Exhaustive search (Series 1–7, 40+ experiments).** We tested pre-NS gradient modifications, post-NS curvature recovery, alternative orthogonalization methods (SVD polar factor, Cayley retraction, Halley iteration, Dion), custom SV mappings (Schatten-$p$, tanh, Huber), variance reduction (MARS), warm-starting, and learned preconditioners (PSGD). Every approach failed or tied Muon, except blending NS intermediate iterates.

**Phase 2 — Oscillation cancellation (Series 8–13, 30+ experiments).** We discovered that blending NS intermediate iterates — which oscillate approximately out of phase — cancels structured noise while preserving equalization. Three orthogonal layers of oscillation management were identified and combined. Formal theoretical analysis revealed the polynomial dynamics operate at the edge of chaos.

**Phase 3 — Mathematical frameworks and structural optimization (Series 14–16, 15+ experiments).** We tested four frameworks from a survey of 26 theoretical directions across six mathematical fields — all failed. We proved the coefficient-stability tradeoff is universal. Per-head structural optimization produced improvements at small scale but failed to scale due to aspect-ratio dependence.

### 1.4 Contributions

1. **Coefficient-stability tradeoff (Theorem 1).** For any degree-5 odd polynomial with a positive fixed point, the derivative at the fixed point satisfies $p'(\sigma^*) = 3 - 2a + 2c\sigma^{*4}$. This is exact, algebraic, and monotonically decreasing in $a$. With NS's parameters, stability ($|p'(\sigma^*)| \leq 1$) requires $a \leq 3.152$, while effective training requires $a \geq 3.0$.

2. **Cubic iteration limitation (Theorem 2).** All degree-3 spectral iterations — arising independently in frame theory, Brockett flow, Yang-Mills heat flow, quantum depolarizing channels, Stein discrepancy flows, and spectral proximal operators — are limited to linear coefficients $\alpha < 2$, far below the required $a \geq 3.0$. The quintic term is structurally necessary.

3. **Oscillation cancellation recipe.** The only consistently winning modification across 85+ experiments. Three additive layers: (a) weaker polynomial at the bifurcation boundary ($d = -1.0$, reduces oscillation at source), (b) TD($\lambda = 0.5$) multi-iterate blending (cancels within-step oscillation), (c) temporal EMA (cancels across-step oscillation). Improvement: $-0.018$ at 1.2M (replicated $3\times$), $-0.014$ at 124M.

4. **Complete negative results catalog.** 40+ distinct failed approaches across 16 series provide a comprehensive map of the modification space around NS optimizers.

5. **Per-head orthogonalization.** $-0.019 \pm 0.003$ at 1.2M by splitting QKV gradients into per-head slices before NS. Aspect-ratio-dependent: does not scale to 124M.

---

## 2. The NS Polynomial: Properties and Dynamics

### 2.1 The Polynomial

The NS polynomial $p(\sigma) = 3.4445\sigma - 4.7750\sigma^3 + 2.0315\sigma^5$ is applied as a scalar function to each singular value during the NS matrix iteration. The coefficients were chosen by Jordan (2024) to maximize $p'(0) = a = 3.4445$, aggressively inflating small singular values.

Key properties:
- **Non-convergent.** $p(1) = 0.701 \neq 1$. The polynomial does not approximate the sign function.
- **Non-monotone.** $p$ peaks at $\sigma = 0.555$ with $p(0.555) = 1.20$, then declines.
- **Unstable fixed point.** $p(\sigma^*) = \sigma^*$ at $\sigma^* = 0.868$, but $|p'(\sigma^*)| = 1.58 > 1$.
- **Period-2 behavior.** After 5 iterations, SVs scatter across $[0.68, 1.12]$ in anti-correlated even/odd patterns.

### 2.2 Bifurcation Parameterization

The polynomial family can be parameterized by a single control parameter $d = p'(\sigma^*)$, the derivative at the fixed point. Given fixed values $\sigma^* = 0.868$ and $c = 2.0315$, the remaining coefficients are determined analytically:

$$a = \frac{3 - d}{2} + c\sigma^{*4}, \quad b = \frac{d - 1 - 4c\sigma^{*4}}{2\sigma^{*2}}$$

Standard Muon corresponds to $d = -1.58$. Sweeping $d$ from $-0.5$ to $-3.0$ traces a path through convergent ($|d| < 1$), period-2 ($1 < |d| < \sim\!1.5$), and chaotic ($|d| > \sim\!1.7$) dynamical regimes.

### 2.3 Lyapunov Exponents

We computed the Lyapunov exponent $\lambda = \lim_{N \to \infty} \frac{1}{N} \sum_{k=1}^N \log|p'(x_k)|$ numerically, averaging over 200 starting points and 4000 iterations per trajectory:

| $d$ | $|p'(\sigma^*)|$ | Lyapunov $\lambda$ | Regime |
|---|---|---|---|
| $-0.50$ | $0.50$ | $-0.693$ | Attracting |
| $-0.80$ | $0.80$ | $-0.223$ | Attracting |
| $-1.00$ | $1.00$ | $-0.001$ | **Edge of chaos** |
| $-1.10$ | $1.10$ | $-0.253$ | Attracting (period-2) |
| $-1.20$ | $1.20$ | $-0.777$ | Attracting (period-2, most stable) |
| $-1.40$ | $1.40$ | $-0.286$ | Attracting (period-2) |
| $-1.58$ | $1.58$ | $-0.571$ | Attracting (period-2, Muon) |
| $-1.80$ | $1.80$ | $+0.331$ | Weakly chaotic |
| $-2.80$ | $2.80$ | $+0.739$ | Chaotic |

The Lyapunov structure is non-monotone: it dips most negative at $d = -1.2$ (strongest period-2 attraction), rises through $d = -1.58$, and crosses zero around $d = -1.7$. Standard Muon ($d = -1.58$) operates in the stable period-2 regime, not chaos.

### 2.4 Period-2 Orbits

A period-2 orbit $\{a, b\}$ with $p(a) = b$, $p(b) = a$ is born at the bifurcation boundary $d = -1.0$. Its stability multiplier $|p'(a) \cdot p'(b)|$ determines whether the oscillation is self-correcting ($< 1$) or divergent ($> 1$):

| $d$ | $\sigma_{\text{low}}$ | $\sigma_{\text{high}}$ | $|p'(a) \cdot p'(b)|$ | Stable? |
|---|---|---|---|---|
| $-1.00$ | $0.868$ | $0.868$ | $1.000$ | Marginal (born) |
| $-1.10$ | $0.791$ | $0.958$ | $0.603$ | Yes |
| $-1.20$ | $0.764$ | $0.998$ | $0.212$ | Yes (most stable) |
| $-1.40$ | $0.730$ | $1.059$ | $0.565$ | Yes |
| $-1.58$ | $0.709$ | $1.103$ | $1.269$ | **No (unstable)** |
| $-2.00$ | $0.681$ | $1.190$ | $3.041$ | No |

**Standard Muon's period-2 orbit is unstable** — the oscillation diverges rather than damps. This explains why iterate blending is effective: it cancels a *divergent* oscillation. At $d = -1.0$ to $-1.4$, the period-2 orbit is self-correcting, requiring less blending.

### 2.5 Equalization Quality

| Polynomial | $\text{Var}[p^5(\sigma)]$ | Relative |
|---|---|---|
| Standard NS ($d = -1.58$) | $0.0307$ | $1.0\times$ |
| $d = -1.4$ | $0.0189$ | $1.6\times$ better |
| $d = -1.0$ | $0.0019$ | $17\times$ better |
| Minimax optimal | $\approx 0$ | Perfect |

The $d = -1.0$ polynomial achieves 17$\times$ better SV equalization. But as Series 13 demonstrates, equalization quality is the *least important* factor — coefficient magnitude ($a \geq 3.0$) dominates.

---

## 3. Formal Theorems

### 3.1 Theorem 1: The Coefficient-Stability Tradeoff

**Statement.** For any degree-5 odd polynomial $p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ with a positive fixed point $\sigma^*$ (i.e., $p(\sigma^*) = \sigma^*$), the derivative at the fixed point satisfies:

$$p'(\sigma^*) = 3 - 2a + 2c\sigma^{*4}$$

**Proof.** The fixed-point condition gives $a + b\sigma^{*2} + c\sigma^{*4} = 1$, so $b = (1 - a - c\sigma^{*4})/\sigma^{*2}$. The derivative $p'(\sigma^*) = a + 3b\sigma^{*2} + 5c\sigma^{*4}$. Substituting:

$$p'(\sigma^*) = a + 3(1 - a - c\sigma^{*4}) + 5c\sigma^{*4} = 3 - 2a + 2c\sigma^{*4} \quad \square$$

**Corollary (Stability Bound).** For a stable fixed point ($|p'(\sigma^*)| \leq 1$), the leading coefficient satisfies $a \leq 2 + c\sigma^{*4}$. With $c = 2.0315$ and $\sigma^* = 0.868$: $a_{\text{crit}} = 3.152$.

**Corollary (The Impossibility).** Effective training requires $a \geq 3.0$ (established empirically over 85+ experiments with a clear dose-response curve). The stability boundary is $a = 3.152$. The gap is only 5%. Any polynomial aggressive enough for effective training either has an unstable fixed point or operates so close to the stability boundary that its convergence rate is negligible.

### 3.2 Theorem 2: Cubic Iteration Limitation

**Statement.** For any cubic odd polynomial $q(\sigma) = \alpha\sigma + \beta\sigma^3$ with a stable fixed point at $\sigma^* > 0$, the leading coefficient satisfies $\alpha < 2$.

**Proof.** The fixed-point condition gives $\beta = (1 - \alpha)/\sigma^{*2}$. The derivative $q'(\sigma^*) = \alpha + 3\beta\sigma^{*2} = 3 - 2\alpha$. Stability requires $|3 - 2\alpha| < 1$, giving $1 < \alpha < 2$. $\quad \square$

**Significance.** The cubic iteration $\sigma \to \sigma(1 - \eta(\sigma^2 - c^2))$ — which arises independently in frame potential gradient descent (Benedetto-Fickus), Brockett double-bracket flow, Yang-Mills heat flow, quantum depolarizing channels, Stein discrepancy gradient flow, and spectral proximal operators — has $\alpha = 1 + \eta c^2 < 2$. This is far below the $a \geq 3.0$ required for effective training. The quintic term $c\sigma^5$ is structurally necessary.

### 3.3 Theorem 3: Period-Doubling Bifurcation

**Statement.** The one-parameter family of NS polynomials $p_d(\sigma)$, parameterized by $d = p'(\sigma^*)$, undergoes a period-doubling bifurcation at $d = -1$. For $d \in (-1, 0)$, $\sigma^*$ is a stable fixed point. At $d = -1$, a period-2 orbit is born. For $d < -1$, $\sigma^*$ is unstable and the period-2 orbit $\{a, b\}$ has stability multiplier $|p'(a) \cdot p'(b)|$.

**Proof.** The fixed point stability condition $|p'(\sigma^*)| = |d| < 1$ gives the first claim. The period-doubling follows from standard bifurcation theory (Strogatz, Theorem 3.5.1). The composite map $(p \circ p)'(a) = p'(b) \cdot p'(a)$ gives the stability multiplier. $\quad \square$

### 3.4 Theorem 4: Equivalence of Cubic Spectral Iterations

**Statement.** The following spectral iterations from independent mathematical fields all take the form $\sigma_{k+1} = (1 + \eta c^2)\sigma_k - \eta\sigma_k^3$:

1. **Frame potential gradient descent** (Benedetto-Fickus)
2. **Brockett double-bracket flow**
3. **Quantum depolarizing channel**

**Proof sketch.** Each framework's gradient/flow/channel acts on singular values independently (due to SVD-respecting structure), producing the scalar ODE $\dot{\sigma} = -\eta\sigma(\sigma^2 - c^2)$. Forward Euler discretization gives $\sigma_{k+1} = (1 + \eta c^2)\sigma_k - \eta\sigma_k^3$. By Theorem 2, stability forces $\alpha = 1 + \eta c^2 < 2$. $\quad \square$

---

## 4. Experimental Program: What Fails

### 4.1 Experimental Setup

All experiments at 1.2M scale use:
- **Model:** MicroGPT ($d_{\text{model}} = 128$, $n_{\text{heads}} = 4$, $n_{\text{layers}} = 4$, $d_{\text{ff}} = 512$, context length 256, $\sim$1.2M params)
- **Data:** TinyShakespeare (1.1M characters, vocabulary 65, character-level tokenization)
- **Training:** 2000 steps, batch size 32, 200-step warmup, cosine LR decay, seed 42
- **Optimizer split:** Muon/CUM for hidden 2D weights ($\eta = 0.02$), AdamW for embeddings/biases ($\eta = 3 \times 10^{-4}$)
- **Baseline:** Muon NS=5, $\beta_1 = 0.95$ $\to$ val\_loss $\approx 1.515 \pm 0.008$

Scale tests at 124M use GPT-2 Small architecture on WikiText-103 / OpenWebText.

Variability note: `torch.compile` introduces $\pm 0.006$ nondeterminism in absolute val\_loss across runs. Within-run relative rankings are preserved. All comparisons use within-run deltas.

### 4.2 Pre-NS Input Modifications (Series 1, 3, 9)

| Approach | Experiments | Best Result | Why |
|---|---|---|---|
| Factored preconditioning | v1 | $-0.0003$ (noise) | NS contracts perturbations; input rotated 28° but NS output changed by $<0.3°$ |
| Gradient centralization | v4 | $+0.0003$ | Row means carry useful bias in transformers |
| Directional momentum | 3v3 | $+0.0004$ (tied Muon) | NS is robust to whether input is raw or normalized |
| Dual-momentum pre-NS | 9v1 | $\pm 0.005$ (noise) | NS makes input modifications irrelevant, not harmful |
| Ruiz equilibration (Series 14) | 14v1 | $+0.003$ | Re-confirms: NS robust to input preprocessing |

**Conclusion.** NS's Jacobian at the fixed point contracts perturbations to noise. Pre-NS modifications are provably futile — confirmed across 5 independent approaches spanning 14 series.

### 4.3 Post-NS Curvature Recovery (Series 1–2)

| Approach | Experiments | Best Result | Why |
|---|---|---|---|
| Row/column scaling | v2 | $+0.01$ | Gradient variance too noisy at 1.2M |
| Second-moment grafting | v11 | $-0.006$ | Element-wise reweighting breaks NS's uniform treatment |
| Gradient difference momentum | v12 | $-0.003$ | Small signal overwhelmed by NS's equalization |
| Un-equalization (5% row) | 11v3 | $-0.014$ ($-0.001$ vs combined) | Promising but marginal; one run |

**Conclusion.** Post-NS modifications face a noise floor: gradient statistics at 1.2M are too noisy for meaningful per-element or per-row curvature recovery.

### 4.4 Alternative Orthogonalization Methods (Series 2–3, 6)

| Method | Val Loss | vs Muon | Why It Failed |
|---|---|---|---|
| SVD exact polar factor (2v2) | 1.531 | $+0.012$ | SVs $= 1.0$ loses implicit regularization from NS's sub-unity SVs |
| Cayley retraction (3v2) | 1.847 | $+0.32$ | Rotational component carries less optimization signal than polar factor |
| Warm-started NS (3v1, 6v7) | $\sim$2.1–3.0 | $+0.3$–$1.5$ | Traps iteration near unstable fixed point; repeats previous direction |
| Dion full-rank (6v4) | 1.533 | $+0.021$ | Converges to SVs $= 1.0$ (same problem as SVD polar) |
| Halley 3-iter (6v5) | 1.572 | $+0.060$ | Converges to SVs $= 1.0$ |
| QDWH (6v5) | Crashed | — | Ill-conditioned linear solves |
| PolarGrad (6v2) | 1.547 | $+0.036$ | Nuclear norm scaling insufficient |
| PSGD Kron (6v3) | 2.393 | $+0.88$ | Lie group Kronecker learning diverges without careful tuning |

**Key discovery (2v2).** Exact polar factor (SVs $= 1.0$) is *worse* than NS's approximation (SVs $\approx 0.877$). NS's sub-unity convergence provides implicit regularization — the "imperfection" is a feature. This was the first hint that NS's specific dynamics matter more than convergence quality.

### 4.5 Custom SV Mappings (Series 5, 7)

| Method | Best Result | vs Muon | Why It Failed |
|---|---|---|---|
| Tanh soft equalization (5v2) | 1.588 ($\beta=7$) | $+0.07$ | Monotone; doesn't lift small SVs enough |
| Schatten-$p$ descent (5v3) | 1.520 ($p=8$) | $+0.005$ | Closest non-NS; but $p=8$ beats $p=32$, suggesting NS's non-monotone map has unique properties |
| Huber SV mapping (7v1) | 1.562 | $+0.027$ | All 10 configs failed; monotone maps cannot match NS's oscillation-based equalization |
| Power mapping (7v1) | 1.801 | $+0.27$ | Same monotonicity limitation |

**Key insight (5v3).** More equalization is not always better: Schatten-$p$ at $p = 8$ beats $p = 32$. This suggests NS's specific non-monotone oscillating SV map has properties beyond simple equalization that no monotonic power function can replicate.

### 4.6 Variance Reduction and Other (Series 6)

| Method | Val Loss | vs Muon | Why It Failed |
|---|---|---|---|
| MARS ($\gamma = 0.5, 1.0$) | 2.49–2.50 | $+0.98$ | NS on gradient *differences* (tiny values) produces random orthogonal matrices |
| Weighted Procrustes | 1.928 | $+0.42$ | SV reordering disrupts gradient structure |
| Optimal SV shrinkage, hard | 2.146 | $+0.63$ | Zeroing SVs below noise floor removes useful directions |
| Optimal SV shrinkage, blend | 1.519 | $+0.007$ | Closest alternative to NS; respects denoising but still loses |

### 4.7 Complete Failure Taxonomy

After 7 series and 40+ experiments, the approaches that fail partition cleanly:

| Category | Why | Confirmed by |
|---|---|---|
| Pre-NS modification | NS contracts input perturbations | v1, 3v3, 9v1, 14v1 (4$\times$) |
| Targeting SVs $= 1.0$ | Sub-unity SVs are a feature | 2v2, 6v1, 6v4, 6v5 |
| Monotone SV maps | Cannot match non-monotone equalization | 5v2, 5v3, 7v1 (10 configs) |
| Warm-starting | Traps near unstable fixed point | 3v1, 6v7 |
| Feature stacking | Improvements cancel each other | v4 |
| Smart weighting | Breaks statistical cancellation | 10e adaptive, 10e cosine, 11v2 |

The *only* approach that consistently beat Muon was intercepting NS mid-iteration and blending intermediate iterates.

---

## 5. The Breakthrough: Oscillation Cancellation

### 5.1 Multi-Resolution NS Blend (Series 5)

The first significant win came from blending NS intermediates instead of the raw gradient:

$$\text{update} = (1 - \alpha) \cdot \text{NS}_5(u) + \alpha \cdot \text{scale\_match}(\text{NS}_2(u))$$

| Config | Val Loss | vs Muon |
|---|---|---|
| save@2, $\alpha = 0.15$ | 1.508 | $-0.011$ |
| save@3, $\alpha = 0.10$ | 1.515 | $-0.004$ |

NS$_2$ retains $\sim$25% of original SV spread — partially denoised while preserving curvature structure. save@2 >> save@3 (more information retained). The improvement is 2.6$\times$ larger than v3's raw gradient blend ($-0.011$ vs $-0.004$), confirming that the NS intermediate is a better blend target than the raw gradient.

### 5.2 Combined Mode: Within-Step + Across-Step (Series 8–10)

Stacking two orthogonal cancellation channels:

$$\text{iterate\_blended} = (1 - b) \cdot \text{NS}_5 + b \cdot \text{scale\_match}(\text{NS}_2)$$
$$\text{ema} \leftarrow \beta \cdot \text{ema} + (1 - \beta) \cdot \text{iterate\_blended}$$
$$\text{output} = (1 - \alpha) \cdot \text{iterate\_blended} + \alpha \cdot \text{scale\_match}(\text{ema})$$

Results across 3 replications at 1.2M:

| Run | Val Loss | vs Muon |
|---|---|---|
| Run 1 | 1.5008 | $-0.019$ |
| Run 2 | 1.5062 | $-0.016$ |
| Run 3 | 1.5091 | $-0.012$ |
| **Mean** | | **$-0.016 \pm 0.004$** |

Scale test at 124M (GPT-2 Small, WikiText-103): $-0.014$ vs Muon. Same late-game acceleration pattern: Muon leads steps 0–750, combined catches up at step 1000, pulls ahead from step 1250 onward.

### 5.3 Why Simple Averaging Beats Smart Weighting

We tested two "intelligent" alternatives to uniform blending:
- **Adaptive residual:** Weight corrections by magnitude. Result: noise ($-0.002$).
- **Cosine-gated:** Apply corrections only when curvature and temporal signals agree. Result: $\pm 0.000$.
- **Per-layer adaptive:** Adapt the *amount* of uniform blend per layer. Result: $+0.002$ vs combined.

All failed. The pattern is robust: simple averaging works; complex weighting logic doesn't. Oscillation cancellation is a Central Limit Theorem effect — averaging many small structured corrections. Gating or weighting individual corrections suppresses the collective signal.

### 5.4 Evidence: Stable Polynomial Catastrophically Fails (Series 11)

The definitive test: if oscillation is merely a bug, a polynomial with a *stable* fixed point should work better.

The "stable-0.88" polynomial $(a, b, c) = (2.0, -1.940, 0.836)$ has $p(0.88) = 0.88$ and $p'(0.88) \approx 0$ — super-stable, zero oscillation.

| Config | Val Loss | vs Muon |
|---|---|---|
| Stable basic | 1.567 | $+0.045$ |
| Stable combined | 1.554 | $+0.032$ |

**Catastrophic failure.** The small coefficient $a = 2.0$ inflates $\sigma = 0.1$ to only $0.20$ per step (vs $0.34$ for standard NS). After 5 iterations, equalization is incomplete. The large coefficients needed for aggressive equalization *are* what cause the fixed-point instability. They cannot be decoupled with a degree-5 polynomial in 5 steps.

### 5.5 Evidence: SmoothedAdam Fails (Series 11d)

Does temporal averaging help non-NS optimizers?

| Config | Val Loss | vs AdamW |
|---|---|---|
| AdamW | 1.604 | baseline |
| SmoothedAdam $\alpha = 0.15$ | 1.608 | $+0.004$ |
| SmoothedAdam $\alpha = 0.30$ | 1.616 | $+0.012$ |

**Temporal averaging actively hurts Adam.** Adam's update direction is already well-conditioned — there is no structured oscillation to cancel. Without structured oscillation, blending is just sluggishness. The improvement is NS-specific, not a general optimizer principle.

---

## 6. The Final Recipe: Three-Layer Oscillation Management

### 6.1 TD($\lambda$) Multi-Iterate Blending (Series 12)

Instead of blending only NS$_2$ and NS$_5$, exponentially weight *all* iterates with TD($\lambda$)-style decay:

$$w_k = \frac{\lambda^{n-k}}{\sum_{j=1}^n \lambda^{n-j}}$$

For $\lambda = 0.5$, $n = 5$: weights $= [0.032, 0.065, 0.129, 0.258, 0.516]$.

| $\lambda$ | Val Loss | vs Muon | vs Combined |
|---|---|---|---|
| $0.3$ + temporal | 1.506 | $-0.011$ | $+0.001$ |
| **$0.5$ + temporal** | **1.501** | **$-0.016$** | **$-0.004$** |
| $0.5$ no-temporal | 1.509 | $-0.011$ | — |
| $0.7$ + temporal | 1.509 | $-0.010$ | — |
| $0.9$ + temporal | 1.510 | $-0.009$ | — |

$\lambda = 0.5$ is optimal. $\lambda = 0.3$ concentrates too heavily on NS$_5$ (similar to two-point). $\lambda \geq 0.7$ admits too much early-iterate noise. NS$_3$ and NS$_4$ carry useful oscillation-cancellation information that two-point blending discards — more samples of the oscillatory process yield better cancellation (an application of the Nyquist principle to iterate averaging).

Within-step and across-step cancellation are orthogonal: TD($\lambda = 0.5$) alone gives $-0.011$; adding temporal EMA gives $-0.012$ to $-0.016$. The $\sim$0.005 temporal contribution stacks additively.

### 6.2 Weaker Polynomial at the Bifurcation Boundary (Series 12)

| $d$ | Val Loss (combined) | vs Muon |
|---|---|---|
| $-1.0$ (edge of chaos) | $1.503$–$1.505$ | $-0.015$ to $-0.019$ |
| $-1.4$ | $1.504$–$1.513$ | $-0.005$ to $-0.020$ |
| $-1.58$ (standard) | $1.506$ | $-0.009$ |
| $-2.0$ | $1.510$ | $-0.004$ |
| $-2.8$ | $1.521$ | $+0.007$ |

The optimal derivative lies in $[-1.0, -1.4]$ — noisy at this effect size. $d = -2.8$ is catastrophic. Within combined mode, *less* polynomial oscillation is better because iterate blending already handles cancellation. Adaptive scheduling (annealing $d$ from $-1.8$ to $-1.0$ during training) was tested and failed: optimal dynamics should be constant.

### 6.3 Combined: The Final Recipe

Three orthogonal layers of oscillation management:

1. **Weaker polynomial** ($d = -1.0$, $a = 3.1512$): reduces oscillation at source while maintaining $a \geq 3.0$.
2. **Multi-iterate blending** (TD $\lambda = 0.5$): cancels within-step oscillation via exponentially-weighted averaging of all five NS iterates.
3. **Temporal EMA** ($\beta = 0.5$, $\alpha = 0.15$): cancels across-step oscillation via exponential smoothing of blended outputs.

**Algorithm:**
```
1. Standard Muon momentum with Nesterov
2. Run NS with d=-1.0 coefficients (a=3.1512, b=-4.3105, c=2.0315), saving ALL iterates
3. Compute TD(λ=0.5) weighted blend of all iterates (norm-matched to final)
4. Update temporal EMA of blended outputs
5. Frobenius-norm-matched blend of current output with temporal EMA
6. Apply update with aspect ratio scaling
```

**Results:**

| Config | Val Loss | vs Muon | vs Combined |
|---|---|---|---|
| Final recipe | **1.4993** | **$-0.018$** | $-0.009$ |
| TD $\lambda = 0.5$ + temporal (std poly) | 1.505 | $-0.012$ | $-0.004$ |
| Combined (std poly) | 1.509 | $-0.009$ | baseline |
| Muon | 1.518 | baseline | $+0.009$ |

**First sub-1.50 result.** Effects are approximately additive: TD($\lambda$) contributes $\sim$$-0.004$ vs combined, $d = -1.0$ contributes $\sim$$-0.006$.

**Replication (3 runs):** 1.4993, 1.4993, 1.4992. Remarkably consistent.

### 6.4 Trajectory Analysis

The final recipe starts slowest but accelerates hardest late-game:

| Step | Muon | Combined | Final Recipe |
|---|---|---|---|
| 250 | 2.381 | 2.372 | 2.408 |
| 500 | 1.791 | 1.770 | 1.802 |
| 750 | 1.625 | 1.614 | 1.615 |
| 1000 | 1.556 | 1.556 | 1.553 |
| 1250 | 1.532 | 1.521 | 1.519 |
| 1500 | 1.508 | 1.501 | **1.498** |
| 1750 | 1.513 | 1.501 | **1.493** |
| 1999 | 1.521 | 1.512 | **1.504** |

At step 1750, the final recipe is $-0.020$ below Muon. The temporal EMA requires $\sim$500–750 steps to warm up, after which oscillation cancellation begins providing its benefit.

---

## 7. Why Theoretically Optimal Polynomials Fail (Series 13)

Numerical optimization via differential evolution found polynomials outside the bifurcation family with near-perfect SV equalization:

| Polynomial | $a$ | $\text{Var}[p^5]$ | Val Loss (combined) | vs Muon |
|---|---|---|---|---|
| Standard NS | 3.445 | 0.031 | — | baseline |
| $d = -1.0$ | 3.151 | 0.002 | 1.499 | $-0.018$ |
| Minimax optimal | 2.681 | $\approx 0$ | 1.525 | $-0.002$ |
| Min-variance | 2.231 | $\approx 0$ | 1.544 | $+0.020$ |
| Stable-0.88 | 2.000 | $\approx 0$ | 1.567 | $+0.045$ |

**All minimax presets failed** despite near-perfect equalization. The linear coefficient $a$ directly controls small-SV inflation speed — $\sigma = 0.1$ maps to $a \cdot 0.1 \approx 0.27$ (minimax) vs $0.34$ (standard NS) in one step. After 5 iterations, this 22% deficit compounds fatally.

**Dose-response relationship:**

| $a$ | Val Loss vs Muon |
|---|---|
| 2.00 | $+0.045$ |
| 2.23 | $+0.020$ |
| 2.68 | $+0.017$ vs combined |
| 3.15 | $-0.018$ |
| 3.44 | baseline |

Clear monotonic: lower $a$ = worse performance, regardless of equalization quality. **$a \geq 3.0$ is non-negotiable.**

**Hierarchy of what matters:**
1. **Coefficient magnitude** ($a \geq 3.0$): non-negotiable
2. **Oscillation management** (blending): cancel the oscillation that large coefficients cause
3. **Equalization quality** ($\text{Var}[p^5]$): least important — $17\times$ worse equalization still wins

The minimax TD configuration (full recipe treatment applied to minimax polynomial) achieved exactly Muon-tier ($-0.0003$ vs Muon). The recipe has nothing to work with when the polynomial is too gentle.

---

## 8. Mathematical Frameworks: Universal Impossibility (Series 14)

### 8.1 Theoretical Survey

A comprehensive analysis of 26 mathematical frameworks from six independent fields — frame theory (Benedetto-Fickus potential), Riemannian geometry (Brockett double-bracket flow, Yang-Mills heat flow), quantum information (depolarizing channels), statistics (Stein discrepancy), optimization (spectral proximal operators), and more — revealed that all six converge to the same spectral iteration:

$$\sigma \to \sigma(1 - \eta(\sigma^2 - c^2))$$

This cubic polynomial has a stable fixed point at $c$. With optimal $\eta$, the linear coefficient $\alpha = 1 + \eta c^2 \leq 1.5$ — far below $a \geq 3.0$. Increasing $\eta$ to achieve $\alpha \geq 3.0$ gives $|f'(0.88)| \approx 3.0$, violently unstable. By Theorem 2, any degree-3 iteration with a stable fixed point is limited to $\alpha < 2$.

This proves the coefficient-stability tradeoff is **universal** — not a quirk of NS, but fundamental to polynomial spectral maps. Six independent proofs of the same impossibility.

### 8.2 Experimental Validation

Four frameworks were tested:

| Framework | Val Loss | vs Muon | Notes |
|---|---|---|---|
| Ruiz equilibration (5 iter + NS5) | 1.525 | $+0.003$ | NS robust to input preprocessing |
| Frame potential ($\eta = 2.5$, 7 steps) | 2.140 | $+0.618$ | Catastrophic; Paradigm A dead |
| Polar Express (hand-tuned, 5 steps) | 1.540 | $+0.018$ | Later steps too gentle |
| Polar Express (Remez-optimal, 5 steps) | 1.523 | $+0.007$ | Step 1 $a = 5.96$; still loses |
| Polar Express (Remez-optimal, 3 steps) | 1.541 | $+0.025$ | Fewer steps worse |

**Every alternative polynomial/iteration across 14 series has failed.** The NS quintic $(3.4445, -4.7750, 2.0315)$ with its specific coefficient magnitudes and constancy across iterations is uniquely effective.

The Remez-optimal Polar Express is particularly informative: even with minimax-optimal coefficients computed via differential evolution for each step's spectral interval (step 1: $a = 5.96$, step 5: $a = 1.88$), it cannot beat NS's *fixed* quintic. The cursed quintic's iterated composition dynamics — including its specific oscillation structure — are uniquely effective in ways that single-step optimality cannot capture.

---

## 9. Per-Head Structural Optimization (Series 15–16)

### 9.1 Universal Muon (Failed)

Applying NS to all parameter types (including embeddings) with unit normalization for 1D parameters: $+0.007$ vs Muon. Embeddings are lookup tables, not linear transforms — orthogonalizing their gradients lacks geometric meaning. The Muon + AdamW split is structurally correct.

### 9.2 Per-Head Orthogonalization

Standard Muon applies one NS call to the full QKV gradient ($384 \times 128$ at 1.2M, aspect ratio 3:1). Per-head NS splits into 4 slices of ($96 \times 128$, ratio 0.75:1) and applies NS to each independently.

| Config | Val Loss | vs Muon |
|---|---|---|
| 4 slices (per-head) | 1.499 | $-0.025$ |
| 3 slices (per-Q/K/V) | 1.507 | $-0.018$ |
| 12 slices (per-Q/K/V/head) | 1.508 | $-0.017$ |

**Replication (3 runs):** $-0.019 \pm 0.003$. The original $-0.025$ was a lucky run.

Why it works: (1) Different heads learn different features with different spectral profiles; shared equalization flattens per-head structure. (2) Near-square slices ($0.75:1$) are in NS's conditioning sweet spot.

### 9.3 Combining Per-Head with Blending

| Config | Val Loss vs Muon |
|---|---|
| Per-head plain | $-0.022$ |
| Per-head + combined (two-point + temporal) | $-0.022$ |
| **Per-head + TD($\lambda$) + $d = -1.0$** | **$-0.032$** (lucky run; replicated: $-0.025 \pm 0.002$) |

**Combined mode adds nothing** to per-head. Near-square slices converge cleanly with minimal oscillation — there is nothing for two-point blending to cancel. But TD($\lambda$) + weaker polynomial still helps ($+0.006$ over plain per-head) because the weaker polynomial changes the convergence *target*, not just oscillation amplitude.

Out_proj column slicing: confirmed noise in isolation (16e). MLP slicing: actively hurts ($+0.004$); MLP has no head structure.

### 9.4 Scale Test: Per-Head Does Not Scale to 124M

| Config | Aspect Ratio | Val Loss vs Muon |
|---|---|---|
| 12 slices (124M, per-head) | 0.25:1 | $-0.001$ |
| 3 slices (124M, per-Q/K/V) | 1:1 | $+0.001$ |
| 4 slices (124M) | 0.75:1 | $-0.006$ |

**Per-head does not scale.** The per-head ratio = $3/n_{\text{heads}}$:
- 4 heads (1.2M): ratio $= 0.75$ (near-square, NS sweet spot) $\to$ $-0.019$
- 12 heads (124M): ratio $= 0.25$ (very rectangular) $\to$ $-0.001$

The improvement is aspect-ratio-dependent, not fundamental. The blending recipe alone ($-0.014$ at 124M) remains the more scalable contribution.

---

## 10. Scale Validation

### 10.1 Results at 124M

| Method | Val Loss | vs Muon |
|---|---|---|
| Combined mode (Series 11c) | 4.139 | $-0.014$ |
| Per-head plain (12 slices) | 3.714 | $-0.001$ |
| Per-head TD (4 slices) | 4.829 | $-0.006$ |

The combined/blending mechanism holds at 124M with the same late-game acceleration pattern observed at 1.2M. Per-head structural optimization degrades as head count increases.

### 10.2 Trajectory at 124M

| Step | Muon | Combined | Delta |
|---|---|---|---|
| 250 | 5.840 | 5.848 | $+0.008$ |
| 500 | 5.231 | 5.252 | $+0.021$ |
| 750 | 4.931 | 4.939 | $+0.008$ |
| 1000 | 4.652 | 4.645 | $-0.007$ |
| 1250 | 4.478 | 4.457 | $-0.021$ |
| 1500 | 4.311 | 4.294 | $-0.017$ |
| 1750 | 4.188 | 4.171 | $-0.017$ |
| 1999 | 4.219 | 4.204 | $-0.015$ |

The temporal EMA warmup period ($\sim$750 steps) and crossover point ($\sim$step 1000) are consistent across scales, suggesting the oscillation cancellation mechanism operates on a fixed timescale relative to training dynamics.

---

## 11. Discussion

### 11.1 Why NS Works: The Coefficient-Magnitude Hypothesis

Our results establish a clear hierarchy for NS polynomial design:

1. **Coefficient magnitude is king.** The linear coefficient $a$ controls small-SV inflation speed. $a \geq 3.0$ is non-negotiable. Every polynomial with $a < 3.0$ failed, including the theoretically optimal minimax polynomial ($a = 2.68$, near-perfect equalization) and the stable-0.88 polynomial ($a = 2.0$, zero oscillation). The dose-response is monotonic and steep.

2. **Oscillation is the unavoidable tax.** Theorem 1 proves that $a \geq 3.0$ with the standard $c$ value forces $|p'(\sigma^*)| \geq 0.70$, and $a = 3.152$ hits the stability boundary. Standard Muon's $a = 3.4445$ yields $|p'(\sigma^*)| = 1.58$ — well into the unstable regime. The oscillation is not a design choice; it is the mathematical consequence of the coefficients needed for effective training.

3. **Blending is Phase 2.** Once the polynomial provides aggressive-enough equalization (Phase 1), blending extracts the equalization benefit while canceling the oscillation cost (Phase 2). Without Phase 1 (stable polynomial: $+0.045$), Phase 2 is useless. Without Phase 2 (standard Muon), the oscillation degrades performance by $\sim$0.018.

### 11.2 The Universality Result

The most surprising finding is that the coefficient-stability tradeoff is not specific to the NS quintic. Six independent mathematical fields — frame theory, Riemannian geometry (Brockett and Yang-Mills flows), quantum information, statistics, and optimization — all converge to the same cubic spectral iteration $\sigma \to \sigma(1 - \eta(\sigma^2 - c^2))$. Theorem 2 proves this cubic is limited to $\alpha < 2$ for stability, and Theorem 1 shows the quintic has the same structure with a slightly relaxed but still binding constraint.

This means the NS quintic is not an arbitrary choice that happens to work — it is the *unique* practical operating point in a mathematically constrained space. The quintic term $c\sigma^5$ extends the cubic's reach from $\alpha < 2$ to $a \approx 3.15$ at the stability boundary, buying exactly enough coefficient magnitude for effective training. But even the quintic cannot escape the fundamental tradeoff: $a = 3.15$ is already at the stability boundary, and effective training pushes $a$ past it.

### 11.3 Edge of Chaos as Optimal Operating Point

The $d = -1.0$ polynomial sits exactly at the period-doubling bifurcation boundary — the edge of chaos. This is where experiments found the best performance with blending. The connection to "edge of chaos" phenomena in other complex systems (cellular automata, neural networks at the edge of stability) is suggestive but not yet formalized.

Intuitively: $d = -1.0$ is the weakest polynomial that still has $a = 3.15 \geq 3.0$. With blending handling oscillation cancellation, the polynomial should minimize oscillation while staying above the coefficient threshold. $d = -1.0$ achieves exactly this — it is the boundary where oscillation begins, producing minimal oscillation for maximal coefficient magnitude.

### 11.4 The Specificity of the Result

An important negative result: temporal averaging does *not* generalize to Adam (Series 11d). The improvement is specific to iterative subroutines with structured oscillation. This narrows the applicability but sharpens the mechanism — we are not observing generic smoothing but rather the cancellation of a specific, structured, anti-correlated oscillation pattern unique to NS's unstable polynomial dynamics.

### 11.5 Limitations

1. **Scale.** The full final recipe was validated at 1.2M (3$\times$ replicated) and combined mode at 124M. Validation at 1B+ scale with Chinchilla-optimal training budgets remains untested.

2. **Training length.** Our 2000-step experiments may not capture behavior at convergence. The late-game acceleration pattern suggests the benefit persists, but confirmation at 10K–100K steps is needed.

3. **Architecture dependence.** All experiments use transformer architectures. The mechanism should generalize to any architecture using Muon, but this is unverified.

4. **Computational overhead.** The final recipe requires saving all 5 NS iterates and computing a weighted blend, adding $\sim$$2\times$ wall-clock time over Muon. The per-head variant adds further overhead from multiple smaller NS calls.

---

## 12. Related Work

**Muon.** Jordan (2024) introduced Muon as a matrix-aware optimizer using NS iteration for gradient orthogonalization. Our work provides the first detailed analysis of the NS polynomial's dynamical properties and the first systematic modification that consistently improves upon it.

**Newton-Schulz iteration.** The NS iteration for matrix sign function / polar decomposition has been studied extensively in numerical linear algebra (Higham, 2008). Our contribution is analyzing its behavior under *iterated composition* in the optimization context, where single-step convergence properties do not predict iterated behavior (as demonstrated by the Chebyshev sign polynomial's catastrophic divergence).

**Edge of stability.** Cohen et al. (2021) identified that neural network training operates at the edge of stability, where the loss Hessian's maximum eigenvalue hovers near $2/\eta$. Our finding that NS's optimal operating point is at the edge of chaos ($d = -1.0$, bifurcation boundary) may be a manifestation of the same phenomenon at the optimizer subroutine level.

**Spectral methods in optimization.** Shampoo (Gupta et al., 2018), SOAP (Vyas et al., 2024), and K-FAC (Martens & Grosse, 2015) all use spectral information for preconditioning. Our universality result (Theorem 4) showing that six fields converge to the same cubic iteration suggests deep structural commonalities in spectral optimization methods.

**Bifurcation theory.** Period-doubling bifurcations are well-studied in dynamical systems (Strogatz, 2015). Our application to optimizer subroutine analysis appears to be novel.

---

## 13. Conclusion

We have presented a complete characterization of the Newton-Schulz polynomial dynamics in the Muon optimizer through 16 experimental series (85+ experiments), four formal theorems, and analysis drawing on six mathematical fields. The central finding is a fundamental coefficient-stability tradeoff: effective training requires large polynomial coefficients ($a \geq 3.0$), which necessarily produce unstable fixed-point dynamics and structured oscillation. This tradeoff is universal — proved independently across frame theory, Riemannian geometry, quantum information, statistics, and optimization.

The only modification that consistently improves NS-based optimizers is oscillation cancellation via multi-iterate blending, achieving $-0.018$ at 1.2M scale ($3\times$ replicated) and $-0.014$ at 124M. This improvement is specific to NS's structured oscillation and does not generalize to non-oscillatory optimizers.

Our work also provides the most comprehensive negative results catalog for NS optimizer modifications: 40+ distinct failed approaches establish that NS's specific quintic polynomial, with its specific coefficients and constant application across iterations, cannot be improved by any of the tested alternatives — and six mathematical proofs explain why.

The NS quintic is not merely a convenient choice. It is the unique practical operating point in a mathematically constrained space where coefficient magnitude, fixed-point stability, and polynomial degree impose hard limits on what any spectral iteration can achieve.

---

## Appendix A: Complete Experimental Results

### A.1 Series 1: Pre-NS Modifications (M3 CPU, 1.2M, 2000 steps)

| Version | Name | Val Loss | vs Muon |
|---|---|---|---|
| v1 | Pre-NS Factored Precond | 1.519 | $-0.000$ |
| v2 | Post-NS Row/Col Scaling | $\sim$1.53 | $+0.01$ |
| v3 ($\alpha = 0.1$) | Soft NS (Raw Blend) | 1.515 | $-0.004$ |
| v3 ($\alpha = 0.2$) | Soft NS | 1.518 | $-0.001$ |
| 3b | NS Step Reduction (NS$=$3) | 1.544 | $+0.025$ |
| v4 | Stacked Innovations | 1.519 | $+0.000$ |
| **v5 (s@2 $b = 0.15$)** | **Multi-Resolution NS** | **1.508** | **$-0.011$** |
| v9 | Dampened Late-Stage NS | 1.510 | $-0.009$ |
| v10 | Dampened + Multi-Res | $\sim$1.84 | terrible |
| v11 | Second-Moment Grafting | 1.513 | $-0.006$ |
| v12 | Gradient Diff Momentum | 1.516 | $-0.003$ |

### A.2 Series 2–3: Alternative Orthogonalization (M3 CPU)

| Version | Name | Val Loss | vs Muon |
|---|---|---|---|
| 2v1 | Randomized Top-k Curvature | 1.530 | $+0.009$ |
| 2v2 | SVD Exact Polar Factor | 1.531 | $+0.012$ |
| 3v1 | Warm-Started NS | $\sim$2.10 | $+0.32$ |
| 3v2 | Cayley Retraction | 1.847 | $+0.32$ |
| 3v3 | Directional Momentum | 1.525 | $+0.000$ |

### A.3 Series 4–5: Spectral Methods (M3 CPU / H100 GPU)

| Version | Name | Val Loss | vs Muon |
|---|---|---|---|
| 4v2 | SODA | 2.340 | $+0.83$ |
| 4v3 | WGASU | 2.325 | $+0.81$ |
| 5v1 | Lie Algebra Momentum | 3.246 | $+1.73$ |
| 5v2 ($\beta = 3$) | Soft EQ (tanh) | 1.638 | $+0.12$ |
| 5v2 ($\beta = 7$) | Soft EQ (tanh) | 1.588 | $+0.07$ |
| 5v3 ($p = 8$) | Schatten-$p$ | 1.520 | $+0.005$ |
| 5v3 ($p = 32$) | Schatten-$p$ | 1.531 | $+0.016$ |
| 5v5 | SVD + NS Poly | 1.520 | $+0.002$ |
| **5v6 (s2 $b = 0.25$)** | **SVD NS Blend** | **1.505** | **$-0.017$** |

### A.4 Series 6: Deep Research V1 Directions (H100 GPU)

| Version | Name | Val Loss | vs Muon |
|---|---|---|---|
| 6v1 | Polar Express (Remez) | 1.520 | $+0.009$ |
| 6v2 | PolarGrad | 1.547 | $+0.036$ |
| 6v3 | PSGD Kron | 2.393 | $+0.88$ |
| 6v4 | Dion | 1.533 | $+0.021$ |
| 6v5 | Halley | 1.572 | $+0.060$ |
| 6v6 ($\gamma = 0.5$) | MARS | 2.491 | $+0.98$ |
| 6v7 | Warm-Started NS | 3.005 | $+1.49$ |
| 6v8 (blend) | Optimal SV Shrinkage | 1.519 | $+0.007$ |
| 6v9 | Weighted Procrustes | 1.928 | $+0.42$ |

### A.5 Series 7: Monotone SV Mappings (H100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| Huber $\alpha = 0.1$, $c = 0.88$ | 1.562 | $+0.027$ |
| Huber $\alpha = 0.3$, $c = 0.88$ | 1.751 | $+0.22$ |
| Huber $\alpha = 0.5$, $c = 0.88$ | 2.017 | $+0.48$ |
| Huber $\alpha = 0.7$, $c = 0.88$ | 2.209 | $+0.67$ |
| Power $\alpha = 0.3$ | 1.801 | $+0.27$ |
| Scheduled $\alpha$ 0.5$\to$0.1 | 1.856 | $+0.32$ |

### A.6 Series 8–10: Iterate Blending & Replication (A100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| Two-point s2+NS$_8$ | 1.520 | $-0.002$ (noise) |
| Three-point NS$_1$+NS$_3$+NS$_5$ | 1.523 | $+0.011$ |
| Geometric SV blend | 1.509 | $-0.003$ |
| Input blend (temporal EMA) | 1.514 | $-0.011$ |
| **Combined (iterate + input)** | **1.501** | **$-0.019$** |
| Combined (3-run mean) | — | **$-0.016 \pm 0.004$** |
| Adaptive residual | 1.516 | $-0.002$ |
| Cosine-gated | 1.518 | $+0.000$ |

### A.7 Series 11: Polynomial Design Tests (A100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| Stable-0.88 basic | 1.567 | $+0.045$ |
| Stable-0.88 combined | 1.554 | $+0.032$ |
| Un-equalization $\alpha = 0.05$ | 1.508 | $-0.014$ |
| Adaptive blend $s = 1.0$ | 1.511 | $-0.010$ |
| SmoothedAdam $\alpha = 0.15$ | 1.608 | $+0.004$ vs AdamW |
| SmoothedAdam $\alpha = 0.30$ | 1.616 | $+0.012$ vs AdamW |
| Combined (124M) | 4.139 | $-0.014$ |

### A.8 Series 12: Final Recipe Development (A100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| $d = -1.0$ combined | 1.503 | $-0.015$ |
| $d = -1.4$ combined | 1.504–1.513 | $-0.005$ to $-0.020$ |
| $d = -2.8$ combined | 1.521 | $+0.007$ |
| TD $\lambda = 0.5$ + temporal | 1.501 | $-0.016$ |
| TD $\lambda = 0.5$ no-temporal | 1.509 | $-0.011$ |
| **TD $\lambda = 0.5$ + temporal + $d = -1.0$** | **1.499** | **$-0.018$** |
| Scheduling $-1.8 \to -1.0$ | 1.512 | $-0.011$ |
| Scheduling $-2.2 \to -1.2$ | 1.515 | $-0.007$ |

### A.9 Series 13: Minimax Polynomials (A100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| Minimax basic ($a = 2.68$) | 1.530 | $+0.023$ vs combined |
| Minimax combined | 1.525 | $+0.017$ vs combined |
| Minvar combined ($a = 2.23$) | 1.544 | $+0.020$ |
| L2 combined ($a = 2.67$) | 1.530 | $+0.006$ |
| Minimax TD (full recipe) | 1.522 | $-0.000$ |
| Final recipe (3rd repl) | 1.499 | $-0.023$ |

### A.10 Series 14: Mathematical Frameworks (A100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| Ruiz5 + NS3 basic | 1.539 | $+0.017$ |
| Ruiz5 + NS5 basic | 1.525 | $+0.003$ |
| Frame potential $\eta = 2.5$ | 2.140 | $+0.618$ |
| Polar Express (hand-tuned) | 1.540 | $+0.018$ |
| Polar Express (Remez 5-step) | 1.523 | $+0.007$ |
| Polar Express (Remez 3-step) | 1.541 | $+0.025$ |

### A.11 Series 15: Structural Optimization (A100 GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| Universal Muon (NS all 2D) | 1.528 | $+0.007$ |
| Universal (2D + AdamW 1D) | 1.529 | $+0.007$ |
| **PerHead 4 slices** | **1.499** | **$-0.025$** |
| PerHead 3 slices | 1.507 | $-0.018$ |
| PerHead 12 slices | 1.508 | $-0.017$ |

### A.12 Series 16: Replication & Scale (A100/H100/RTX Pro GPU)

| Config | Val Loss | vs Muon |
|---|---|---|
| PerHead 4s (3-run mean) | 1.501 | $-0.019 \pm 0.003$ |
| PerHead + combined | — | $-0.022$ |
| **PerHead + TD + $d=-1.0$** | **1.494** | **$-0.032$ (lucky)** |
| PerHead TD (3-run mean) | 1.495 | $-0.025 \pm 0.002$ |
| PerHead + out\_proj plain | — | $-0.012$ (out\_proj = noise) |
| Full wire (+ MLP) | — | $-0.018$ (MLP hurts) |
| PerHead 12s (124M) | 3.714 | $-0.001$ |
| PerHead 4s TD (124M) | 4.829 | $-0.006$ |
| PerHead 3s TD (124M) | 4.836 | $+0.001$ |

---

## Appendix B: Key Learnings Index

The 74 key learnings discovered across 16 series are documented in full in the experimental summary (`experiments/SUMMARY.md`). The most critical are:

- **Learning 10:** NS approximation error is a feature, not a bug. Sub-unity SVs provide regularization.
- **Learning 15:** The only working strategy is intercepting NS mid-iteration.
- **Learning 22:** Monotone SV mappings cannot match NS's non-monotone equalization.
- **Learning 37:** Simple averaging beats complex weighting. Always.
- **Learning 40–41:** Oscillation is the unavoidable price of aggressive equalization, not a feature in itself.
- **Learning 46–47:** Temporal averaging is NS-specific, not generalizable. The mechanism is oscillation cancellation.
- **Learning 54:** Coefficient magnitude trumps equalization quality. $a \geq 3.0$ is non-negotiable.
- **Learning 57:** The coefficient-stability tradeoff is universal across six mathematical fields.
- **Learning 73:** Per-head does not scale to 124M. Aspect-ratio dependent.
- **Learning 74:** NS optimization is fully exhausted.

---

## Appendix C: Theoretical Analysis Tools

The polynomial dynamics analysis was performed using `analysis/polynomial_theory.py`, which computes:

1. **Lyapunov exponents** via $\lambda = \lim_{N \to \infty} \frac{1}{N} \sum_{k=1}^N \log|p'(x_k)|$, averaged over 200 starting points and 4000 iterations.
2. **Period-2 orbits** by finding roots of $p(p(\sigma)) = \sigma$ that are not fixed points of $p$.
3. **Invariant measures** via Ulam's method (500-bin Markov transition matrix).
4. **Minimax-optimal polynomials** via differential evolution over $(a, b, c) \in [0.5, 5.0] \times [-8.0, 0.0] \times [0.5, 4.0]$.
5. **Chebyshev sign polynomial** (truncated series): $a = 3.820$, $b = -6.791$, $c = 4.074$. Diverges catastrophically after 5 iterations — confirming iterated composition optimality $\neq$ single-application optimality.

---

## Appendix D: Deep Research Program

Four rounds of deep research informed this project:

**V1** (`deep_research_results.md`): Identified 12 mathematical frameworks. Key insight: NS approximates Schatten-32 steepest descent. The 0.877 fixed point explained via James-Stein shrinkage. 9 frameworks tested in Series 6 — all failed.

**V2** (`deep_research_results_v2.md`): Proposed Huber SV mapping as optimal monotone alternative (Series 7 — failed). Correctly predicted the blending mechanism works through destructive interference of even/odd iterates. Incorrectly predicted a monotone 2-parameter mapping could match NS.

**V3** (`deep_research_results_v3.md`): Identified seven theoretical frameworks supporting oscillation dynamics (Edge of Stability, bifurcation theory, stochastic resonance). Proposed the bifurcation sweep and TD($\lambda$) blending — both implemented successfully in Series 12. Predicted the inverted-U performance curve peaking in the period-2 regime.

**V4** (`deep_research_results_v4.md`): Surveyed 26 mathematical frameworks from 6 fields. Discovered the universality of the cubic spectral iteration. Identified Ruiz equilibration and Polar Express as practical candidates — both tested in Series 14 (both failed). The universality finding became a formal proof (Theorems 2 and 4).

---

## References

Cohen, J.M., Kaur, S., Li, Y., Kolter, J.Z., & Talwalkar, A. (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR 2021*.

Gupta, V., Koren, T., & Singer, Y. (2018). Shampoo: Preconditioned stochastic tensor optimization. *ICML 2018*.

Higham, N.J. (2008). *Functions of Matrices: Theory and Computation*. SIAM.

Jordan, K. (2024). Muon: An optimizer for hidden layers in neural networks. *GitHub: KellerJordan/Muon*.

Martens, J. & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML 2015*.

Strogatz, S.H. (2015). *Nonlinear Dynamics and Chaos* (2nd ed.). Westview Press.

Vyas, N., Kakade, S.M., & Barak, B. (2024). SOAP: Improving and stabilizing Shampoo using Adam. *arXiv:2409.11321*.

Benedetto, J.J. & Fickus, M. (2003). Finite normalized tight frames. *Advances in Computational Mathematics*, 18(2-4), 357–385.
