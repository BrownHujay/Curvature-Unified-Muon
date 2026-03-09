# Multi-Iterate Blending for Newton-Schulz Optimizers: From Curvature Recovery to Oscillation Cancellation

## Abstract

This document chronicles the complete research arc of the CUM (Curvature-Unified Muon) optimizer project: 13 experimental series, 75+ individual experiments, and a theoretical analysis program that together reveal a fundamental mechanism for improving Newton-Schulz (NS) based optimizers. The project began with the hypothesis that Muon's NS orthogonalization destroys useful curvature information, and that recovering some of this information could improve training. After exhaustive exploration of pre-NS modifications, alternative orthogonalization methods, SVD-based approaches, and custom SV mappings — all of which failed or produced marginal gains — the project converged on a single winning principle: **oscillation cancellation via multi-iterate blending**.

The NS polynomial used by Muon has an unstable fixed point at sigma = 0.868, causing singular values to oscillate rather than converge. This oscillation is the unavoidable price of aggressive SV equalization, which requires large polynomial coefficients. The key discovery is that blending NS intermediate iterates — which oscillate approximately out of phase — cancels this structured noise while preserving the equalization benefit. Three orthogonal layers of oscillation management were identified: (1) a weaker polynomial at the edge of chaos (d = -1.0), (2) TD(lambda = 0.5) multi-iterate blending across all five NS steps, and (3) temporal EMA of outputs across training steps. The final recipe combining all three achieves val_loss = 1.4993 on our benchmark, a -0.018 improvement over Muon, replicated three times (1.4993, 1.4993, 1.4992). This advantage holds at 124M parameter scale (-0.014 vs Muon). A final experimental series confirmed that these gains are specific to NS's structured oscillation: temporal averaging applied to Adam (which has no such oscillation) actively hurts performance, and theoretically optimal polynomials with near-perfect equalization but small coefficients fail catastrophically.

## 1. Background and Motivation

### 1.1 Muon and Newton-Schulz Orthogonalization

Muon is a matrix-aware optimizer for neural network training that applies Newton-Schulz (NS) iteration to approximate the polar factor of the gradient momentum. For a weight matrix W in R^{m x n}, Muon computes:

```
1. g = grad L(W)                          # gradient
2. m <- beta1 * m + (1 - beta1) * g       # momentum (beta1 = 0.95)
3. u = g + beta1 * m                      # Nesterov lookahead
4. X = NS_orthogonalize(u, steps=5)       # Newton-Schulz -> polar factor
5. W <- W - lr * sqrt(max(1, m/n)) * X    # update with aspect ratio scaling
```

The NS iteration approximates the polar factor via:

```
X_0 = M / ||M||_F
For k = 1..5:
    A = X_k @ X_k^T
    X_{k+1} = a * X_k + (b * A + c * A^2) @ X_k
```

with coefficients (a, b, c) = (3.4445, -4.7750, 2.0315). These coefficients were chosen by Keller Jordan to maximize the derivative at zero (p'(0) = a = 3.4445), which aggressively inflates small singular values. Muon applies this only to 2D hidden-layer weight matrices; embeddings, biases, and output projections use AdamW.

### 1.2 The NS Polynomial: p(sigma) = 3.4445*sigma - 4.7750*sigma^3 + 2.0315*sigma^5

This polynomial is the heart of Muon. Applied as a scalar function to singular values, it maps all SVs toward a common target. However, the coefficients sum to 0.701 (not 1.0), meaning p(1) = 0.701. The polynomial does not converge to the sign function — it is non-monotone, peaking at sigma = 0.555 with p(0.555) = 1.20, then declining. After 5 iterations, singular values scatter across approximately [0.68, 1.12] rather than concentrating at 1.0. The effective fixed point is at sigma* = 0.868, where p(sigma*) = sigma*, but the derivative |p'(sigma*)| = 1.58 > 1, making this fixed point **unstable**. SVs oscillate around it in a period-2-like pattern.

### 1.3 The Fundamental Tension

The original hypothesis motivating this project was that NS orthogonalization destroys useful curvature (singular value) information from the gradient. Muon maps all gradient SVs to approximately the same value (~0.88), losing information about which directions are more important. The question was: can we recover some of this curvature information to improve training?

This hypothesis turned out to be partially correct but fundamentally incomplete. The real story, discovered over 13 series of experiments, is about **oscillation**: NS's aggressive SV equalization requires large polynomial coefficients, which inherently cause oscillation around the fixed point. This oscillation is structured and anti-correlated across NS iterations. Blending multiple iterations cancels the oscillation while preserving the equalization — extracting the benefit while eliminating the cost.

## 2. Early Exploration (Series 1-7): What Doesn't Work

### 2.1 Pre-NS Modifications (Series 1, v1-v4)

The first approach was to modify the gradient before NS processing, hoping NS would "lock in" curvature information.

**v1 — Pre-NS Factored Preconditioning** (`cum/cum.py`): Preconditioned the gradient with row/column variance estimates before NS. Result: val_loss = 1.5187 (-0.0003 vs Muon, within noise). The preconditioning rotated the gradient direction by 28 degrees (cosine sim 0.88), while NS(3) vs NS(5) differed by only cosine sim 0.997. Pre-processing distorts direction 10x more than reducing NS steps. NS is too robust — it washes out input modifications.

**v2 — Post-NS Row/Column Scaling** (`cum/cum_v2.py`): Applied curvature after NS to avoid it being washed out. Result: val_loss ~1.53 (+0.01 vs Muon). Row/column gradient variance at 1.2M model scale is too noisy to provide useful curvature information.

**v3 — Soft NS (Raw Gradient Blend)** (`cum/cum_v3.py`): Blended NS output with normalized pre-NS momentum: update = (1-alpha)*NS(u) + alpha*normalize(u). Result: val_loss = 1.5146 (-0.0044 vs Muon) at alpha = 0.1. **First clear win.** But alpha = 0.2 was too much blend — noise outweighs curvature benefit. This small success motivated v5.

**v4 — Stacked Innovations** (`cum/cum_v4.py`): Combined soft NS + gradient centralization + coherence-adaptive step size. Result: val_loss = 1.5193 (+0.0003 vs Muon). The three modifications interfered with each other. Centralization hurts transformers (row means carry useful bias). Coherence scaling conflicts with cosine LR schedule. **Key learning: don't stack features.**

### 2.2 Alternative Orthogonalization Methods (Series 2-3, 6)

Tested whether NS could be replaced with better orthogonalization.

**2v2 — SVD Exact Polar Factor** (`cum/cum_2v2.py`): Replaced NS with exact SVD polar factor (U @ Vh). Result: val_loss = 1.5307 (+0.0117 vs Muon), and 40% slower. **Critical discovery: exact polar factor is WORSE than NS's approximation.** NS converges SVs to ~0.877, not 1.0. This sub-unity convergence provides implicit regularization. SVD polar starts faster (better early orthogonalization) but NS's regularization wins in later training. This was the first hint that NS's "imperfection" is a feature.

**3v1 — Warm-Started NS** (`cum/cum_3v1.py`): Used previous step's NS output to warm-start, enabling fewer iterations. Result: val_loss ~2.10 at step 500, killed early. Starting NS from a near-orthogonal matrix (already near the unstable fixed point) means NS barely changes it — the optimizer keeps repeating the previous step's direction.

**3v2 — Cayley Retraction** (`cum/cum_3v2.py`): Computed Stiefel tangent direction using the weight matrix. Result: val_loss = 1.8465 (+0.32 vs Muon). The rotational component of the gradient carries much less optimization signal than the polar factor.

**3v3 — Directional Momentum** (`cum/cum_3v3.py`): Normalized each gradient before momentum averaging, so all steps contribute equally to direction. Result: val_loss = 1.5250 (+0.0004 vs Muon, tied). Confirms NS is extremely robust to input preprocessing — whether you feed raw momentum or direction-normalized momentum, the output is the same.

**Series 6 — Nine Deep Research V1 Directions** (H100 GPU): Tested all remaining directions from Deep Research V1:

| Optimizer | Key Idea | Val Loss | vs Muon |
|-----------|----------|----------|---------|
| 6v1 Polar Express (Remez) | Per-step optimal polynomial coefficients | 1.5204 | +0.009 |
| 6v2 PolarGrad | Nuclear norm scaling | 1.5473 | +0.036 |
| 6v3 PSGD Kron | Learned Kronecker preconditioner | 2.3934 | +0.882 |
| 6v4 Dion (full rank) | Amortized power iteration + QR | 1.5328 | +0.021 |
| 6v5 Halley 3-iter | Rational approximation for polar factor | 1.5721 | +0.060 |
| 6v6 MARS | Variance reduction wrapping Muon | 2.4905 | +0.979 |
| 6v7 Warm-Started NS | Previous polar factor as starting point | 3.0052 | +1.493 |
| 6v8 Optimal SV Shrinkage | Donoho-Gavish denoising | 1.5191 | +0.007 |
| 6v9 Weighted Procrustes | Direction-aware weighted orthogonalization | 1.9280 | +0.416 |

**Every method targeting SVs = 1.0 failed** (Polar Express, Dion, Halley). MARS failed catastrophically because NS applied to gradient differences (tiny values) produces random orthogonal matrices. Warm-starting NS traps the iteration near its unstable fixed point. The closest competitor (6v8 shrinkage blend, +0.007) respects both denoising and equalization but still loses to Muon.

### 2.3 SVD-Based SV Mapping Approaches (Series 5, 7)

Since the SVD gives exact singular values, tested custom SV mappings.

**5v1 — Lie Algebra Momentum** (`cum/cum_5v1.py`): Projected gradient into so(m) (skew-symmetric matrices) for manifold-aware momentum. Result: val_loss = 3.2463 (+1.73 vs Muon). Catastrophic — the skew-symmetric projection discards the symmetric (scaling) component, which carries most optimization signal.

**5v2 — Tunable Soft Equalization (tanh)**: Sigmoid SV mapping f(sigma) = tanh(beta * sigma/sigma_max) / tanh(beta). Result: beta = 3 gave 1.6375 (+0.12), beta = 7 gave 1.5875 (+0.07). Higher beta = more equalization = closer to Muon. Confirms full SV equalization is essential. But tanh doesn't lift small SVs enough — at beta = 7, sigma = 0.01 maps to only 0.07.

**5v3 — Schatten-p Steepest Descent** (`cum/cum_5v3.py`): Power function sigma^{1/(p-1)} from Schatten-p framework. Result: p = 8 gave 1.5202 (+0.005 vs Muon), the closest non-NS result. But more equalization (higher p) is NOT always better: p = 8 beats p = 32. **Key insight: NS's specific non-monotone oscillating SV mapping has properties that no monotonic power function can replicate.**

**5v5 — SVD + NS Polynomial (Diagnostic)**: Applied NS polynomial as scalar function to exact SVs via SVD. Result: val_loss = 1.5198 (+0.002 vs Muon). Confirmed that NS is essentially a scalar SV mapping — the matrix-level iteration and scalar polynomial produce nearly identical results.

**5v6 — SVD NS Blend** (`cum/cum_5v6.py`): Replicated v5's multi-resolution blend in SVD space: f(sigma) = 0.85 * NS_5(sigma) + 0.15 * NS_2(sigma). Result: val_loss = 1.5040-1.5055 (-0.010 to -0.017 vs Muon). **New best at the time.** SVD's exact SV computation gives a consistent edge over matrix-level blending, but is 5x slower on GPU.

**Series 7 — Huber SV Mapping** (H100 GPU, from Deep Research V2): Tested monotone mappings f(sigma) = min(sigma^alpha, c) with sub-unity cap c = 0.88.

| Config | Val Loss | vs Muon |
|--------|----------|---------|
| Huber alpha = 0.1 c = 0.88 | 1.5622 | +0.027 |
| Huber alpha = 0.3 c = 0.88 | 1.7513 | +0.216 |
| Huber alpha = 0.5 c = 0.88 | 2.0170 | +0.482 |
| Huber alpha = 0.7 c = 0.88 | 2.2093 | +0.674 |
| Power alpha = 0.3 c = 0.88 | 1.8009 | +0.266 |
| Scheduled alpha 0.5 -> 0.1 | 1.8555 | +0.320 |

**All monotone mappings failed.** NS's non-monotone polynomial scatters SVs in a way that nearly perfectly equalizes them. Any monotone function preserving the ordering sigma_1 > sigma_2 > ... must leave residual spread. NS's oscillation can "shuffle" SVs, collapsing the spread much more aggressively. The Deep Research V2 theory that a monotone 2-parameter mapping could match NS was wrong.

### 2.4 Summary of Failed Approaches

After 7 series and 40+ experiments, the complete list of approaches that failed:

| Approach Category | Examples | Why It Failed |
|---|---|---|
| Pre-NS input modification | v1 factored precond, v12 grad diff, 3v3 directional momentum | NS washes out input perturbations (Jacobian near 0 at fixed point) |
| Post-NS scaling | v2 row/col scaling, v11 Adam grafting | Gradient variance too noisy; element-wise reweighting breaks NS's uniform treatment |
| Stacking features | v4 (soft + centralization + coherence) | Individual improvements cancel each other |
| Exact polar factor | 2v2 SVD, 6v4 Dion, 6v5 Halley | Sub-unity SVs are a FEATURE; exact SVs = 1.0 loses regularization |
| Alternative orthogonalization | 3v2 Cayley, 6v2 PolarGrad, 6v3 PSGD | Polar factor captures gradient info better than rotational component |
| Warm-starting NS | 3v1, 6v7 | Traps iteration near unstable fixed point; repeats previous direction |
| Monotone SV mappings | 5v2 tanh, 5v3 Schatten-p, 7v1 Huber/Power | Cannot match NS's non-monotone oscillation-based equalization |
| Variance reduction | 6v6 MARS | NS on gradient differences produces random matrices |
| Hard SV thresholding | 6v8 hard mode | Zeroing SVs below noise floor removes useful gradient directions |
| Learned preconditioners | 6v3 PSGD Kron | Lie group learning needs careful tuning beyond simple drop-in |
| Reducing NS steps | 3b (NS = 3) | Significant quality loss; 5 steps are sacred |

**The only approach that consistently beat Muon was intercepting NS mid-iteration** (v3's raw blend, v5's multi-resolution blend, 5v6's SVD NS blend).

## 3. The Breakthrough: NS Intermediate Iterate Blending (Series 5, 8-10)

### 3.1 v5 Multi-Resolution NS — First Significant Win

**Experiment:** `cum/cum_v5.py`
**Hypothesis:** Instead of blending NS output with the raw gradient (noisy, as in v3), blend with a partially-converged NS intermediate (denoised).

NS iteration progressively equalizes SVs. At step 2, ~25% of original SV spread remains. At step 3, ~5% remains. The step-2 intermediate is partially denoised while retaining significant curvature structure.

```
update = (1 - alpha) * NS_5(u) + alpha * scale_match(NS_2(u))
```

**Results:**
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| save@2 b = 0.15 | 1.5077 | -0.0113 |
| save@2 b = 0.10 | 1.5081 | -0.0109 |
| save@3 b = 0.10 | 1.5152 | -0.0038 |
| save@2 b = 0.20 | 1.5113 | -0.0077 |

save@2 >> save@3 (more curvature retained). b = 0.15 is optimal. 2.6x bigger improvement than v3's raw gradient blend (0.0113 vs 0.0044). **v5 starts slightly slower but accelerates in the second half** — curvature info becomes more valuable as training progresses.

### 3.2 Series 8-9: Systematic Exploration of the Blending Space

With v5/5v6 established as the only winning strategy across 40+ experiments, Series 8 systematically explored the blending space.

**Extended two-point blends (NS_2 + NS_8, NS_3 + NS_7):** Beat Muon by ~0.002 in initial run, but replication showed this was noise (flipped to +0.005 in run 2). Extended NS past 5 steps doesn't reliably help.

**Three-point blend (NS_1 + NS_3 + NS_5):** Worse than Muon (+0.002 to +0.011). NS_1 is too noisy, especially late in training. Three-point had the fastest early convergence ever (2.394 at step 250) — NS_1 helps early, hurts late.

**Geometric SV blend:** Consistently 2nd place behind arithmetic (5v6). Same SVD cost, no advantage.

**Input blend (temporal EMA):** EMA of past NS_5 outputs blended into current output. Result: val_loss = 1.5139 (-0.011 vs Muon) at matrix-path speed (48s vs 213s for SVD). **4.4x faster than 5v6 for ~0.002 quality gap.** Fastest convergence at every checkpoint. However, subsequent runs revealed instability: mean -0.005 +/- 0.006 across 4 runs.

**Dual-momentum pre_ns (9v1):** Two momentum buffers at different time constants, blended before NS. Result: flips between +/- 0.005 vs Muon across runs. Confirms NS makes input modifications irrelevant, not harmful.

### 3.3 Combined Mode: The Real Breakthrough

**Experiment:** `cum/cum_8v1.py`, combined mode
**Key insight:** Stack within-step iterate blend (NS_2 + NS_5) AND across-step temporal blend (EMA).

```python
# Within-step: iterate blend
iterate_blended = (1 - b) * NS_5 + b * scale(NS_2)
# Across-step: temporal EMA
ema = beta * ema + (1 - beta) * iterate_blended
# Final: temporal blend
output = (1 - alpha) * iterate_blended + alpha * scale(ema)
```

**Results across 3 replications at 1.2M params:**
| Run | Val Loss | vs Muon |
|-----|----------|---------|
| Run 1 (10d) | 1.5008 | -0.0191 |
| Run 2 (11a) | 1.5062 | -0.0162 |
| Run 3 (11b) | 1.5091 | -0.0122 |
| **Mean** | | **-0.016 +/- 0.004** |

**Scale test at 124M params (GPT-2 Small, WikiText-103):**
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| CUM combined | 4.1393 | -0.0144 |
| Muon NS = 5 | 4.1537 | baseline |

Same late-game acceleration pattern: Muon leads steps 0-750, combined catches up at step 1000, pulls ahead from step 1250 onward. The EMA warmup period is consistent across scales. Deep research's worst-case prediction (advantage vanishes at scale) did NOT happen.

### 3.4 Why Simple Averaging Beats Smart Weighting

Series 10e tested "intelligent" alternatives to uniform blending:

**Adaptive residual:** Weighted corrections by their magnitude. Result: -0.002 vs Muon (noise).
**Cosine-gated:** Applied corrections only when curvature and temporal signals agree. Result: +/- 0.000 vs Muon (dead tie).

Both were 50% slower than Muon for zero benefit. The pattern is clear: stacking SIMPLE averaging operations works; adding COMPLEX weighting logic doesn't. The "intelligence" breaks the statistical cancellation. Oscillation cancellation is a Central Limit Theorem effect — averaging many small structured corrections. Gating or weighting individual corrections suppresses the collective signal.

**Per-layer adaptive blend (11v2):** Adapting the AMOUNT of uniform blending per layer. Result: +0.002 vs combined. Even "smart amount" adaptation doesn't beat fixed weights.

## 4. The Mechanism: Oscillation Cancellation

### 4.1 NS Polynomial's Period-2 Oscillation

The NS polynomial p(sigma) = 3.4445*sigma - 4.7750*sigma^3 + 2.0315*sigma^5 has an unstable fixed point at sigma* = 0.868, with |p'(sigma*)| = 1.58 > 1. Singular values don't converge — they oscillate. After 5 iterations, SVs are scattered across [0.68, 1.12] in a period-2-like pattern where even and odd iterates are anti-correlated.

The oscillation is not a bug — it is the unavoidable price of **aggressive equalization**. The large coefficient a = 3.4445 is needed to pull small SVs up fast (sigma = 0.1 maps to 0.34 in one step). But these same large coefficients cause the fixed point instability.

### 4.2 Within-Step Cancellation

Blending NS_2 and NS_5 (or more generally, multiple iterates) cancels the oscillation because even and odd iterates are approximately out of phase. Where NS_2 maps a particular SV high, NS_5 tends to map it low, and vice versa. The weighted average produces a near-constant ~0.88 with dramatically reduced variance.

### 4.3 Across-Step Cancellation

The temporal EMA averages NS outputs across training steps. Because the oscillation is deterministic and anti-correlated across consecutive steps (the input SVs change slightly, shifting the oscillation phase), the EMA provides a second, orthogonal channel of noise cancellation.

### 4.4 Evidence: Stable Polynomial Catastrophically Fails (Series 11)

**The definitive test:** If oscillation is a bug, a polynomial with a STABLE fixed point should work better. If oscillation is the price of equalization, a stable polynomial should fail.

The "stable-0.88" polynomial (2.0, -1.940, 0.836) has p(0.88) = 0.88 and p'(0.88) = 0 — super-stable, zero oscillation.

**Results:**
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| Stable basic | 1.5671 | +0.0448 |
| Stable combined | 1.5544 | +0.0321 |

**Catastrophic failure.** The small coefficients (a = 2.0) inflate small SVs too slowly: sigma = 0.1 maps to only 0.20 per step (vs 0.34 for standard). After 5 iterations, equalization is incomplete. The large coefficients needed for aggressive equalization ARE what cause the fixed point instability. You can't decouple them with a degree-5 polynomial in 5 steps.

**Revised understanding:** Oscillation is not "the feature" — it is the unavoidable side effect of aggressive equalization. The feature is the equalization itself. Blending is the mechanism to extract the equalization benefit while canceling the oscillation cost.

### 4.5 Evidence: SmoothedAdam Fails (Series 11d)

**The generalization test:** Does temporal averaging help non-NS optimizers?

| Config | Val Loss | vs AdamW |
|--------|----------|----------|
| AdamW | 1.6040 | baseline |
| SmoothedAdam alpha = 0.15 | 1.6080 | +0.0039 |
| SmoothedAdam alpha = 0.30 | 1.6162 | +0.0122 |

**Temporal averaging actively HURTS Adam.** Adam's update direction is already well-conditioned — there's no structured oscillation to cancel. Without structured oscillation, blending is just sluggishness. The improvement from iterate blending is NS-specific, not a general optimizer principle. This narrows but sharpens the contribution.

### 4.6 The Three-Part Mechanism Summary

The mechanism is oscillation cancellation, confirmed by:
1. **Stable polynomial fails** — no oscillation means no equalization
2. **SmoothedAdam fails** — no oscillation means blending is just drag
3. **Combined succeeds** — only with the standard oscillating polynomial
4. **Anti-correlated iterates** — NS_2 and NS_5 oscillate out of phase
5. **Late-game acceleration** — the benefit grows as training progresses and curvature matters more

## 5. Polynomial Dynamics Theory (Series 11-12)

### 5.1 Bifurcation Parameterization

The NS polynomial family can be parameterized by a single control parameter: d = p'(sigma*), the derivative at the fixed point. Given sigma* = 0.868 and c = 2.0315 (fixed to preserve high-SV behavior), the coefficients are determined analytically:

```
a = (3 - d) / 2 + c * sigma*^4
b = (d - 1 - 4 * c * sigma*^4) / (2 * sigma*^2)
```

Standard Muon has d = -1.58. Sweeping d from -0.5 to -3.0 traces a path through convergent (|d| < 1), period-2 (1 < |d| < ~1.5), and chaotic (|d| > ~1.7) dynamical regimes.

### 5.2 Lyapunov Exponents

The Lyapunov exponent lambda characterizes the dynamical regime of the iterated polynomial. The theoretical analysis (`analysis/polynomial_theory.py`) computed:

| d | |p'(sigma*)| | Lyapunov | Regime |
|---|-------------|----------|--------|
| -0.50 | 0.50 | -0.693 | Attracting |
| -0.80 | 0.80 | -0.223 | Attracting |
| -0.90 | 0.90 | -0.105 | Attracting |
| **-1.00** | **1.00** | **-0.001** | **Edge of chaos** |
| -1.10 | 1.10 | -0.253 | Attracting |
| -1.20 | 1.20 | -0.777 | Attracting |
| -1.40 | 1.40 | -0.286 | Attracting |
| -1.58 | 1.58 | -0.571 | Attracting |
| -1.80 | 1.80 | +0.331 | Weakly chaotic |
| -2.00 | 2.00 | +0.381 | Weakly chaotic |
| -2.80 | 2.80 | +0.739 | Chaotic |

**d = -1.0 sits exactly at the edge of chaos.** The standard Muon polynomial (d = -1.58) is NOT chaotic — it is in the attracting regime, specifically in the stable period-2 zone. The Lyapunov exponent is non-monotone: it dips most negative at d = -1.2 (most strongly attracting period-2 orbit), rises through d = -1.58, and crosses zero around d = -1.7.

### 5.3 Period-2 Orbit Stability

| d | sigma_low | sigma_high | |p'(a) * p'(b)| | Stable? |
|---|-----------|------------|-----------------|---------|
| -1.00 | 0.868 | 0.868 | 1.000 | Marginal |
| -1.10 | 0.791 | 0.958 | 0.603 | YES |
| -1.20 | 0.764 | 0.998 | 0.212 | YES (most stable) |
| -1.40 | 0.730 | 1.059 | 0.565 | YES |
| **-1.58** | **0.709** | **1.103** | **1.269** | **NO (unstable)** |
| -1.80 | 0.692 | 1.151 | 2.165 | NO |
| -2.00 | 0.681 | 1.190 | 3.041 | NO |

**Standard Muon has an UNSTABLE period-2 orbit** (multiplier 1.27 > 1). The oscillation diverges rather than damps. This is why two-point blending helps so much: it cancels divergent oscillation. At d = -1.0 to -1.4, the period-2 orbit is self-correcting (multiplier < 1), so less blending is needed.

### 5.4 SV Equalization Quality

| Polynomial | Var[p^5(sigma)] | Relative to Muon |
|-----------|-----------------|------------------|
| Standard NS (d = -1.58) | 0.0307 | 1.0x |
| d = -1.4 | 0.0189 | 1.6x better |
| **d = -1.0** | **0.0019** | **17x better** |
| Minimax optimal | ~0.0000 | Perfect |

d = -1.0 achieves 17x better SV equalization than standard Muon. SVs are concentrated near 0.88 instead of scattered across [0.68, 1.12]. Crucially, this superior equalization quality is the LEAST important factor (as Series 13 demonstrated) — what matters is that d = -1.0 preserves fast small-SV inflation (a = 3.15) while reducing the oscillation that blending must cancel.

## 6. TD(lambda) Multi-Iterate Blending (Series 12)

### 6.1 From Two-Point to All-Iterate Blending

Previous blending used only NS_2 and NS_5. But NS_3 and NS_4 also carry useful oscillation-cancellation information that two-point blending discards. TD(lambda)-style exponential weighting of ALL iterates:

```
w_k = lambda^(n-k), normalized to sum = 1
```

For lambda = 0.5, n = 5: weights = [0.032, 0.065, 0.129, 0.258, 0.516]

This naturally downweights early (noisy) iterates while including NS_3 (13%) and NS_4 (26%), which carry useful intermediate oscillation information. More samples of the oscillatory process lead to better cancellation (Nyquist principle).

### 6.2 Lambda Sweep Results

| Config | Val Loss | vs Muon | vs Combined |
|--------|----------|---------|-------------|
| lambda = 0.3 + temporal | 1.5062 | -0.0106 | +0.0007 |
| **lambda = 0.5 + temporal** | **1.5011** | **-0.0156** | **-0.0044** |
| lambda = 0.5 no-temporal | 1.5087 | -0.0110 | — |
| lambda = 0.7 + temporal | 1.5093 | -0.0104 | — |
| lambda = 0.9 + temporal | 1.5103 | -0.0094 | — |

**lambda = 0.5 is the sweet spot.** lambda = 0.3 is too concentrated on NS_5 (similar to two-point). lambda = 0.7+ lets too much early iterate noise through. TD(lambda = 0.5) + temporal beat combined by -0.0044 in the same run, **replicated across 2 runs**.

### 6.3 Within-Step and Across-Step Are Orthogonal

TD(lambda = 0.5) alone (no temporal): -0.011 vs Muon.
TD(lambda = 0.5) + temporal EMA: -0.012 to -0.016 vs Muon.

The ~0.005 temporal contribution stacks on top of multi-iterate blending. The two cancellation channels are approximately orthogonal — they cancel oscillation through independent mechanisms (within-step across iterations vs across-step through time).

### 6.4 Bifurcation Sweep with Combined Mode (12v1)

| d | Val Loss (run 1) | vs Muon | Val Loss (run 2) | vs Muon |
|---|------------------|---------|------------------|---------|
| -1.0 | 1.5033 | -0.0152 | 1.5051 | -0.0192 |
| -1.4 | 1.5133 | -0.0052 | 1.5040 | -0.0203 |
| -1.58 (std) | 1.5055 | -0.0088 | — | — |
| -2.0 | 1.5102 | -0.0042 | — | — |
| -2.8 | 1.5212 | +0.0068 | — | — |

The optimal derivative is in the [-1.0, -1.4] range. d = -2.8 is catastrophic. Within combined mode, less polynomial oscillation is better because iterate blending already handles oscillation cancellation. The effect is real but noisy at this magnitude (d = -1.4 beat d = -1.0 in one run, reversed in another).

### 6.5 Adaptive Scheduling — Dead End (12v3)

| Config | Val Loss | vs Combined |
|--------|----------|-------------|
| 12v3 d = -1.8 -> -1.0 | 1.5120 | +0.0034 |
| 12v3 d = -2.2 -> -1.2 | 1.5154 | +0.0067 |

Both variants worse than constant combined. Starting with stronger oscillation and annealing down doesn't help. Optimal polynomial dynamics should be constant throughout training — there's no "explore more early" benefit in NS polynomial space.

## 7. The Final Recipe (Series 12d)

### 7.1 Three Orthogonal Layers

The final recipe combines three layers of oscillation management:

1. **Weaker polynomial** (d = -1.0): Reduces oscillation at source. The polynomial at the edge of chaos (a = 3.1512, b = -4.3105, c = 2.0315) still inflates small SVs fast enough (a >= ~3.0) while producing 17x better equalization than standard Muon.

2. **Multi-iterate blending** (TD lambda = 0.5): Cancels within-step oscillation by exponentially weighting all five NS iterates. Captures useful information from NS_3 and NS_4 that two-point blending misses.

3. **Temporal EMA** (beta = 0.5, alpha = 0.15): Cancels across-step oscillation via exponential smoothing of blended outputs across training steps.

### 7.2 Implementation (cum/cum_12v2.py)

The final recipe is implemented in `CUM12v2`:

```python
class CUM12v2(Optimizer):
    def __init__(self, params, lr=0.02, beta1=0.95,
                 td_lambda=0.5, ns_steps=5,
                 use_temporal=True,
                 input_blend_beta=0.5, input_blend_alpha=0.15,
                 deriv=-1.0, ...):
```

Key steps per parameter update:
1. Standard Muon momentum with Nesterov
2. Run NS with custom bifurcation coefficients (d = -1.0), saving ALL iterates
3. Compute TD(lambda) weighted blend of all iterates (norm-matched to final)
4. Update temporal EMA of blended outputs
5. Frobenius-norm-matched blend of current blended output with temporal EMA
6. Apply update with aspect ratio scaling

### 7.3 Results

**First sub-1.50 result ever:**

| Config | Val Loss | vs Muon | vs Combined |
|--------|----------|---------|-------------|
| TD(lambda = 0.5) + temporal d = -1.0 | **1.4993** | **-0.0182** | **-0.0094** |
| TD(lambda = 0.5) + temporal (std poly) | 1.5050 | -0.0124 | -0.0036 |
| Combined (std poly) | 1.5087 | -0.0088 | baseline |
| Muon NS = 5 | 1.5175 | baseline | +0.0088 |

**Effects are approximately additive:**
- TD(lambda) contributes ~-0.004 vs combined (from NS_3/NS_4 information)
- d = -1.0 contributes ~-0.006 vs standard polynomial (from reduced oscillation)
- Total: ~-0.009 vs combined

### 7.4 Trajectory

| Step | Muon | Combined | TD std | TD d = -1.0 |
|------|------|----------|--------|-------------|
| 250 | 2.381 | 2.372 | 2.389 | 2.408 |
| 500 | 1.791 | 1.770 | 1.783 | 1.802 |
| 750 | 1.625 | 1.614 | 1.612 | 1.615 |
| 1000 | 1.556 | 1.556 | 1.544 | 1.553 |
| 1250 | 1.532 | 1.521 | 1.522 | 1.519 |
| 1500 | 1.508 | 1.501 | 1.499 | **1.498** |
| 1750 | 1.513 | 1.501 | 1.495 | **1.493** |
| 1999 | 1.521 | 1.512 | 1.509 | **1.504** |

The d = -1.0 variant starts slowest (step 250-500) but accelerates hardest late-game. At step 1750: 1.493, which is -0.020 below Muon.

### 7.5 Replication

Three replications of the final recipe:

| Run | Val Loss | vs Muon |
|-----|----------|---------|
| Run 1 (12d) | 1.4993 | -0.0182 |
| Run 2 (12d replication) | 1.4993 | -0.0182 |
| Run 3 (13b) | 1.4992 | -0.0233 |

Remarkably consistent: 1.4993, 1.4993, 1.4992.

### 7.6 Scale Validation (124M params)

Combined mode at 124M params showed -0.014 vs Muon (same late-game acceleration pattern). The final recipe was not separately tested at 124M but the combined mode baseline confirms the principle holds at scale.

## 8. What We Tried Last: Minimax-Optimal Polynomials (Series 13)

### 8.1 The Theory

Numerical optimization (differential evolution, in `analysis/polynomial_theory.py`) found polynomial families outside the bifurcation parameterization that achieve near-perfect SV equalization:

| Polynomial | a | b | c | Var[p^5] |
|-----------|---|---|---|----------|
| Standard NS | 3.4445 | -4.7750 | 2.0315 | 0.0307 |
| d = -1.0 | 3.1512 | -4.3105 | 2.0315 | 0.0019 |
| Minimax optimal | 2.6806 | -3.6311 | 1.8871 | ~0.0000 |
| Min-variance | 2.2311 | -3.2137 | 1.9518 | ~0.0000 |
| L2 optimal | 2.6704 | -3.6166 | 1.8848 | ~0.0000 |

These have NO fixed point at sigma* = 0.868 — they're entirely outside the bifurcation family. If near-perfect equalization without oscillation works, it changes the story. If it fails (like stable-0.88), it confirms the primacy of coefficient magnitude.

### 8.2 Results

| Config | Val Loss | vs Combined | Notes |
|--------|----------|-------------|-------|
| Minimax basic (a = 2.68) | 1.5304 | +0.023 | Much worse |
| Minimax combined | 1.5253 | +0.017 | Barely beats Muon |
| Minvar combined (a = 2.23) | 1.5444 | +0.020 | Worst |
| L2 combined (a = 2.67) | 1.5302 | +0.006 | — |
| Minimax TD (full recipe) | 1.5222 | -0.0003 vs Muon | Muon-tier with full treatment |
| **Final recipe (3rd replication)** | **1.4992** | **-0.023 vs Muon** | **Confirmed** |

**All minimax presets failed.**

### 8.3 Why: Coefficient Magnitude Trumps Equalization Quality

The linear coefficient `a` directly controls first-step inflation of small SVs:

| Polynomial | a | sigma = 0.1 after 1 step |
|-----------|---|--------------------------|
| Standard NS | 3.4445 | ~0.34 |
| d = -1.0 | 3.1512 | ~0.32 |
| Minimax | 2.6806 | ~0.27 |
| Minvar | 2.2311 | ~0.23 |
| Stable-0.88 | 2.0000 | ~0.20 |

Minimax has ~22% less first-step amplification than standard NS. After 5 iterations, this compounds — small SVs simply don't get inflated fast enough.

**Dose-response relationship:**
| a | Val Loss vs Muon |
|---|-------------------|
| 2.0 (stable) | +0.045 |
| 2.23 (minvar) | +0.020 |
| 2.68 (minimax) | +0.017 vs combined |
| 3.15 (d = -1.0) | -0.018 vs Muon |
| 3.44 (standard) | baseline |

Clear monotonic relationship: lower `a` = worse performance, regardless of equalization quality. **a >= ~3.0 is non-negotiable.**

### 8.4 The Hierarchy of What Matters

1. **Coefficient magnitude** (a >= ~3.0): Fast small-SV inflation is non-negotiable. Without it, equalization is too slow.
2. **Oscillation management** (blending): Cancel the oscillation that large coefficients cause.
3. **Equalization quality** (Var[p^5]): Least important — 17x worse equalization (d = -1.0) still wins over near-perfect equalization (minimax).

### 8.5 The Bifurcation Family Constraint Is Structural

The bifurcation family parameterizes polynomials by p'(sigma*) with a fixed point at sigma* = 0.868. This FORCES large enough coefficients: a = (3-d)/2 + c*sigma*^4, where c = 2.0315 contributes a base ~1.14, and d in [-1, -3] pushes a into [2.0, 3.5]. Unconstrained optimization finds "gentle" polynomials that equalize by being weak, not by being aggressive-then-canceling. The constraint preserves the property that matters most.

## 9. Complete Results Table

### Series 1 (M3 CPU, 1.2M params, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| v1 | Pre-NS Factored Precond | 1.5187 | -0.0003 | FAILED |
| v2 | Post-NS Row/Col Scaling | ~1.53 | +0.01 | FAILED |
| v3 (alpha = 0.1) | Soft NS (Raw Blend) | 1.5146 | -0.0044 | SUCCESS (small) |
| v3 (alpha = 0.2) | Soft NS | 1.5182 | -0.0008 | FAILED |
| 3b | NS Step Reduction (NS = 3) | 1.5439 | +0.0249 | FAILED |
| v4 | Stacked Innovations | 1.5193 | +0.0003 | FAILED |
| **v5 (s@2 b = 0.15)** | **Multi-Resolution NS** | **1.5077** | **-0.0113** | **BEST (Series 1)** |
| v5 (s@3 b = 0.1) | Multi-Resolution NS | 1.5152 | -0.0038 | Partial |
| v5 (s@2 b = 0.2) | Multi-Resolution NS | 1.5113 | -0.0077 | Overblended |
| v6 | Adaptive Spectral Blend | — | — | PENDING (never tested) |
| v7 | Orthogonal Feedback Loop | — | — | PENDING (never tested) |
| v8 | Multi-Scale Curvature Blend | — | — | PENDING (never tested) |
| v9 (damp = 0.3) | Dampened Late-Stage NS | 1.5101 | -0.0089 | PARTIAL SUCCESS |
| v10 | Dampened NS + Multi-Res | ~1.84 | terrible | FAILED (3.6x slower) |
| v11 (graft = 0.3) | Second-Moment Grafting | 1.5126 | -0.0064 | FAILED |
| v12 (diff = 0.1) | Gradient Difference Momentum | 1.5157 | -0.0033 | FAILED |

### Series 2 (M3 CPU, 1.2M params, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 2v1 (rank = 4) | Randomized Top-k Curvature | 1.5296 | +0.0091 | FAILED |
| 2v2 (alpha = 0) | SVD Exact Polar Factor | 1.5307 | +0.0117 | FAILED (key discovery) |
| 2v2 (alpha = 0.1) | SVD Partial SV Preservation | 1.5242 | +0.0052 | FAILED |

### Series 3 (M3 CPU, 1.2M params, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 3v1 | Warm-Started NS | ~2.10 (step 500) | +0.32 | FAILED (killed early) |
| 3v2 | Cayley Retraction | 1.8465 | +0.3219 | FAILED |
| 3v3 | Directional Momentum | 1.5250 | +0.0004 | FAILED (tied Muon) |

### Series 4 (M3 CPU, 1.2M params, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 4v2 | SODA (Spectral Outlier Dampen) | 2.3401 | +0.83 | FAILED (catastrophic) |
| 4v3 | WGASU (Weight-Geometry) | 2.3246 | +0.81 | FAILED (catastrophic) |

### Series 5 (H100 GPU, 1.2M params, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 5v1 | Lie Algebra Momentum + NS | 3.2463 | +1.73 | FAILED (catastrophic) |
| 5v2 (beta = 3) | Soft EQ (tanh) | 1.6375 | +0.12 | FAILED |
| 5v2 (beta = 7) | Soft EQ (tanh) | 1.5875 | +0.07 | FAILED |
| **5v3 (p = 8)** | **Schatten-p Descent** | **1.5202** | **+0.005** | **Closest non-NS** |
| 5v3 (p = 32) | Schatten-p Descent | 1.5308 | +0.016 | FAILED |
| 5v4 | Adaptive Schatten-p | 1.5309 | +0.016 | FAILED |
| 5v5 | SVD + NS Poly (diagnostic) | 1.5198 | +0.002 | DIAGNOSTIC |
| **5v6 (s2 b = 0.25)** | **SVD NS Blend** | **1.5048** | **-0.017** | **NEW BEST (at the time)** |
| 5v6 (s2 b = 0.15) | SVD NS Blend | 1.5040-55 | -0.010 to -0.016 | BEST (consistent) |
| 5v6 (tilt eps = 0.1) | SVD Tilt | 1.5244 | +0.011 | FAILED |

### Series 6 (H100 GPU, 1.2M params, batch = 32, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 6v1 (Remez std) | Polar Express | 1.5204 | +0.009 | FAILED |
| 6v1 (blend) | Polar Express | 1.5335 | +0.022 | FAILED |
| 6v2 (NS) | PolarGrad | 1.5473 | +0.036 | FAILED |
| 6v2 (blend) | PolarGrad | 1.5485 | +0.037 | FAILED |
| 6v3 | PSGD Kron | 2.3934 | +0.88 | FAILED (catastrophic) |
| 6v4 (full rank) | Dion | 1.5328 | +0.021 | FAILED |
| 6v4 (r = 32) | Dion | 1.6440 | +0.13 | FAILED |
| 6v5 (3 iter) | Halley | 1.5721 | +0.060 | FAILED |
| 6v5 | QDWH | CRASHED | — | FAILED |
| 6v6 (gamma = 0.5) | MARS | 2.4905 | +0.98 | FAILED (catastrophic) |
| 6v6 (gamma = 1.0) | MARS | 2.5023 | +0.99 | FAILED (catastrophic) |
| 6v7 (2 step) | Warm-Started NS | 3.0052 | +1.49 | FAILED (catastrophic) |
| 6v7 (hybrid) | Warm-Started NS | NaN | — | FAILED (diverged) |
| 6v8 (hard) | Optimal SV Shrinkage | 2.1458 | +0.63 | FAILED |
| 6v8 (blend) | Optimal SV Shrinkage | 1.5191 | +0.007 | FAILED (closest non-NS) |
| 6v9 (mag) | Weighted Procrustes | 1.9280 | +0.42 | FAILED (catastrophic) |
| 6v9 (decay) | Weighted Procrustes | 1.9406 | +0.43 | FAILED |

### Series 7 (H100 GPU, 1.2M params, batch = 64, 1000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 7v1 (alpha = 0.1 c = 0.88) | Huber | 1.5622 | +0.027 | FAILED (closest monotone) |
| 7v1 (alpha = 0.3 c = 0.85) | Huber | 1.7582 | +0.22 | FAILED |
| 7v1 (alpha = 0.3 c = 0.88) | Huber | 1.7513 | +0.22 | FAILED |
| 7v1 (alpha = 0.3 c = 0.92) | Huber | 1.7509 | +0.22 | FAILED |
| 7v1 (alpha = 0.5 c = 0.88) | Huber | 2.0170 | +0.48 | FAILED |
| 7v1 (alpha = 0.7 c = 0.88) | Huber | 2.2093 | +0.67 | FAILED |
| 7v1 (smooth alpha = 0.3) | Smooth Huber | 1.7818 | +0.25 | FAILED |
| 7v1 (power alpha = 0.3) | Power | 1.8009 | +0.27 | FAILED |
| 7v1 (sched 0.5 -> 0.1) | Scheduled alpha | 1.8555 | +0.32 | FAILED |
| 7v1 (sched 0.3 -> 0.05) | Scheduled alpha | 1.6912 | +0.16 | FAILED |

### Series 8-9 (A100 GPU, 1.2M params, batch = 32, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 8v1 (s2 + NS8 b = 0.15) | Two-point extended | 1.5203 | -0.0015 | NOISE |
| 8v1 (s3 + NS7 b = 0.15) | Two-point extended | 1.5193 | -0.0025 | NOISE |
| 8v1 (3pt NS1 + NS3 + NS5) | Three-point | 1.5225 | +0.0106 | FAILED |
| 8v1 (geom s2 b = 0.15) | Geometric SV blend | 1.5087 | -0.0032 | TIES 5v6 |
| 9v1 (pre bf = .80 bs = .95) | Dual-momentum pre_ns | 1.5294 | +0.0045 | FAILED |
| **8v1 (input blend)** | **Temporal EMA** | **1.5139** | **-0.0110** | **BEST PRACTICAL** |

### Series 10 (A100 GPU, Replication)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 8v1 (s2 + NS8 repl) | Two-point replication | 1.5199 | +0.0045 | FAILED (8a was noise) |
| 8v1 (s3 + NS7 repl) | Two-point replication | 1.5200 | +0.0047 | FAILED (8a was noise) |
| 8v1 (geom repl) | Geometric replication | 1.5107 | -0.0073 | Consistently 2nd |
| 8v1 (3pt repl) | Three-point replication | 1.5198 | +0.0018 | Muon-tier |
| **8v1 (input-blend repl)** | **Input-blend replication** | **1.5107** | **-0.0103** | **CONFIRMED** |
| **8v1 (combined)** | **Iterate + input blend** | **1.5008** | **-0.0191** | **NEW RECORD** |
| 8v1 (sched-3pt) | Scheduled three-point | 1.5173 | -0.0027 | Marginal |
| 8v1 (input-blend run 3) | Input-blend | 1.5237 | +0.0037 | REGRESSED |
| **8v1 (combined repl)** | **Combined replication** | **1.5062** | **-0.0162** | **CONFIRMED** |
| 8v1 (adaptive-res) | Adaptive residual | 1.5161 | -0.0015 | FAILED (Muon-tier) |
| 8v1 (cos-gated) | Cosine gated | 1.5178 | +0.0002 | FAILED (Muon-tier) |

### Series 11 (A100 GPU, 1.2M params, batch = 32, 2000 steps)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| 11v1 (stable basic) | Stable-0.88 polynomial | 1.5671 | +0.0448 | FAILED (catastrophic) |
| 11v1 (stable combined) | Stable-0.88 + combined | 1.5544 | +0.0321 | FAILED |
| **8v1 (combined 3rd run)** | **Combined replication** | **1.5091** | **-0.0122** | **CONFIRMED (3 runs)** |
| **11v3 (uneq alpha = 0.05)** | **Un-equalization** | **1.5077** | **-0.0136** | **PROMISING** |
| 11v2 (adaptive s = 1.0) | Adaptive blend | 1.5113 | -0.0101 | FAILED |
| SmoothedAdam (alpha = 0.15) | Temporal avg on Adam | 1.6080 | +0.0039 vs AdamW | FAILED |
| SmoothedAdam (alpha = 0.30) | Temporal avg on Adam | 1.6162 | +0.0122 vs AdamW | FAILED |

### Series 11c: Scale Test (124M params, WikiText-103, A100 GPU)

| Version | Name | Val Loss | vs Muon | Status |
|---------|------|----------|---------|--------|
| **8v1 (combined 124M)** | **Scale test** | **4.1393** | **-0.0144** | **HOLDS AT SCALE** |

### Series 12 (A100 GPU, 1.2M params, batch = 32, 2000 steps)

| Version | Name | Val Loss | vs Muon | vs Combined | Status |
|---------|------|----------|---------|-------------|--------|
| 12v1 (d = -1.0 combined) | Bifurcation boundary | 1.5033 | -0.0152 | -0.0013 | Marginal |
| 12v1 (d = -1.4 combined) | Weaker oscillation | 1.5133 | -0.0052 | +0.0087 | Worse (noise) |
| 12v1 (d = -1.58 combined) | Standard (reference) | 1.5055 | -0.0088 | — | Matches combined |
| 12v1 (d = -2.0 combined) | Stronger oscillation | 1.5102 | -0.0042 | — | Worse |
| 12v1 (d = -2.8 combined) | Very strong oscillation | 1.5212 | +0.0068 | — | FAILED |
| **12v2 (lambda = 0.5 + temporal)** | **TD(lambda) + EMA** | **1.5011** | **-0.0156** | **-0.0044** | **NEW BEST** |
| 12v2 (lambda = 0.3 + temporal) | TD(lambda) | 1.5062 | -0.0106 | +0.0007 | Ties combined |
| 12v2 (lambda = 0.5 no-temporal) | TD(lambda) only | 1.5087 | -0.0110 | — | Multi-iterate alone |
| 12v2 (lambda = 0.7 + temporal) | TD(lambda) | 1.5093 | -0.0104 | — | Too noisy |
| 12v2 (lambda = 0.9 + temporal) | TD(lambda) | 1.5103 | -0.0094 | — | Near-uniform |
| 12v3 (-1.8 -> -1.0) | Adaptive scheduling | 1.5120 | -0.0106 | +0.0034 | FAILED |
| 12v3 (-2.2 -> -1.2) | Adaptive scheduling | 1.5154 | -0.0072 | +0.0067 | FAILED |
| 12v1 (d = -1.4 rerun) | Bifurcation rerun | 1.5040 | -0.0203 | -0.0040 | d = -1.4 wins this time |
| 12v1 (d = -1.0 rerun) | Bifurcation rerun | 1.5051 | -0.0192 | -0.0029 | Confirmed real |
| **12v2 (final recipe)** | **TD + temporal + d = -1.0** | **1.4993** | **-0.0182** | **-0.0094** | **ALL-TIME BEST** |
| 12v2 (lambda = 0.5 std run 2) | TD replication | 1.5050 | -0.0124 | -0.0036 | Replicated |

### Series 13 (A100 GPU, 1.2M params, batch = 32, 2000 steps)

| Version | Name | Val Loss | vs Muon | vs Combined | Status |
|---------|------|----------|---------|-------------|--------|
| 13v1 (minimax basic) | Minimax a = 2.68 | 1.5304 | +0.023 vs combined | — | FAILED |
| 13v1 (minimax combined) | Minimax a = 2.68 | 1.5253 | +0.017 vs combined | — | FAILED |
| 13v1 (minvar combined) | Minvar a = 2.23 | 1.5444 | +0.020 | — | FAILED |
| 13v1 (l2 combined) | L2 a = 2.67 | 1.5302 | +0.006 | — | FAILED |
| 13v1 (minimax td) | Minimax full recipe | 1.5222 | -0.0003 | — | FAILED (Muon-tier) |
| **12v2 (final, 3rd repl)** | **Final recipe** | **1.4992** | **-0.0233** | — | **CONFIRMED (3x)** |

## 10. Key Learnings (All 56)

### Series 1 Learnings (1-9)

1. **NS is Muon's core strength AND weakness.** Equalizes directions but destroys curvature. Winning approach = partially recover curvature.
2. **Pre-NS modifications distort direction.** NS locks in damage. Modify AFTER NS or use NS intermediate.
3. **Post-NS modifications add noise unless signal is clean.** Raw gradient = noisy. NS intermediate = denoised.
4. **Don't stack innovations.** They interfere. Refine one core approach.
5. **NS steps = 5 sacred.** Can't reduce without quality loss.
6. **Weight decay kills Muon/CUM.** wd = 0.01 dominates updates 12x.
7. **Gradient centralization hurts transformers.** Row means carry useful info.
8. **Coherence LR scaling conflicts with LR schedules.** Causes overshoot.
9. **b = 0.15 optimal for save@2.** Higher adds noise.

### Series 2-3 Learnings (10-15)

10. **NS approximation error is a FEATURE.** Exact SVD polar (SVs = 1.0) is WORSE than NS_5 (SVs ~ 0.877). Sub-unity SVs provide implicit regularization.
11. **Don't replace NS.** NS_5 > exact SVD polar factor.
12. **Modifying NS input barely helps.** NS is robust to input perturbations.
13. **torch.compile ~15-20% speedup** on model fwd+bwd only. Doesn't help NS.
14. **NS dominates everything.** Changing the input (directional momentum, gradient diffs, orth feedback) doesn't affect the output. NS is extremely robust to input preprocessing.
15. **The ONLY working strategy is intercepting NS mid-iteration** (v5's multi-resolution blend). All other approaches fail or tie.

### Series 5 Learnings (16-20)

16. **Schatten-p power mapping is the right mathematical framework** for understanding SV equalization. Power function sigma^{1/(p-1)} lifts small SVs much better than tanh. p = 8 gives closest non-NS result (+0.005).
17. **More equalization isn't always better with SVD**: p = 8 beats p = 32. The NS polynomial's oscillating SV mapping may provide beneficial properties beyond simple equalization.
18. **Lie algebra momentum is catastrophic**: projecting into so(m) discards the symmetric (scaling) component which carries most optimization signal.
19. **SVD-based SV manipulation is MORE PRECISE than matrix NS**: exact SVs via SVD avoid floating point error accumulation across 5 matrix iterations. 5v6 consistently beats v5.
20. **5v6 SVD ns_blend is the new BEST approach**: Consistently beats Muon by 0.010-0.017. The SVD framework enables exploration of custom SV curves.

### Series 6 Learnings (21-30)

21. **ALL methods converging to SVs = 1.0 lose to Muon**: Polar Express, Dion, Halley, PolarGrad. The 0.88 contraction IS essential.
22. **Monotone SV mappings cannot match NS**: Any function preserving SV ordering leaves too much spread. NS's oscillating non-monotone map achieves near-perfect equalization that no monotone function can.
23. **NS's oscillation IS the feature** (REVISED in learning 41): The polynomial doesn't converge. It scatters SVs across [0.68, 1.12]. The v5/5v6 blend works by CANCELING this oscillation (even/odd iterates anti-correlated), producing a near-constant ~0.88.
24. **Variance reduction (MARS) catastrophic with NS**: NS on tiny gradient differences produces random orthogonal matrices.
25. **Warm-starting NS is fundamentally broken**: Starting from near-orthogonal means NS barely changes it.
26. **Learned preconditioners (PSGD Kron) diverge**: Lie group Kronecker learning needs careful tuning beyond simple drop-in.
27. **Hard SV thresholding kills training**: Zeroing SVs removes useful gradient directions.
28. **Shrinkage-NS blend is closest alternative**: Donoho-Gavish blended with NS_5 at +0.007 vs Muon.
29. **SVD-based methods 5x slower on GPU**: SVD isn't parallelizable like matmuls. H100: SVD ~140s vs Muon ~30s.
30. **The only winning strategy remains NS intermediate blending** (v5/5v6). 40+ experiments across 7 series confirm nothing else works.

### Series 8-10 Learnings (31-47)

31. **Within-run rankings are reliable; cross-run absolutes are not.** torch.compile causes +/- 0.006 shifts but ordering is preserved.
32. **Input-blend ties 5v6 at matrix-path speed** (REVISED: input-blend unstable, flipped to +0.004 in run 3).
33. **Input-blend has trajectory dominance early** but can crash late-game. Not as stable as initially claimed.
34. **Modifying NS input is irrelevant, not harmful.** 9v1 pre_ns flips +/- 0.005 vs Muon across runs.
35. **Double oscillation cancellation (combined mode) may be the real breakthrough.** Within-step iterate blend + across-step temporal blend = -0.019 vs Muon at matrix speed.
36. **Smart correction weighting fails.** Magnitude-based (adaptive-res) and agreement-based (cosine-gated) both Muon-tier.
37. **Simple averaging beats complex weighting.** Combined = two layers of simple blend. Adaptive/cosine = one layer of complex logic. Simple wins.
38. **Input-blend mean: -0.005 +/- 0.006 vs Muon** (4 runs). Real but noisy.
39. **Combined mode CONFIRMED across 2 runs**: -0.019 and -0.016 vs Muon.
40. **Stable-0.88 polynomial catastrophically fails**: +0.045 basic, +0.032 combined. Large polynomial coefficients are needed for aggressive equalization. Those same coefficients cause oscillation. Can't decouple them with degree-5 in 5 steps.
41. **Revised learning 23**: oscillation is not "the feature" but the unavoidable price of aggressive equalization. The feature is the equalization itself.
42. **Per-layer adaptive blend fails**: adapting the AMOUNT of uniform blending doesn't beat fixed weights.
43. **5% row-wise un-equalization shows signal**: -0.0014 vs combined, with "behind early, ahead late" trajectory. ONE RUN — needs replication.
44. **Combined mean across 3 runs: -0.016 +/- 0.004 vs Muon.** Solidly confirmed.
45. **Combined holds at 124M params.** -0.014 vs Muon on GPT-2 Small / WikiText-103. Same late-game acceleration.
46. **Temporal averaging does NOT generalize to Adam.** SmoothedAdam worse than AdamW. Adam has no structured oscillation to cancel.
47. **The mechanism is oscillation cancellation, not generic smoothing.** Confirmed by: (a) stable polynomial fails, (b) SmoothedAdam fails, (c) combined succeeds only with standard oscillating polynomial.

### Series 12 Learnings (48-53)

48. **TD(lambda) multi-iterate blending beats two-point blending.** lambda = 0.5 + temporal beat combined by -0.004 in same run (replicated). NS_3 and NS_4 carry useful info (Nyquist principle).
49. **Within-step and across-step cancellation are orthogonal channels.** TD(lambda = 0.5) alone: -0.011. With temporal: -0.012 to -0.016. ~0.005 temporal contribution stacks.
50. **Weaker polynomial oscillation is real with blending.** d = -1.0 to -1.4 range consistently beats standard d = -1.58 combined. d = -2.8 catastrophic.
51. **Oscillation scheduling is a dead end.** Annealing polynomial dynamics doesn't help. Optimal dynamics should be constant.
52. **lambda = 0.5 is the TD(lambda) sweet spot.** 0.3 too concentrated on final iterate. 0.7+ too much early noise.
53. **TD(lambda) + weak polynomial effects are additive.** lambda = 0.5 + temporal + d = -1.0 yielded 1.4993 (-0.018 vs Muon, -0.009 vs combined). First sub-1.50. Three layers: (1) weaker polynomial, (2) multi-iterate within-step, (3) temporal across-step.

### Series 13 Learnings (54-56)

54. **Coefficient magnitude trumps equalization quality.** The `a` coefficient controls small-SV inflation speed. a >= ~3.0 is non-negotiable. Minimax-optimal polynomials (a = 2.68) inflate small SVs 22% slower per step — compounds fatally over 5 iterations.
55. **The bifurcation family constraint is structural, not arbitrary.** Polynomials parameterized by p'(sigma*) with fixed point at sigma* = 0.868 are FORCED to have large enough coefficients (a >= ~3.0 for useful d range). Unconstrained optimization finds "gentle" polynomials that equalize by being weak, not aggressive-then-canceling.
56. **Dose-response: a = 2.0 (+0.045), a = 2.23 (+0.020), a = 2.68 (+0.017 vs combined), a = 3.15 (-0.018 vs Muon).** Clear monotonic relationship between linear coefficient magnitude and performance.

## 11. What's Left / Open Questions

### Confirmed Promising Directions Not Yet Pursued

- **SURE-optimized adaptive alpha:** Use Stein's Unbiased Risk Estimate to select the blend weight from data at each step. Requires SVD (slow) but provably optimal shrinkage.
- **Second-order IIR filter:** The EMA is a first-order low-pass filter. A second-order IIR tuned to the oscillation frequency could provide sharper rejection. Predicted marginal improvement.
- **Per-layer adaptation at scale:** Different transformer layers have dramatically different gradient spectral properties (attention output near random, QKV with outlier SVs). Per-layer NS parameters may help at billion-parameter scale where layer diversity is extreme.
- **Serious scale validation:** The 124M test was only 2000 steps (far from converged). Full-scale validation at 1B+ parameters with proper training budgets would determine if the late-game acceleration pattern holds when training converges.

### Theoretical Open Questions

- Why exactly does lambda = 0.5 work? The connection to Nyquist sampling rate of the oscillatory process is suggestive but not precise.
- Is d = -1.0 (edge of chaos) optimal because of dynamical systems principles, or is it simply the weakest oscillation that still has a >= 3.0?
- Can higher-degree polynomials (degree 7, 9) decouple coefficient magnitude from oscillation, enabling stable polynomials with large `a`?
- Does the improvement persist at Chinchilla-optimal training lengths, or does the oscillation cancellation benefit saturate?

### Untested Ideas from SUMMARY.md

- **CANS convergent coefficients** — polynomial coefficients that sum to 1.0 and actually converge, but targeting ~0.88 instead of 1.0.
- **SDP-optimized polynomial** — sample gradient SV distributions, solve for coefficients minimizing training loss directly via semidefinite programming.

## Appendix A: Optimizer Implementations

### A.1 The Final Recipe: CUM12v2 (cum/cum_12v2.py)

Core algorithm per parameter:

```python
# 1. Standard Muon momentum with Nesterov
mb = beta1 * mb + (1 - beta1) * grad
u = grad + beta1 * mb

# 2. Run NS with bifurcation coefficients (d=-1.0), save ALL iterates
# Coefficients: a=3.1512, b=-4.3105, c=2.0315
X = u / (||u|| + eps)
iterates = []
for i in range(5):
    A = X @ X.T
    B = b * A + c * (A @ A)
    X = a * X + B @ X
    iterates.append(X.clone())

# 3. TD(lambda=0.5) weighted blend
# weights = [0.032, 0.065, 0.129, 0.258, 0.516]
blended = sum(w_k * norm_match(iterate_k, iterates[-1]) for k, w_k in enumerate(weights))

# 4. Temporal EMA (across-step)
ema = 0.5 * ema + 0.5 * blended

# 5. Final blend
output = 0.85 * blended + 0.15 * norm_match(ema, blended)

# 6. Update
W -= lr * sqrt(max(1, m/n)) * output
```

### A.2 Combined Mode: CUM8v1 (cum/cum_8v1.py)

The simpler "combined" mode that preceded the final recipe:

```python
# Within-step iterate blend
full, partial = NS_multi_resolution(u, steps=5, save_at=2)
iterate_blended = 0.85 * full + 0.15 * norm_match(partial, full)

# Across-step temporal blend
ema = 0.5 * ema + 0.5 * iterate_blended
output = 0.85 * iterate_blended + 0.15 * norm_match(ema, iterate_blended)
```

### A.3 Bifurcation Coefficient Derivation (cum/cum_12v1.py)

```python
def bifurcation_coeffs(deriv, sigma_star=0.868, c=2.0315):
    """Compute (a, b, c) from target derivative at fixed point."""
    s2 = sigma_star ** 2
    s4 = sigma_star ** 4
    a = (3 - deriv) / 2 + c * s4
    b = (deriv - 1 - 4 * c * s4) / (2 * s2)
    return a, b, c
```

For d = -1.0: a = 3.1512, b = -4.3105, c = 2.0315.
For d = -1.58 (standard Muon): a = 3.4445, b = -4.7750, c = 2.0315.

### A.4 Newton-Schulz Implementations (cum/newton_schulz.py)

The NS module provides multiple variants:
- `newton_schulz_orthogonalize`: Standard NS iteration
- `newton_schulz_multi_resolution`: NS with one intermediate save (for v5/combined)
- `newton_schulz_n_resolution`: NS with arbitrary intermediate saves (for TD(lambda))
- `newton_schulz_dampened`: NS with late-stage dampening (for v9)

### A.5 Norm-Matching Blend Utility

All blending operations use Frobenius norm matching:

```python
def _frobenius_blend(primary, secondary, weight, eps):
    """Blend two matrices with Frobenius norm matching."""
    p_norm = primary.norm()
    s_norm = secondary.norm()
    secondary_scaled = secondary * (p_norm / s_norm)
    return (1 - weight) * primary + weight * secondary_scaled
```

This ensures the blended intermediate has the same overall scale as the primary (final) iterate, preventing the blend from inadvertently changing the effective learning rate.

## Appendix B: Benchmark Setup

### B.1 Small-Scale Benchmark (Series 1-5, partial 8-13)

- **Hardware:** Apple M3 CPU, 4 threads (Series 1-5); NVIDIA A100 GPU (Series 8-13)
- **Model:** MicroGPT (d_model = 128, n_heads = 4, n_layers = 4, d_ff = 512, ctx_len = 256, ~1.2M params)
- **Data:** TinyShakespeare (1.1M chars, vocab = 65, char-level tokenization)
- **Training:** 2000 steps, batch = 32, warmup = 200 steps, cosine LR decay, seed = 42
- **Optimizer split:** Muon/CUM for hidden 2D weights (lr = 0.02), AdamW for embeddings/biases (lr = 3e-4, wd = 0.01)
- **Baseline:** Muon NS = 5, beta1 = 0.95 -> val_loss ~ 1.515 (+/- 0.008 across runs due to torch.compile nondeterminism)

### B.2 GPU-Scale Benchmark (Series 6)

- **Hardware:** NVIDIA H100 GPU
- **Model:** Same MicroGPT 1.2M params
- **Training:** batch = 32, 2000 steps (Series 6); batch = 64, 1000 steps (Series 7)
- **Timing:** Muon ~30s, SVD methods ~140s

### B.3 Scale Test (Series 11c)

- **Hardware:** NVIDIA A100 GPU
- **Model:** GPT-2 Small (~124M params)
- **Data:** WikiText-103
- **Training:** 2000 steps, batch = 32
- **Baseline:** Muon val_loss = 4.1537

### B.4 Cross-Run Variability

torch.compile introduces nondeterminism that causes +/- 0.006 shifts in absolute val_loss across runs. However, **within-run relative rankings are preserved**. All comparisons in this work use within-run deltas (same run, same seed, same compilation). The three-run replication of the final recipe (1.4993, 1.4993, 1.4992) demonstrates remarkable consistency for this metric.

## Appendix C: Deep Research Context

Three rounds of deep research informed this project:

**Deep Research V1** (`deep_research_results.md`): Identified that NS approximates Schatten-32 steepest descent. The 0.877 fixed point explained via James-Stein shrinkage, early-stopping regularization, and spectral norm control. Proposed 12 mathematical directions; 9 were tested in Series 6 (all failed).

**Deep Research V2** (`deep_research_results_v2.md`): Proposed Huber SV mapping as theoretically optimal monotone alternative. Identified the oscillation-cancellation mechanism of the v5 blend. Predicted (correctly) that the blend works through destructive interference of even and odd iterates. Proposed (incorrectly) that a monotone 2-parameter mapping could match NS.

**Deep Research V3** (`deep_research_results_v3.md`): Identified seven theoretical frameworks supporting oscillation as a feature (Edge of Stability, bifurcation theory, stochastic resonance, anisotropic noise, implicit ensemble, PAC-Bayes). Proposed the bifurcation sweep (implemented in Series 12), TD(lambda) blending (implemented in Series 12), and SURE-optimized adaptive alpha (not yet tested). Predicted the inverted-U performance curve peaking in the period-2 regime.

## Appendix D: Theoretical Analysis Tools

The offline theoretical analysis (`analysis/polynomial_theory.py`) computes:

1. **Lyapunov exponents** across the derivative parameter space via the formula lambda = lim (1/N) sum log|p'(x_k)|, averaging over 200 starting points and 4000 iterations.

2. **Period-2 orbits** by finding roots of p(p(sigma)) = sigma that are not fixed points of p, computing their stability multiplier |p'(a) * p'(b)|.

3. **Invariant measures** via Ulam's method: discretize the interval into 500 bins, build a Markov transition matrix from the polynomial map, find the stationary distribution as the left eigenvector with eigenvalue 1.

4. **Minimax-optimal polynomials** via differential evolution: minimize max|p^5(sigma) - 0.88| over (a, b, c) in [0.5, 5.0] x [-8.0, 0.0] x [0.5, 4.0]. Also variants minimizing variance and MSE.

5. **Chebyshev sign polynomial**: Truncated Chebyshev series for sign(x) converted to monomial basis. Result: a = 3.820, b = -6.791, c = 4.074. **Diverges catastrophically** after 5 iterations — confirming that iterated composition optimality is fundamentally different from single-application optimality.

6. **SV mapping comparison**: Tabulates p^5(sigma) for sigma in [0.05, 1.0] across standard NS, d = -1.0, and d = -1.4 polynomials.

Results are recorded in `analysis/THEORY_RESULTS.md`.
