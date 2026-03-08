# Deep Research V3: Polynomial Design Space + Temporal Averaging as General Principle

## What This Project Is

I'm building CUM (Curvature-Unified Muon), an optimizer that extends Muon for training neural networks. Muon uses Newton-Schulz (NS) iteration to orthogonalize gradient updates — it applies a polynomial p(σ) = aσ + bσ³ + cσ⁵ five times to the gradient's singular values, approximately equalizing them. This is faster than SGD/Adam because equal-magnitude directions prevent any single direction from dominating training.

**Current best:** "Combined mode" — a double oscillation cancellation strategy that hits -0.019 val_loss improvement over standard Muon on a 1.2M parameter language model. This uses within-step iterate blending (blend NS₂ and NS₅ outputs) plus across-step temporal blending (EMA of past blended outputs).

## What We've Discovered (60+ Experiments, 10 Series)

### The Core Finding: NS Oscillation Cancellation

The NS polynomial (a=3.4445, b=-4.7750, c=2.0315) was designed to converge to the polar factor (SVs→1.0). But:
1. **SVs→1.0 is WRONG.** We tested 10+ methods that achieve exact polar decomposition (Halley iteration, Dion, QDWH, Polar Express). ALL perform worse than NS₅. The sub-unity SVs (~0.88) provide implicit regularization.
2. **NS₅ doesn't converge.** After 5 iterations, SVs oscillate around ~0.88 with structured noise. The polynomial's fixed point at σ≈0.868 is UNSTABLE (|p'(0.868)|≈1.58 > 1).
3. **The oscillation is cancellable.** NS₂ and NS₅ have anti-correlated oscillation patterns. Blending them (85% NS₅ + 15% NS₂, Frobenius norm-matched) cancels much of the structured noise. Similarly, EMA-averaging NS₅ outputs across training steps cancels temporal oscillation.
4. **Double cancellation is additive.** Within-step cancellation (iterate blend, ~-0.005-0.010 vs Muon) + across-step cancellation (temporal EMA, ~-0.005 vs Muon) combined gives ~-0.019. The two axes are approximately orthogonal.

### What Works
- **NS iterate blending** (within-step): blend NS₂ into NS₅ at 15% → -0.005 to -0.010 vs Muon
- **Temporal NS output averaging** (across-step): EMA of past NS₅ outputs, β=0.5, blend at 15% → -0.005±0.006 vs Muon
- **Combined mode** (both): -0.019 vs Muon in best run, at matrix-path speed (54s vs 39s for Muon)
- **SVD-based SV manipulation**: Exact SVD + polynomial applied to scalar SVs → more precise than matrix NS, -0.010 to -0.016 vs Muon (but 5x slower due to SVD)

### What Fails (and Why)
- **ALL methods targeting SVs=1.0** — exact polar factor loses to NS's 0.88 contraction (10+ experiments)
- **ALL monotone SV mappings** — any order-preserving function leaves too much SV spread. NS's NON-MONOTONE oscillating polynomial achieves near-perfect equalization that no monotone function can
- **Modifying NS INPUT** — NS's Jacobian ≈ 0 near its fixed point. Pre-NS gradient modifications (dual-momentum, directional momentum, gradient differences) get contracted away. ±0.005 = noise
- **Smart correction weighting** — adaptive residual (magnitude-weighted) and cosine-gated (agreement-filtered) corrections both fail. The oscillation cancellation is a bulk STATISTICAL effect (CLT across many SVs). Per-element intelligence breaks the statistics
- **Warm-starting NS** — starting from previous polar factor (near-orthogonal) means NS barely changes it → repeats last step
- **Variance reduction (MARS)** — NS applied to tiny gradient differences produces random orthogonal matrices
- **Learned preconditioners (PSGD Kron)** — Lie group Kronecker learning diverges without careful tuning
- **Lie algebra momentum** — projecting into so(m) discards the symmetric gradient component
- **Extended NS (7-8 steps)** — more iterations but same final quality (±0.005). Save point matters more than total steps
- **Three-point blending** — NS₁+NS₃+NS₅ has fastest early convergence but worst final loss. NS₁ = too noisy late

### Key Mathematical Insights
- The NS polynomial's TRUE fixed point is at σ≈0.868 (solving p(σ)=σ), but it's unstable
- p'(0.868) ≈ -1.58, so the iteration alternates above/below the fixed point
- After 5 iterations, SVs scatter across [0.68, 1.12] — structured oscillation
- The v5/combined blend works by canceling this: even and odd iterate outputs are anti-correlated
- The resulting "effective SV mapping" after cancellation gives near-constant ~0.88 with very low variance
- This 0.88 contraction provides implicit regularization that exact polar (SVs=1.0) doesn't

## What I Need Researched

### 1. Optimal Polynomial Design for Direct SV Targets

I derived a "stable-0.88" polynomial: (a, b, c) = (2.0, -1.940, 0.836) where:
- p(0.88) ≈ 0.88 (IS a fixed point)
- p'(0.88) ≈ 0 (super-stable, zero derivative)
- Small SVs grow toward 0.88 in 5 steps

**Questions:**
- Is this the optimal degree-5 odd polynomial for targeting flat 0.88?
- What does Chebyshev/minimax approximation theory say? We want to minimize max_σ |p⁵(σ) - 0.88| for σ ∈ (0, 1) after initial Frobenius normalization
- Could we use Remez algorithm to find optimal coefficients?
- What about rational approximations (Padé) instead of polynomial?
- Is there a tradeoff between convergence speed (fast approach to 0.88) and stability (low oscillation near 0.88)?
- The current unstable polynomial + blending achieves LOW VARIANCE near 0.88 via cancellation. Can a stable polynomial achieve EQUALLY LOW VARIANCE directly?
- Could we optimize coefficients via gradient-based hyperparameter optimization (differentiating through NS iterations)?

### 2. Temporal Averaging of Optimizer Subroutines — General Theory

Our key novel finding is: EMA-averaging the outputs of an iterative optimization subroutine improves training. This applies to NS but the principle should generalize.

**Questions:**
- Is there existing theory on "temporal coherence" of optimization subroutine outputs?
- Adam uses EMA of gradients (first moment) and EMA of gradient squares (second moment). Is Adam ALREADY doing a version of this? If so, why does additional EMA on top of NS help?
- K-FAC uses Kronecker-factored Fisher estimation updated periodically. Would EMA-smoothing the Kronecker factors help?
- Shampoo computes preconditioners via matrix powers. Would temporal averaging of Shampoo's preconditioners improve convergence?
- Power iteration (used for spectral norm estimation) is iterative. Would intermediate-iterate blending help there?
- Is there a connection to Polyak averaging / stochastic weight averaging (SWA)? Those average the PARAMETERS, we average the SUBROUTINE OUTPUTS (different thing)
- What's the theoretical framework? Is this related to variance reduction? Denoising? Control variates?

### 3. Oscillation as Implicit Exploration vs. Bug

If the stable polynomial (no oscillation) works equally well: oscillation was a bug, and blending was a complex fix for a simple problem (wrong coefficients).

If the stable polynomial is worse: the oscillation itself provides beneficial diversity/regularization, similar to noise injection or Langevin dynamics.

**Questions:**
- Is there theory on when DETERMINISTIC oscillation (vs stochastic noise) is beneficial for optimization?
- Connection to chaotic optimization (deliberately using chaos for exploration)?
- Could the oscillation be providing a form of "implicit ensemble" — each training step sees a slightly different effective optimizer due to SV oscillation?
- What does the dynamical systems literature say about optimization near unstable fixed points?
- Is there an optimal "oscillation amplitude"? Too little = boring (exact polar), too much = noisy, just right = NS₅ + blending?

### 4. Per-Layer Spectral Adaptation

Different transformer layers have very different gradient spectra. Our uniform NS₅ treatment is suboptimal.

**Questions:**
- What's known about the spectral properties of gradients in different transformer layers (attention QKV, FFN, embeddings)?
- Is there theory on optimal per-layer preconditioning that accounts for varying condition numbers?
- Connection to block-diagonal Fisher information / natural gradient methods?
- Could we derive an ANALYTICAL formula for optimal blend weight given a layer's spectral profile?
- Would "spectral routing" work — route different SV components through different numbers of NS iterations?

### 5. Optimal Fraction of Curvature Recovery After Equalization

We're testing restoring 5% of gradient curvature (row-wise) after NS equalization.

**Questions:**
- James-Stein estimation: what's the optimal shrinkage intensity toward the "equalized" estimate?
- Donoho-Gavish optimal singular value thresholding: how does this relate to our problem?
- What fraction of information in gradient matrices is "curvature signal" vs "noise"?
- Is there theory on the optimal balance between direction equalization and magnitude preservation?
- Connection to empirical Bayes methods for estimating the "true" gradient from noisy observations?

### 6. Scaling Predictions

All our experiments are on a 1.2M parameter model (MicroGPT on TinyShakespeare, 2000 steps).

**Questions:**
- As model size grows (10M, 100M, 1B), do NS oscillation effects grow, shrink, or stay constant?
- Do gradient spectral properties change with scale in ways that affect our approach?
- Does the optimal SV target (currently ~0.88) change with model size?
- Are there scaling laws for optimizer improvement magnitude?
- What's the minimum model size where our improvements would be practically meaningful (e.g., >0.1 val_loss improvement)?

## What NOT to Suggest

Based on 60+ failed experiments, DO NOT suggest:
- **Any method targeting SVs = 1.0** (polar decomposition, exact orthogonalization) — confirmed worse, 10+ experiments
- **Warm-starting NS** from previous step's output — fundamentally broken (NS barely changes near-orthogonal input)
- **MARS / variance reduction** with NS — catastrophic (NS on gradient differences → random orthogonal matrices)
- **Element-wise post-NS reweighting** (second-moment grafting, Adam-magnitude injection) — fails at any strength
- **Shampoo/SOAP** — different optimizer entirely, not what we're exploring
- **Simple Adam variants** (AdaFactor, Lion, etc.) — not NS-based
- **Replacing NS with "better" orthogonalization** — NS₅ > every alternative we tested
- **Monotone SV mappings** — mathematically cannot match NS's equalization
- **Lie algebra / manifold optimization** — catastrophic for this use case

## Output Format

For each idea, provide:
1. **Core idea** (2-3 sentences)
2. **Mathematical formulation** (equations)
3. **Why it might work** (theoretical justification)
4. **Why it might fail** (honest assessment)
5. **Computational cost** relative to Muon
6. **Relevant papers** (with specific results/theorems to look up)
7. **Implementation sketch** (pseudocode, ~10 lines)

## Prioritization

Prioritize ideas that are:
1. **Mathematically principled** — have theoretical backing, not just heuristics
2. **Computationally feasible** — same cost as Muon or at most 2x
3. **Genuinely different** from our 60+ tried approaches
4. **Build on our discoveries** — oscillation cancellation, double cancellation, stable polynomial
5. **Likely to scale** — effects should grow (or at least persist) with model size
