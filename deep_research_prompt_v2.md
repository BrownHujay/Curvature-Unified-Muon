# Deep Research V2: Optimal SV Mapping for Matrix Optimization

## Context Update (What Worked Since Last Time)

We've made significant progress. Our best optimizer now **beats Muon by 0.010-0.017 val_loss** on our benchmark. Here's what we discovered:

### Key Breakthrough: NS ≈ Scalar SV Polynomial
We proved that Newton-Schulz iteration is equivalent to applying a scalar polynomial `p(σ) = 3.4445σ - 4.775σ³ + 2.0315σ⁵` independently to each singular value. Singular vectors are PRESERVED through the iteration. This means NS can be replicated exactly with SVD + scalar function application.

### What Works: SVD + Polynomial Blend
Our best approach (CUM 5v6):
1. Compute momentum `u = g + β₁*m` (standard Muon)
2. SVD: `U, S, Vh = svd(u)`
3. Scale: `S_scaled = S / ||S||₂` (same as NS normalization)
4. Apply NS polynomial 2x and 5x to each SV independently:
   - `S₂ = p(p(S_scaled))` — partial equalization, retains curvature
   - `S₅ = p(p(p(p(p(S_scaled)))))` — full equalization
5. Blend: `S_out = 0.75 * S₅ + 0.25 * S₂`
6. Reconstruct: `orth = U @ diag(S_out) @ Vh`
7. Update: `W -= lr * orth * scale`

**This is better than matrix-level NS because SVD gives EXACT singular values**, avoiding floating point error accumulation across 5 matrix iterations.

### What We Learned From Our Full Experiment History (30+ experiments)

**SV equalization is essential:**
- No non-equalization approach comes within 0.07 of Muon
- Schatten-p power mapping σ^{1/(p-1)} at p=6-8 gets +0.004 from Muon (closest non-NS)
- Full polar factor (SVs=1.0) is WORSE than NS (SVs≈0.877) — sub-unity provides regularization

**The NS polynomial's specific SV mapping matters:**
- NS polynomial iterated 5x gives a specific curve that's better than any simple power function
- The polynomial has NO stable fixed points — SVs oscillate around 0.868
- After 5 iterations, SVs are approximately 0.877 (average of oscillation)
- This oscillation may provide beneficial regularization

**Partial equalization from NS₂ intermediate carries curvature info:**
- NS after 2 steps is partially denoised but retains SV ordering
- Blending 15-25% of NS₂ with NS₅ improves results by 0.01-0.017 vs Muon
- This is the ONLY modification to NS that consistently helps

**SVD-based is better than matrix-level for blending:**
- SVD + scalar polynomial gives more precise SV manipulation
- Matrix NS accumulates floating-point error across iterations
- Our SVD-based blend beats the matrix-level blend (v5) consistently

### Negative Results (Definitely Don't Work)
- Pre-NS gradient modifications (preconditioning, directional momentum, Lie algebra)
- Post-NS reweighting (row/col scaling, Adam-style grafting)
- Alternative orthogonalization (Cayley retraction, QR-based)
- Partial SV processing (top-k dampening, spectral outlier removal)
- Weight-geometry aware updates (using W's spectral structure)
- tanh/sigmoid SV mapping (doesn't lift small SVs enough)

## What I Need Researched

### 1. Optimal Scalar SV Mapping Function
Given that the ENTIRE optimizer reduces to choosing a scalar function f: R → R applied to singular values, what is the OPTIMAL such function?

- **Connection to minimax polynomial approximation**: NS's polynomial was optimized for CONVERGENCE SPEED. What polynomial/function minimizes TRAINING LOSS instead? These are different optimization targets.

- **Optimal shrinkage theory**: Donoho-Gavish, Stein's shrinkage, James-Stein — these give optimal ways to shrink noisy singular values for ESTIMATION. How do they translate to OPTIMIZATION? Is the optimal SV mapping for optimization the same as optimal shrinkage for denoising?

- **Connection to Schatten norm steepest descent**: We confirmed NS ≈ Schatten-32. What if Schatten-p with p ≠ 32 is better for optimization at our scale? Is there theory on optimal p?

- **Why does NS₂ intermediate help?** The blend of NS₂ and NS₅ creates a specific SV curve. What mathematical property of this curve makes it better than flat equalization? Can we characterize the curve analytically and then optimize it?

### 2. Better Polynomial/Function Design
Instead of iterating NS's polynomial 5x and blending intermediates, can we design a SINGLE scalar function that directly maps SVs to the optimal targets?

- **Chebyshev/Remez polynomial approximation**: Given a target SV mapping (e.g., the effective curve from our best blend), find the optimal polynomial approximation of a given degree. Then apply it in ONE step instead of 5 iterative steps.

- **Rational function approximation**: Zolotarev and related theory uses rational functions (polynomial/polynomial) instead of pure polynomials. These converge much faster. Could a carefully designed rational function give a better SV mapping in fewer evaluations?

- **Non-polynomial scalar functions**: The SVD framework lets us apply ANY function. What about:
  - Piecewise functions (different mapping for different SV ranges)?
  - Log-domain mapping: log(σ) → linear transform → exp?
  - Functions from robust statistics (Huber, Tukey, etc.)?

### 3. Adaptive/Dynamic SV Mapping
Should the SV mapping change during training?

- **Curriculum-like scheduling**: More curvature preservation (less equalization) early in training, more equalization late?
- **Per-layer adaptation**: Different layers may have different spectral characteristics. Should each layer get its own mapping?
- **Condition-number-based adaptation**: High condition number → more equalization? Or the opposite?
- **What changes about gradient spectra during training?** Do the SVs of the gradient momentum become more or less spread as training progresses?

### 4. Exploiting SVD Structure Beyond Singular Values
We've focused on SV mapping. But SVD also gives us singular VECTORS (U, Vh). Can we use these?

- **Temporal consistency of singular vectors**: Do the singular vectors of consecutive gradients align? If so, can we do EMA of singular vectors (not just SVs)?
- **Cross-gradient subspace tracking**: Track the principal subspace of gradients over time. This is related to GROUSE and incremental SVD.
- **Alignment between gradient and weight singular vectors**: If they align, it means the gradient is "compatible" with the weight structure. If they don't, something interesting is happening.

### 5. Theoretical Foundations
- **Why does SV equalization help optimization at all?** Is there a theorem relating update conditioning to convergence rate for neural networks specifically?
- **Why is 0.877 better than 1.0?** We know empirically that NS's ~0.877 beats SVD's exact 1.0. Is there a theoretical explanation? Connection to learning rate scaling? Implicit regularization theory?
- **Matrix optimization theory**: For min f(W) where W is a matrix, what does optimization theory say about the optimal update direction? Steepest descent in what norm?

### 6. Novel Directions Not Yet Explored
- **Randomized SVD**: At scale, full SVD is expensive. Randomized SVD (e.g., Halko-Martinsson-Tropp) gives approximate top-k SVD in O(mnk) time. Can we use approximate SVD + our SV mapping?
- **Incremental SVD update**: Instead of computing full SVD each step, incrementally update the SVD from the previous step. Much cheaper but approximate.
- **Spectral gradient descent variants**: Methods from numerical linear algebra for optimizing matrix-valued objectives.
- **What do the NS₂ and NS₅ intermediates look like in the SVD basis?** Can we characterize EXACTLY what curvature information NS₂ retains that NS₅ destroys?

## Output Format
Same as last time:
1. **Core mathematical idea** (equations)
2. **Why it might improve our current best** (specific mechanism)
3. **How to implement** in our SVD framework (f(σ) = ???)
4. **Key papers/references**
5. **Computational cost**

Prioritize:
- Ideas that work WITHIN our SVD+scalar-function framework (easy to test)
- Mathematically principled approaches (not heuristic)
- Ideas that could meaningfully improve on our current -0.017 vs Muon
- Theory that explains WHY our current approach works (so we can do better)

Do NOT suggest:
- Anything we've already tried and failed (see negative results above)
- Simple modifications to NS (different step count, different coefficients without theory)
- Shampoo/SOAP/Adam variants
- Approaches that require changing the training setup (LR schedule, batch size, etc.)
