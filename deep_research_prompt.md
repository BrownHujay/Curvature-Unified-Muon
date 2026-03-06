# Deep Research: Novel Optimizer Directions Beyond Muon/NS

## Context

We're building a novel optimizer for training neural networks (specifically transformers). The current state-of-the-art for hidden layer optimization is **Muon**, which uses Newton-Schulz (NS) iteration to compute an approximate polar factor of the gradient momentum, then uses that as the update direction.

We've run 20+ experiments trying to beat Muon. Here's what we know:

### What Muon Does
- Computes momentum: `m = β*m + (1-β)*g`
- Nesterov: `u = g + β*m`
- Newton-Schulz iteration (5 steps) with polynomial `p(σ) = 3.4445σ - 4.775σ³ + 2.0315σ⁵`
- This approximately computes the polar factor of u (closest orthogonal matrix)
- Singular values converge to ~0.877 (NOT 1.0 — the fixed point is unstable, SVs oscillate)
- Update: `W -= lr * NS(u) * sqrt(max(1, m/n))`

### What We've Proven Empirically (20+ experiments)
1. **Exact SVD polar factor (SVs=1.0) is WORSE than NS (SVs≈0.877)** by 0.012 val_loss. The sub-unity convergence provides implicit regularization.
2. **Modifying NS input doesn't help.** NS is so aggressive at equalizing SVs that the input's spectral structure doesn't matter. We tried: directional momentum (normalize before EMA), gradient difference momentum, orthogonal feedback. All tie Muon.
3. **Modifying NS output doesn't help.** Post-NS element-wise reweighting (Adam-style second moment grafting), post-NS row/col scaling — all add noise.
4. **Replacing NS with something else fails badly.** Cayley retraction (+0.32 worse), spectral outlier dampening (+0.83 worse), weight-geometry reweighting (+0.81 worse). Nothing partial (top-k SV dampening) comes close to full SV equalization.
5. **The ONLY improvement found: intercepting NS mid-iteration.** Saving the NS state at step 2 and blending 15% of it into the final output gives -0.011 improvement. This preserves some curvature info that full NS destroys.
6. **Pre-NS modifications destroy direction.** NS locks in whatever direction it receives. Preconditioning before NS rotates the gradient 28° and NS amplifies the damage.
7. **Weight decay kills Muon-family optimizers.** NS-orthogonalized updates have tiny magnitude; even wd=0.01 dominates 12x.

### The Core Challenge
NS equalization is extremely effective — so effective that nothing we put before or after it matters. But it's also destructive: it throws away ALL singular value information. The only successful approach partially recovers this lost information from the NS intermediate.

## What I Need Researched

### 1. Fundamentally Different Approaches to Matrix-Aware Optimization
NOT modifications to Muon. I need completely different mathematical frameworks for computing update directions for weight matrices in neural networks. Specifically:

- **What mathematical objects/operations besides the polar factor are useful for choosing update directions for matrices?** The polar factor (closest orthogonal matrix) is one choice. What are others? What does the optimization theory literature say about optimal update directions for matrix-valued parameters?

- **Riemannian optimization on matrix manifolds** — NOT the Stiefel manifold (we tried Cayley retraction, it failed). What about the Grassmannian? The manifold of fixed-rank matrices? The space of matrices with bounded spectral norm? What retractions/vector transports are cheap?

- **Spectral optimization methods** — algorithms that work directly in the spectral domain (SVD space) rather than element space. Are there optimization methods that maintain and update a spectral decomposition incrementally rather than computing it fresh each step?

### 2. Why Does SV Equalization Help So Much?
This is the key mystery. NS maps all singular values to ~0.877 and this massively helps optimization. Why?

- **Connection to natural gradient / Fisher information** — is the polar factor related to the natural gradient? Does SV equalization approximate preconditioning by the Fisher information matrix?

- **Connection to condition number** — does SV equalization help because it reduces the effective condition number of the update? If so, are there other ways to achieve low condition number that preserve more information?

- **Connection to implicit regularization** — NS's 0.877 fixed point is better than exact 1.0 (SVD). This sub-unity scaling acts as implicit regularization. What does the theory say about optimal regularization strength during training? Does the optimal strength change over training?

- **Connection to steepest descent in non-Euclidean metrics** — SV equalization can be interpreted as steepest descent in the spectral norm. What other matrix norms give useful descent directions? Nuclear norm? Schatten p-norms?

### 3. Information-Theoretic Perspective
- **What information does NS destroy, and how much of it is actually useful?** NS eliminates all singular value structure. Our only improvement (v5) recovers ~15% of the step-2 intermediate's SV structure. Is there a principled way to determine WHICH spectral information to keep and which to discard?

- **Rate-distortion theory for gradient compression** — can we frame NS as a lossy compression of the gradient? If so, what's the optimal compression that preserves the most optimization-relevant information?

### 4. Novel Mathematical Ingredients
Look for mathematical tools/operations that could be useful for optimization but haven't been applied:

- **Matrix means** (geometric mean, log-Euclidean mean) — instead of element-wise EMA for momentum, use matrix geometric mean of consecutive gradients?

- **Procrustes problems** — instead of "closest orthogonal to G" (polar factor), solve "closest orthogonal to G that's also close to the PREVIOUS update"? This is the orthogonal Procrustes problem with a regularization term.

- **Optimal transport on matrices** — transport the gradient's spectral distribution toward a target distribution rather than hard-equalizing?

- **Polynomial filtering on singular values** — instead of NS's specific polynomial, design an optimal polynomial filter that selectively compresses/expands different parts of the SV spectrum?

### 5. What Do Other Fields Do?
Matrix optimization appears in many fields beyond deep learning:

- **Control theory** — how do they update gain matrices? Lyapunov-based methods?
- **Signal processing** — adaptive beamforming, MIMO systems update weight matrices. What algorithms do they use?
- **Numerical linear algebra** — iterative methods for matrix equations. Is there something better than Newton-Schulz for our specific use case?
- **Quantum computing** — unitary optimization, variational quantum circuits. How do they optimize over matrix-valued parameters?

### 6. Scaling Considerations
Our current benchmark is tiny (1.2M params, 128-dim). Muon scales to trillions (Kimi K2). What changes at scale?

- Do the relative advantages shift? Maybe something that fails at small scale works at large scale (or vice versa)?
- Are there approaches that are theoretically better but only kick in at larger dimensions?

## Output Format

For each promising direction you find, provide:
1. **The core mathematical idea** (equations, not just words)
2. **Why it might beat NS/Muon** (specific mechanism)
3. **Computational cost** compared to NS (5 steps of `X = aX + (bA + cA²)X` where `A = XX.T`)
4. **Key papers/references** to read
5. **How to implement it** as a PyTorch optimizer (pseudocode)

Prioritize ideas that are:
- Mathematically principled (not heuristic)
- Computationally feasible (not more than 2x Muon's cost)
- Genuinely different from Muon (not "Muon + small modification")
- Likely to work at small scale (128-dim matrices) since that's where we test first

Do NOT suggest:
- Shampoo/SOAP (Kronecker preconditioning) — well-known, not novel
- Adam variants — we already use Adam for non-matrix params
- Simple modifications to NS (different step count, different coefficients, blending) — we've exhausted these
- MuonClip — that's for stability at scale, not convergence improvement
