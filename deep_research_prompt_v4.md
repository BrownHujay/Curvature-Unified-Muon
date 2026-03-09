# Deep Research Prompt V4: Novel Mathematical Foundations for Optimization

## Context

We've spent 13 series and 75+ experiments refining NS iterate blending for Muon. We've hit the ceiling (~-0.018 vs Muon). Now we want to build a genuinely novel optimizer from mathematical foundations that have NOT been applied to neural network optimization.

**CRITICAL CONSTRAINT:** Do NOT search for existing optimizers or optimizer papers. If a method already exists as an optimizer, it's either already worse than Muon or would already be SOTA. We need **mathematical frameworks from other fields** (physics, dynamical systems, differential geometry, algebraic topology, harmonic analysis, information geometry, optimal transport, etc.) that could be *converted into* an optimization algorithm.

We want equations and theorems, not code. The translation from math → optimizer is our job.

## What We Know Works (Don't Reinvent)

1. **Newton-Schulz orthogonalization** — iterative polar decomposition via p(σ) = aσ + bσ³ + cσ⁵. Muon's core: apply to gradient, get near-orthogonal update. SVs → ~0.88.
2. **Oscillation cancellation via multi-iterate blending** — NS iterates oscillate with period-2 dynamics. Blending intermediate iterates cancels oscillation while preserving equalization.
3. **Nesterov momentum + cosine LR decay** — standard deep learning recipe.
4. **Separate parameter groups** — Muon for 2D hidden weights, AdamW for embeddings/biases/norms.

## What We Know Fails (Don't Suggest)

- Exact polar decomposition (SVs → 1.0 is worse than 0.88)
- Any monotone SV mapping (can't equalize as well as NS's non-monotone polynomial)
- Pre-NS gradient modifications (NS is too robust, washes them out)
- Element-wise or per-SV post-processing (noise dominates signal)
- Smart/adaptive weighting of corrections (simple averaging always wins)
- Warm-starting NS from previous iterate
- MARS (variance reduction catastrophic with NS)
- PSGD Kron (Lie group learning diverges without careful tuning)
- Optimal SV shrinkage/thresholding

## The Design Space

A neural network optimizer takes a gradient G ∈ ℝ^{m×n} and produces an update Δ ∈ ℝ^{m×n}. For 2D weight matrices, the best known approach is:

1. Build momentum: M_t = β·M_{t-1} + (1-β)·G_t
2. Nesterov lookahead: U = G_t + β·M_t
3. Transform: Δ = f(U) where f should:
   - Equalize singular values (so all gradient directions contribute equally)
   - Be fast (matrix multiplications, not SVD)
   - Preserve gradient direction information (U and V in SVD sense)
   - Target sub-unity SV magnitude (~0.88, not 1.0 — implicit regularization)

Currently f = Newton-Schulz⁵ (5 iterations of polynomial). Can we find a fundamentally different f?

## What We're Looking For

### Category 1: Matrix Transformations from Pure Math

**Mathematical operations on matrices that equalize singular values while preserving singular vectors, but through completely different mechanisms than polynomial iteration.**

Specific questions:
- Are there results in **random matrix theory** about transformations that push singular value distributions toward a target? (Not the Marchenko-Pastur law itself, but operations that create it or move toward it)
- What does **free probability theory** say about how matrix operations affect singular value distributions? The R-transform, S-transform, etc. — can any of these be inverted to design a desired SV transformation?
- In **symplectic geometry**, are there canonical transformations on matrices that have SV-equalizing properties? Symplectic integrators preserve phase space volume — is there an analog that preserves some matrix structure while equalizing?
- **Lie group theory**: The orthogonal group O(n) sits inside GL(n). What are the known projections/retractions from GL(n) → O(n) besides polar decomposition? Are there projections that DON'T converge to SVs=1.0?
- **Tropical geometry / valuations**: In tropical algebra, the "eigenvalues" of a matrix are determined by optimal assignment problems. Are there tropical analogs of singular value decomposition that could define a different kind of equalization?

### Category 2: Dynamical Systems for Iterative Maps

**Our best result came from understanding NS as a dynamical system (Lyapunov exponents, bifurcation theory, edge of chaos). What other dynamical systems frameworks could generate useful iterative maps on matrices?**

- **Hamiltonian mechanics on matrix manifolds**: If we treat the gradient matrix as position and the update as velocity, what does Hamiltonian flow on the Stiefel manifold look like? Are there integrable Hamiltonian systems whose trajectories naturally equalize singular values?
- **Reaction-diffusion systems**: Pattern formation in PDEs (Turing patterns, etc.) involves competition between diffusion (spreading/equalizing) and reaction (amplification). Can this be discretized into a matrix iteration that balances equalization with information preservation?
- **Renormalization group flow**: In statistical physics, RG flow transforms a system across scales while preserving universal properties. Is there an RG-like flow on matrix spectra that equalizes while preserving "relevant" directions?
- **Integrable systems / Lax pairs**: The Toda lattice and other integrable systems have matrix representations where eigenvalues are conserved while eigenvectors evolve. Can we design a system where singular VALUES equalize while singular VECTORS are preserved?
- **Gradient flow on matrix norms**: What does the gradient flow of ||UΣV^T - target||_F look like when target has equal singular values? Is this flow efficiently computable?

### Category 3: Information-Theoretic Transformations

**Instead of thinking geometrically (SVs), think about information content of the gradient.**

- **Rate-distortion theory**: Given a gradient G with some information content, what is the optimal lossy compression of G into a matrix with constrained singular value spread? What does the rate-distortion function look like for this problem?
- **Maximum entropy methods**: What is the maximum entropy matrix with prescribed singular vector structure and bounded SV ratio? Is there an efficient projection onto this set?
- **Optimal transport on matrix spectra**: View the gradient's SV distribution as a source measure and the target (equalized) as a target measure. What is the Wasserstein-optimal map? Is it more efficient than NS?
- **Fisher information geometry**: The space of matrices has a natural information-geometric structure. What are the geodesics? Does the natural gradient on this manifold have SV-equalizing properties?

### Category 4: Algebraic/Combinatorial Structures

**Completely different mathematical objects that might encode optimization dynamics.**

- **Matroid theory**: Matroids generalize linear independence. The greedy algorithm on matroids is optimal. Is there a matroid structure on gradient subspaces that would define an optimal update selection?
- **Spectral graph theory on computation graphs**: The neural network defines a graph. Each weight matrix's gradient has spectral structure related to the graph Laplacian. Are there results connecting graph spectra to optimal preconditioning?
- **Algebraic K-theory / characteristic classes**: These are invariants of vector bundles. As training progresses, the weight matrices trace a path on a Grassmannian. Do topological invariants of this path predict where preconditioning helps most?
- **Representation theory of GL(n)**: The gradient G transforms under left/right multiplication by GL(m) × GL(n). What are the irreducible representations? Does decomposing the gradient into irreps suggest a natural preconditioning?

### Category 5: Physics-Inspired Approaches

**Physical systems that naturally equalize or distribute energy.**

- **Thermalization in quantum systems**: Quantum systems naturally thermalize — energy distributes across modes. The Eigenstate Thermalization Hypothesis describes how. Is there an analog for singular values? A "thermalization operator" that drives SVs toward equilibrium?
- **Boltzmann machines / statistical mechanics of matrices**: The Gibbs distribution exp(-βH(Σ)) on singular values, where H penalizes SV spread — what does sampling from this look like? Is there an efficient MCMC step?
- **Superfluidity / Bose-Einstein condensation of SVs**: In BEC, particles "condense" into the lowest energy state. Is there a matrix operation that "condenses" singular values toward a target?
- **Gauge theory**: Gauge transformations change the representation without changing the physics. Left/right multiplication of the gradient by orthogonal matrices is a gauge transformation. What gauge-invariant quantities determine the optimal update?

## Constraints on Answers

1. **Must be computable in O(mn) or O(mn·min(m,n)) time** — we can't afford O(n³) operations like SVD every step. Matrix multiplications and element-wise operations are the speed path.
2. **Must work on rectangular matrices** — gradients are m×n where m ≠ n generally.
3. **Must preserve singular vector structure** — the "direction" of the gradient matters. We only want to change the magnitudes (singular values).
4. **The target is NOT SVs = 1.0** — we've proven that sub-unity targets (~0.88) with some spread work better than exact equalization.
5. **Iterative methods preferred** — NS works because it's 5 cheap matrix multiplications. A new method that's also iterative matrix multiplications would slot right in.

## What a Good Answer Looks Like

"In [field X], [theorem Y] establishes that the operation Z(A) = [formula] applied iteratively converges to [property P]. Specifically, if A = UΣV^T, then Z^k(A) → U·f(Σ)·V^T where f has [equalization property]. The per-iteration cost is [complexity]. The convergence rate is [rate]. The key reference is [paper]."

Or: "The mathematical framework of [X] provides a family of matrix transformations parameterized by [θ] that trade off [property A] against [property B]. No one has applied this to neural network optimization. The relevant equations are [equations]."

We want EQUATIONS and THEOREMS with citations. Not hand-waving. Not existing optimizers rebranded. Pure math that we can engineer into something new.

## Anti-Patterns (DO NOT SUGGEST)

- "Use Adam with [modification]" — NO. We know all the Adam variants.
- "Use natural gradient / Fisher information matrix" — NO. K-FAC and friends exist, they're slower than Muon.
- "Use Shampoo / SOAP / any existing preconditioned optimizer" — NO. These exist and are known.
- "Apply random matrix theory to ANALYZE gradients" — NO. We want to TRANSFORM gradients, not analyze them.
- "Use spectral normalization" — NO. Already well-known.
- "Use learning rate adaptation per singular value" — NO. This is just a fancier Adam.
- Anything that requires SVD per step — NO. Too slow.
- Anything that converges to exact orthogonal/unitary matrix — NO. SVs=1.0 is worse.
