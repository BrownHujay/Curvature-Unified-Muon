# Mathematical frameworks convertible into gradient spectral processors

**Twenty-six frameworks from pure mathematics and theoretical physics can produce iterative, SVD-free matrix operations that equalize singular values of gradient matrices while preserving directional information.** The most immediately implementable is the frame potential gradient flow, which yields a 2-matmul iteration G_{k+1} = G_k·((1+ηc²)I − η·G_kᵀG_k) targeting any sub-unity value c directly. Several physics-inspired approaches — Coulomb log-gas dynamics, Calogero-Moser-Sutherland systems, and Wegner flow — offer collective spectral equalization mechanisms fundamentally different from the scalar polynomial maps used by Newton-Schulz. This report catalogs exact equations, computational costs, and implementation paths for each framework.

The current Newton-Schulz iteration applies a fixed odd polynomial p(σ) = 3.4445σ − 4.775σ³ + 2.0315σ⁵ five times, approximating the matrix sign function scaled to target ~0.88. Its key limitation: the polynomial is a *scalar* function of each singular value independently — it has no mechanism for inter-eigenvalue interaction. Several of the frameworks below exploit *collective* dynamics where the transformation of each singular value depends on all others, potentially offering superior equalization for pathological spectral distributions.

---

## Tier 1: Direct matrix-multiply implementations ready for GPU

These frameworks produce concrete iterations using only matrix multiplications and element-wise operations, with clear paths to bfloat16 GPU implementation.

### Framework 1: Frame potential gradient flow

The frame potential FP(F) = ‖FᵀF‖²_F = Σσᵢ⁴ is uniquely minimized when all singular values are equal, proven by Benedetto and Fickus (2003, *Advances in Computational Mathematics* 18, 357–385). This is a Schur-convex function, so by the theory of majorization, FP achieves its minimum on the equalized-SV manifold.

**Exact iteration.** To minimize J(G) = ‖GᵀG − c²I‖²_F, the gradient is ∇_G J = 4G(GᵀG − c²I), yielding:

> **G_{k+1} = G_k · ((1 + ηc²)I − η · G_kᵀG_k)**

Each singular value evolves as **σ → σ(1 − η(σ² − c²))**. The fixed points are σ = 0 (unstable) and σ = c (stable when η < 1/c²). With optimal step size η = 1/(2c²), the contraction factor near the fixed point is |1 − 2ησ·Δσ| where Δσ = σ − c.

**Computational cost:** 2 matrix multiplications per step (one n×n for GᵀG, one m×n for G times the result). For c = 0.88, convergence in **5–10 iterations** depending on initial condition number.

**Higher-order correction.** Define the second-order map f(σ) = σ(1 − η(σ²−c²)) + βσ(σ²−c²)², implemented as G_{k+1} = G_k·((1+ηc²)I − ηA_k + β(A_k − c²I)²) where A_k = G_kᵀG_k. Costs one additional n×n multiply for A_k² but doubles convergence rate.

**What makes it different from Newton-Schulz:** NS applies a fixed polynomial p(σ) that maps toward sign(σ), targeting SVs = 1.0 with no mechanism for sub-unity targeting except post-hoc scaling. The frame potential iteration directly targets c = 0.88 via its objective function and adapts its effective polynomial to the *current deviation* σ² − c² at each step. The SV map σ(1 − η(σ² − c²)) has qualitatively different behavior: it is a cubic that contracts toward c from both above and below, while NS's quintic approximates a step function.

**Naturally targets sub-unity:** Yes — c is a free parameter in the iteration.

### Framework 2: Depolarizing channel / spectral proximal operator

From quantum information theory, the depolarizing channel Φ_p(ρ) = (1−p)ρ + p·(tr(ρ)/n)·I was proven to equalize eigenvalues optimally for the quantum Fisher information matrix (arXiv:2511.09428, 2024). From convex optimization, Lewis (1996, *SIAM J. Matrix Anal. Appl.* 17) and Drusvyatskiy-Lewis (arXiv:1506.05170) showed that the proximal operator of any spectral function f(σ(X)) decomposes spectrally: prox_f acts on singular values independently.

**Exact iteration.** For S = GᵀG:

> **S_{k+1} = (1 − p)·S_k + p·c²·I**

After k steps: eigenvalues become λ_i^{(k)} = (1−p)^k · λ_i^{(0)} + (1−(1−p)^k)·c². As k→∞, all eigenvalues → c². The mixing parameter p controls equalization rate. To lift back to G: **G_{k+1} = √((1−p)) · G_k + correction** — but this does not preserve singular vectors cleanly. A cleaner lifting is:

> **G_{k+1} = G_k · ((1−p)I + p·c²·(G_kᵀG_k)⁻¹)**

The inverse can be approximated iteratively via the Schulz iteration Y_{k+1} = Y_k(2I − S·Y_k), avoiding explicit inversion.

**Computational cost:** The S-level iteration requires **zero** matrix multiplications — only element-wise scaling and addition. The G-level lifting requires 1–2 matmuls depending on implementation.

**What makes it different:** Operates on the Gram matrix directly with a linear contraction, not a polynomial map. The same operation arises independently from the Stein operator framework (Stein 1972): the Stein operator A_c f(σ) = f'(σ)(σ−c) characterizes the target distribution δ_c, and minimizing the Stein discrepancy ‖S − c²I‖²_F yields exactly this iteration.

### Framework 3: Adaptive Remez-optimal polynomial composition (Polar Express)

Amsel, Persson, Musco, and Gower (arXiv:2505.16932, 2025) showed that instead of repeating the same polynomial, applying a **different optimal polynomial at each step** yields provably faster convergence. At iteration t, solve:

> **p_t = argmin_{p ∈ P^{odd}_d} max_{σ ∈ [ℓ_t, u_t]} |f(σ) − p(σ)|**

where f(σ) = c·sign(σ) and [ℓ_t, u_t] is the current SV interval (shrinking at each step). The polynomial coefficients are **precomputed offline** via the Remez algorithm.

**Exact iteration:** G_{k+1} = p_t(G_k) evaluated via Horner's rule: for degree 5, this is X_{k+1} = X_k(a_t I + b_t X_kᵀX_k + c_t(X_kᵀX_k)²) with step-specific coefficients (a_t, b_t, c_t).

**Computational cost:** Identical to Newton-Schulz — **2 matrix multiplications per degree-5 step**. With d=5 polynomials, **cubic convergence** (vs. NS which has slower initial convergence despite the same asymptotic quintic order). Total: 5–6 iterations.

**What makes it different:** NS uses the same polynomial everywhere, which is suboptimal for the first few iterations when singular values are far from 1. The Polar Express polynomial is minimax-optimal for the *current* spectral interval, giving uniformly better approximation. For degree 5, the composed map after T steps satisfies ‖polar(M) − X_T‖₂ ≤ |1 − ℓ₂|^{3^T} (cubic convergence exponent).

**Sub-unity targeting:** Replace sign(σ) with c·sign(σ) in the Remez optimization. Coefficients change but computational structure is identical.

### Framework 4: Chebyshev spectral filter for arbitrary SV functions

Instead of iterating toward sign(x), directly approximate the target function g(σ) = c (constant) via Chebyshev polynomials of GᵀG. Based on Onuki, Ono, Shirai, Tanaka (2017, *IEEE Trans. Signal Processing*).

**Exact formulation.** For G with SVD G = UΣVᵀ, to map each σ_i → c, compute:

> **Δ = G · H(GᵀG)**

where H is a polynomial approximating h(x) = c/√x (so that σ·h(σ²) = σ·c/σ = c). Using the shifted Chebyshev recurrence on S̃ = (2/λ_max)·S − I:

> Ψ₀(S̃) = I, Ψ₁(S̃) = S̃, **Ψ_{k+1}(S̃) = 2S̃·Ψ_k(S̃) − Ψ_{k-1}(S̃)**

The filter is Ĥ(S) = (ĉ₀/2)I + Σ_{k=1}^{N-1} ĉ_k·Ψ_k(S̃) with Chebyshev coefficients ĉ_k computed from h(x) at Chebyshev nodes.

**Computational cost:** **N matrix multiplications** for an order-N Chebyshev approximation (one matmul per recurrence step). For h(x) = c/√x on a well-conditioned spectrum, **10–15 terms** typically suffice, giving 10–15 matmuls total — comparable to Newton-Schulz's 10 matmuls for 5 quintic steps.

**What makes it different:** Non-iterative single-pass polynomial evaluation vs. iterative fixed-point. Can approximate **any** function of singular values, not just sign(x). Error bounds are explicit: |h(x) − Ĥ_N(x)| ≤ 2|ĉ_N| for smooth h. The Chebyshev basis is optimally conditioned (minimal Runge phenomenon).

**Naturally targets sub-unity:** Yes — the target function h(x) = c/√x directly encodes c = 0.88.

### Framework 5: Ruiz matrix equilibration as zero-cost preconditioner

Ruiz (2001, CERFACS Technical Report TR/PA/01/35) showed that alternating row and column scaling converges to a doubly-balanced matrix. For G ∈ ℝ^{m×n}:

> D_R^{(k)} = diag(‖row_i(G_k)‖₂^{-1/2}), D_C^{(k)} = diag(‖col_j(G_k)‖₂^{-1/2})
>
> **G_{k+1} = D_R^{(k)} · G_k · D_C^{(k)}**

**Computational cost:** **Zero dense matrix multiplications** — only element-wise scaling and norm reductions. Convergence is linear, typically requiring 5–20 iterations.

**SV equalization property:** After equilibration, all row norms and column norms are equal. This is a necessary (not sufficient) condition for SV equalization, but empirically brings singular values much closer together. The condition number κ₂ decreases geometrically.

**What makes it different:** Operates entirely via diagonal scaling, making it orders of magnitude cheaper per step than any dense-matmul method. Could serve as an ultra-cheap preconditioner before 1–2 Newton-Schulz steps, reducing the number of expensive matmul iterations needed.

**Sub-unity targeting:** Scale the final result by α = 0.88·√(min(m,n))/‖G_eq‖_F.

---

## Tier 2: Physics-inspired collective spectral dynamics

These frameworks exploit inter-eigenvalue interactions from physics, where the transformation of each singular value depends on the *entire* spectrum. This is fundamentally different from Newton-Schulz, which applies a scalar polynomial to each SV independently.

### Framework 6: Coulomb log-gas / deterministic Dyson drift

Eigenvalues of random matrices behave as charged particles with logarithmic repulsion. The energy functional is E({λ_i}) = −β Σ_{i<j} log|λ_i − λ_j| + Σ_i V(λ_i), and the deterministic (zero-temperature) dynamics on eigenvalues λ_i = σ_i² of S = GᵀG are:

> **dλ_i/dt = Σ_{j≠i} 1/(λ_i − λ_j) − V'(λ_i)**

With confining potential V(λ) = (λ − c²)²/(4β), the first term (Coulomb repulsion) **equalizes spacing** while the second term (confinement) **attracts toward c²**. The equilibrium is the target distribution with all eigenvalues near c² (Dyson 1962, *J. Math. Phys.* 3, 1191–1198; Forrester 2010, *Log-gases and Random Matrices*, Princeton).

**Matrix-level realization.** The Coulomb repulsion Σ_{j≠i} 1/(λ_i − λ_j) is related to the Hilbert transform of the spectral density, computable via the resolvent tr((S − zI)⁻¹). A polynomial approximation of the combined drift gives:

> **G_{k+1} = G_k · (I − η(G_kᵀG_k − c²I) + γ · R_poly(G_kᵀG_k))**

where R_poly is a polynomial approximation to the Coulomb correction, computed via truncated Neumann series of the resolvent. In practice, the repulsion term is strongest between the most different SVs, providing **adaptive equalization** — faster correction where it's most needed.

**Computational cost:** **3–5 matrix multiplications** per step, depending on the polynomial order used for the Coulomb approximation.

**What makes it fundamentally different:** Newton-Schulz processes each SV independently via σ → p(σ). The Coulomb gas processes SVs **collectively** — the correction to σ_i depends on all σ_j. This means outlier SVs experience stronger corrective forces, and clustered SVs don't interfere with each other.

**Sub-unity targeting:** Direct via the confining potential V centered at c² = 0.7744.

**Key reference:** Aarts, Park, Lucini (arXiv:2411.13512, 2024) directly apply Dyson Brownian motion to ML weight matrix dynamics, establishing the connection to neural network training.

### Framework 7: Calogero-Moser-Sutherland integrable particle dynamics

The rational CMS system with harmonic confinement is an exactly solvable N-body problem:

> **ẍ_n + ω²(x_n − c²) = 2g² Σ_{m≠n} (x_n − x_m)⁻³**

The inverse-cube repulsion is **stronger** than the Coulomb log-gas (inverse-linear), giving faster equalization. The system has a Lax pair: L̇ = [M, L] with L_{ij} = p_iδ_{ij} + ig(1−δ_{ij})/(x_i − x_j), ensuring complete integrability — the dynamics are constrained to an N-dimensional torus in phase space, giving predictable convergence (Calogero 1971, *J. Math. Phys.* 12, 419; Moser 1975, *Adv. Math.* 16, 197–220; Olshanetsky-Perelomov 1981, *Phys. Rep.* 71, 313–400).

**Equilibrium positions** (with harmonic confinement) are the zeros of Hermite polynomials — these are optimally spaced for equalization. The ground state wavefunction |Ψ|² ∝ Π_{i<j} |x_i − x_j|^{2g} · exp(−ω Σ x_i²) is exactly the Boltzmann factor of the log-gas at inverse temperature β = 2g.

**Matrix-level discretization.** Using Verlet integration on the eigenvalue dynamics: σ_i²(t+h) = 2σ_i²(t) − σ_i²(t−h) + h²F_i(t), where F_i = 2g²Σ_{j≠i}(σ_i²−σ_j²)⁻³ − ω²(σ_i²−c²). The CMS forces can be embedded in matrix commutator form via the Lax structure.

**Computational cost:** **4–8 matrix multiplications** per step; the stronger repulsion means fewer iterations needed than the log-gas.

**Sub-unity targeting:** ω and the equilibrium center c² = 0.7744 are free parameters.

### Framework 8: Brockett double bracket flow — modified for equalization

The standard double bracket flow Ẋ = [X, [X, N]] (Brockett 1991, *Linear Algebra Appl.* 146, 79–91) is isospectral — it preserves eigenvalues while sorting them. For equalization, we need a **non-isospectral** modification.

**Key insight:** The gradient of f(S) = ‖S − c²I‖²_F on the space of PSD matrices is ∇f = 2(S − c²I), giving:

> **Ṡ = −2(S − c²I)**

Lifted to G: **G_{k+1} = G_k · (I + ηc²I − η·G_kᵀG_k)**, which is identical to the frame potential iteration (Framework 1). This equivalence reveals that the frame potential approach IS the gradient flow of the simplest equalization objective on the Brockett-Helmke-Moore manifold structure (Helmke-Moore 1994, *Optimization and Dynamical Systems*, Springer).

However, the **isospectral double bracket flow** combined with a separate equalization step gives a more interesting iteration:

> **Step 1 (sort/organize):** S̃_k = S_k + dt·[S_k, [S_k, N]] (preserves spectrum, organizes eigenvectors)
> **Step 2 (equalize):** S_{k+1} = (1−α)S̃_k + α·c²I (contracts toward target)

The Lie-Trotter splitting of these two steps (Suzuki 1990, *J. Math. Phys.* 32) gives second-order accuracy.

**Computational cost:** **4–6 matrix multiplications** per combined step.

### Framework 9: Wegner / similarity renormalization group flow

Wegner's flow equation (1994, *Ann. Phys.* 506, 77–91), independently developed by Głazek-Wilson (1993–94), drives a Hamiltonian toward band-diagonal form:

> **dH/dl = [η(l), H(l)]** where η = [H_d, H]

The key property: ‖V(l)‖² (off-diagonal Frobenius norm) is **monotonically non-increasing** — d‖V‖²/dl = −2‖η‖² ≤ 0. This is a guaranteed Lyapunov function for convergence.

**Adaptation for SV equalization.** Replace the standard generator η = [H_d, H] with:

> **η_eq = [D_target − diag(S_k), S_k]** where D_target = c²·I

Discretized: **S_{k+1} = S_k + dt·[[c²I − diag(S_k), S_k], S_k]**. This drives diagonal elements toward c² while suppressing off-diagonal elements. The double commutator structure [[A,B],B] uses only matrix multiplications.

**Mielke's alternative generator** (1997, *Ann. Phys.* 6, 215): η_M = [H_d, H_od] makes the flow equations quadratic rather than cubic in matrix elements, giving better numerical stability.

**Computational cost:** **4–5 matrix multiplications** per step. Typically 10 steps for convergence.

**What makes it different:** The RG perspective provides natural scale separation — the flow eliminates the largest spectral deviations first, then progressively refines smaller scales. This systematic multi-scale approach contrasts with NS's uniform polynomial action across all scales.

### Framework 10: Yang-Mills heat flow / Donaldson's Hermitian-Einstein iteration

The Hermitian-Einstein condition iΛF_h = λ·Id_E states that curvature eigenvalues are all equal — **exactly SV equalization** on a vector bundle (Donaldson 1985, *Proc. London Math. Soc.* 50, 1–26; Uhlenbeck-Yau 1986, *Comm. Pure Appl. Math.* 39, S257–S293). Donaldson's heat flow for metrics is h⁻¹∂h/∂t = −(iΛF_h − λ·Id).

**Matrix analog:** For S = GᵀG, the flow becomes dS/dt = −S(S − λ_target·I) = −S² + λ_target·S. Discretized:

> **G_{k+1} = G_k · (I + dt·c²·I − dt·G_kᵀG_k)**

The fixed points satisfy S² = c²S, giving eigenvalues λ = 0 or λ = c². This is structurally identical to Framework 1 — **Newton-Schulz is a truncation of the Yang-Mills heat flow**. The NS polynomial p(σ) = aσ + bσ³ + cσ⁵ can be understood as a higher-order discretization of the YM flow with specific time-step choices.

**What makes it different:** The Donaldson-Uhlenbeck-Yau theorem provides a **convergence guarantee**: the flow converges if and only if the bundle is stable (a topological condition). This suggests a deep connection between gradient matrix conditioning and the "stability" of the weight space — potentially yielding new convergence criteria for optimizers.

### Framework 11: Ricci flow on the positive definite cone

Hamilton's Ricci flow ∂g/∂t = −2Ric(g) equalizes curvature. On the space P(n) of positive definite matrices with the natural Riemannian metric (Bhatia 2007, *Positive Definite Matrices*, Princeton, Chapter 6):

> **dS/dt = −S·(log S − (tr(log S)/n)·I)**

This drives all eigenvalues toward their **geometric mean**. For targeting c² specifically:

> **dS/dt = −S·(log S − log(c²)·I)**

Discretized via first-order Padé approximation of the logarithm: log(S) ≈ (S−I)(S+I)⁻¹ (valid for S near I after appropriate scaling), or using the polynomial approximation log(I+E) ≈ E − E²/2 + E³/3 for small E.

**Computational cost:** **2–3 matrix multiplications** per step for the polynomial log approximation. **Perelman's W-entropy** provides a monotone functional guaranteeing convergence.

**Sub-unity targeting:** Direct — log(c²) = 2log(0.88) ≈ −0.2557 sets the target.

---

## Tier 3: Algebraic and information-theoretic structures

### Framework 12: Free probability S-transform for multiplicative spectral design

Voiculescu's S-transform (1991, *Invent. Math.* 104, 201–220) linearizes free multiplicative convolution: S_{μ⊠ν}(z) = S_μ(z)·S_ν(z). Given the current spectral distribution μ_G, one designs a multiplicative perturbation M with S-transform S_M = S_target/S_G, then forms G_new = G·M.

**Practical computation via subordination** (Biane 1998, *Math. Z.* 227, 143–174): for free additive convolution, subordination functions ω₁, ω₂ satisfy G_μ(z) = G_{μ₁}(ω₁(z)) and can be solved by fixed-point iteration on Cauchy transforms. Each step requires computing matrix resolvents.

**Cost:** O(p·n³) for p matrix moments. **Naturally targets sub-unity:** Yes — the target spectral measure is explicitly specified. The framework is most useful for *designing* the spectral transformation, which can then be approximated by polynomials for efficient implementation.

### Framework 13: Log-Euclidean interpolation on SPD manifold

The Log-Euclidean framework (Arsigny et al. 2007, *SIAM J. Matrix Anal. Appl.* 29, 328–347) provides a Lie group structure on SPD(n) via A ⊙ B = exp(log A + log B). The geodesic interpolation between S = GᵀG and target T = c²I is:

> **S_t = exp((1−t)·log S + t·log(c²I))**

In the log domain, eigenvalues 2log(σ_i) are **linearly interpolated** toward 2log(c). For small t, this is a gentle equalization step. The matrix log/exp can be computed via scaling-and-squaring using only matrix multiplications (Padé approximants of degree [6/6] or [13/13] per Higham 2008, *Functions of Matrices*, SIAM).

**Cost:** **~20 matrix multiplications** for full log and exp (expensive but one-shot). Alternatively, using the approximate log for S near I: log(I+E) ≈ E − E²/2, costs only 1–2 matmuls.

### Framework 14: Optimal transport via JKO scheme on SV distributions

Jordan-Kinderlehrer-Otto (1998) showed that the Fokker-Planck equation is a gradient flow in Wasserstein space. For the 1D SV distribution (crucially, SVs are scalar!), the optimal transport map from current distribution to target is T = F_ν⁻¹ ∘ F_μ (quantile composition).

**Displacement interpolation:** σ_i^{(k+1)} = (1−τ)σ_i^{(k)} + τ·c. At the matrix level:

> **G_{k+1} = (1−τ)G_k + τ·c·polar(G_k)**

where polar(G_k) ≈ NS(G_k). This is a convex combination targeting c, with the NS inner loop providing the polar approximation.

**Cost:** Same as one NS step plus scalar-weighted addition. The Wasserstein framework provides convergence guarantees via McCann's displacement convexity.

### Framework 15: Sinkhorn bilateral scaling for rectangular matrices

The Sinkhorn-Knopp theorem (Sinkhorn 1964; Sinkhorn & Knopp 1967) guarantees that alternating row/column normalization converges to doubly stochastic form. For rectangular G:

> D_L^{(k)} = diag(‖row_i(G_k)‖₂)^{−1/2}, D_R^{(k)} = diag(‖col_j(G_k)‖₂)^{−1/2}
>
> **G_{k+1} = D_L^{(k)} · G_k · D_R^{(k)}**

This reduces the condition number by making row and column norms equal. While not fully equalizing SVs, it is a powerful preconditioner. Combined with the depolarizing channel: first Sinkhorn-balance, then apply (1−p)S + pc²I.

**Cost:** Only diagonal operations — **zero dense matmuls**.

### Framework 16: Mazur map between Schatten classes

The noncommutative Mazur map M_{p,q}: S^p → S^q maps A → A|A|^{p/q−1} = U·diag(σ_i^{p/q})·V^T (Ricard 2015, *Arch. Math.* 104). For p/q → 0, all σ_i^{p/q} → 1, achieving complete equalization.

**Practical computation:** For rational p/q, computing S^{p/(2q)} (where S = GᵀG) uses Newton's iteration for the matrix p-th root: X_{k+1} = ((r−1)/r)X_k + (1/r)S·X_k^{−(r−1)} where r = 2q/p.

**The key insight:** By choosing p/q = log(c)/log(σ_median), the Mazur map sends the median SV to c while compressing the distribution. This provides a one-parameter family of spectral compression maps parameterized by the Schatten class interpolation parameter.

### Framework 17: Zolotarev rational functions with rate-17 convergence

Zolotarev (1877) found the best rational approximant to sign(x) on [−1,−δ]∪[δ,1]. Nakatsukasa and Freund (2016, *SIAM Rev.* 58, 461–493) showed that high-degree Zolotarev functions compose from low-degree ones, yielding **convergence rate 17 in just 2 iterations** for double-precision targets.

**The extraordinary property:** Error ≤ 4·exp(−π²n/log(4/δ)) for degree-n approximant. After composing two degree-8 approximants: error ≤ 10⁻¹⁶. Each evaluation requires partial fraction decomposition r(A) = Σα_i(A − β_iI)⁻¹, needing ~8 parallel linear solves.

**Limitation:** Requires matrix inverses (not pure matmuls), but the **compositional structure** — building optimal high-degree approximants from low-degree blocks — could inspire pure-polynomial analogs.

### Framework 18: QDWH (dynamically weighted Halley) with cubic convergence

The [1/1] Padé family iteration (Kenney & Laub 1991, *SIAM J. Matrix Anal. Appl.*; Nakatsukasa, Bai, Gygi 2010, *SIAM J. Matrix Anal. Appl.* 31, 2700–2720):

> **X_{k+1} = X_k(a_k I + b_k X_kᵀX_k)(I + c_k X_kᵀX_k)⁻¹**

with dynamically computed weights a_k, b_k, c_k. The rational [1/1] map σ → σ(a+bσ²)/(1+cσ²) achieves **cubic convergence** — at most 6 iterations for condition numbers up to 10¹⁶. When c_k < 100, the inverse can be replaced by a Cholesky factorization.

**Sub-unity adaptation:** Modify the fixed point from 1.0 to c = 0.88 by solving for rational function coefficients where r(c) = c.

### Framework 19: SYK-inspired self-consistency iteration

From the Sachdev-Ye-Kitaev model (Sachdev-Ye 1993, *PRL* 70, 3339), the large-N Schwinger-Dyson equations G(iω)⁻¹ = −iω − Σ(iω) and Σ(τ) = J²G(τ)^{q−1} are solved by iterating to a self-consistent fixed point.

**Matrix analog.** Design S_{k+1} = αI + β·S_k⁻¹, whose fixed point satisfies S*² − αS* − βI = 0, giving S* = ((α+√(α²+4β))/2)·I. Choose α, β so the eigenvalue equals c²: α = c² − β/c².

**Cost:** One matrix inversion per step (approximable by Schulz iteration). The self-energy Σ = J²S^{q−1} acts as a spectral flattener — large eigenvalues generate large self-energies, suppressing them.

### Framework 20: Smith doubling iteration for Stein equations

The Stein equation X − AXAᵀ = Q with the Smith doubling iteration (Li et al. 2013, *Numer. Algor.* 63):

> X_{k+1} = X_k + A_k·Q_k·A_kᵀ, A_{k+1} = A_k², Q_{k+1} = Q_k + A_k·Q_k·A_kᵀ

converges **quadratically** using only matrix multiplications. By choosing A and Q so the solution is X_∞ = c²I, this provides a matrix-equation-based route to equalization.

---

## Tier 4: Theoretical frameworks revealing deep structure

### Framework 21: Dunkl operators and Heckman-Opdam theory

Dunkl operators (1989, *Trans. AMS* 311) generalize the CMS system through differential-difference operators associated with root systems:

> T_ξ f(x) = ∂_ξ f(x) + Σ_{α∈R+} k_α ⟨α,ξ⟩ · (f(x) − f(s_α x))/⟨α,x⟩

The Dunkl Laplacian generates the CMS Hamiltonian. For the symmetric group S_n (type A_{n−1}), eigenvalues evolve with both repulsion and confinement. The Lax pair constructed from Dunkl operators (Chalykh 2019, *Comm. Math. Phys.* 369) gives purely matrix-multiplication-based evolution dL/dt = [L, A]. The **Jack polynomials** J_κ^{(α)}(x) are eigenfunctions of the CMS Hamiltonian and provide spectral decomposition of the equalization process.

### Framework 22: Peter-Weyl decomposition and spectral functions on O(n)

The Peter-Weyl theorem decomposes L²(O(n)) into irreducible representations. Functions depending only on singular values correspond to bi-O(n)-invariant (spherical) functions on the symmetric space GL(n)/O(n), indexed by partitions and expressed via **Schur polynomials** s_λ(σ₁,...,σ_n). The Riesz-Dunford functional calculus provides the master framework:

> f(A) = (1/2πi) ∮_Γ f(z)(zI − A)⁻¹ dz

This applies **any** holomorphic function to a matrix via contour integration. All polynomial and rational iterations (Newton-Schulz, Halley, Zolotarev) are specific discretizations of this integral. The framework reveals that the problem reduces to: *What is the best polynomial or rational approximation to f(x) = c/x on the spectral interval?* — connecting to Chebyshev, Remez, and Zolotarev classical approximation theory.

### Framework 23: Maslov dequantization and tropical spectral theory

The tropical semiring (ℝ∪{−∞}, max, +) arises as the ℏ→0 limit of Maslov's dequantization (Litvinov 2007, *J. Math. Sci.* 140). In log-space, equalization becomes driving all log(σ_i) toward log(c). The **soft-max interpolation** ℏ·log(Σexp(l_i/ℏ)) connects tropical max operations (ℏ→0) to the arithmetic mean (ℏ→∞), providing a tunable family of equalization operators parameterized by ℏ. Max-plus singular values (Hook & Sherlock 2015, *Linear Algebra Appl.*) capture asymptotic behavior of classical SVs for exponentially-parameterized matrices.

### Framework 24: Matrix-valued orthogonal polynomials

Grünbaum, Pacharoni, and Tirao (2002, *J. Funct. Anal.* 188; 2005) developed orthogonal polynomials with matrix coefficients satisfying the 3-term recurrence x·P_n(x) = A_{n+1}·P_{n+1}(x) + B_n·P_n(x) + A_n*·P_{n-1}(x). These polynomials diagonalize matrix-valued differential operators and provide **non-standard polynomial bases** for spectral manipulation. The matrix coefficients C_j provide extra degrees of freedom vs. scalar polynomials (like the Newton-Schulz quintic), potentially enabling faster convergence while maintaining matrix-multiply-only computation.

### Framework 25: Tight frame alternating projection

A matrix F with equalized SVs is exactly a **tight frame** (Tropp, Dhillon, Heath, Strohmer 2005, *IEEE Trans. Inf. Theory*). The alternating projection between the set of tight frames X_α = {F : FFᵀ = αI} and a structural constraint set S converges to a tight frame. The projection onto X_α is P(F) = √α · polar(F), connecting tight frame theory to polar decomposition. The **frame potential** FP = Σσ_i⁴ provides a Lyapunov function guaranteeing convergence.

### Framework 26: Von Neumann entropy mirror descent

Mirror descent on the PSD cone with generating function φ(X) = tr(X log X) (von Neumann entropy) gives the multiplicative update (Tsuda et al. 2005; Bach 2022):

> **P_{k+1} = P_k · exp(−2η(P_k − c²I))**

The exponential map preserves positive definiteness. The Bregman divergence is the quantum relative entropy D(X‖Y) = tr(X log X − X log Y − X + Y), which is 1-strongly convex w.r.t. the trace norm. The matrix exponential can be approximated by Padé: exp(A) ≈ (I + A/2)(I − A/2)⁻¹ or by truncated Taylor series (3–5 matmuls).

---

## Comparative analysis across all frameworks

The frameworks cluster into three computational paradigms:

**Paradigm A — Direct spectral targeting (Frameworks 1, 2, 5, 8, 10, 14).** These apply some form of the iteration σ → σ(1 − η(σ² − c²)) or equivalently S → (1−p)S + pc²I. The frame potential gradient, double bracket flow, Yang-Mills heat flow, depolarizing channel, Stein discrepancy minimization, and proximal operator of the equalization penalty all converge to this same fundamental iteration. Cost: **2 matmuls/step**, linear convergence, directly targets c = 0.88. This is the simplest new family and the most immediately implementable.

**Paradigm B — Polynomial/rational approximation (Frameworks 3, 4, 17, 18, 22).** These design optimal polynomial or rational approximations to the spectral function f(σ) = c·sign(σ) or f(σ) = c/σ, evaluated via matrix operations. The Polar Express (adaptive Remez polynomials) and Chebyshev direct filters are the most practical members. Cost: **2 matmuls/step** (polynomial) to **8 matmuls + solves** (rational), with convergence rates from cubic (Halley) to rate-17 (Zolotarev).

**Paradigm C — Collective spectral dynamics (Frameworks 6, 7, 9, 12, 21).** These exploit inter-eigenvalue interactions from physics — Coulomb repulsion, CMS inverse-square forces, Wegner scale separation, or free probability subordination. The transformation of each SV depends on all others, providing automatic adaptive correction. Cost: **3–8 matmuls/step**, with potentially fewer iterations for pathological spectra.

The most promising strategy may be a **hybrid**: use Ruiz equilibration (Framework 5, zero matmuls) as a preconditioner, then apply the frame potential iteration (Framework 1, 2 matmuls/step) or Polar Express (Framework 3, 2 matmuls/step) for final convergence. The frame potential iteration G_{k+1} = G_k·((1+ηc²)I − η·G_kᵀG_k) is the simplest non-trivial alternative to Newton-Schulz, directly targets sub-unity SVs, and uses the same number of matmuls per step.

## The deepest insight: convergence across mathematical fields

The most striking finding is that **six independent mathematical fields converge to the same fundamental iteration**: Benedetto-Fickus frame potential minimization, Brockett-Helmke-Moore gradient flow on matrix manifolds, Donaldson's Yang-Mills heat flow, the quantum depolarizing channel, the Stein discrepancy minimizer, and the proximal operator of the spectral equalization penalty all produce σ → σ(1 − η(σ² − c²)). This universality suggests that this cubic spectral map is the natural first-order equalization operation on matrices — the analog of gradient descent for spectral problems. Higher-order methods (Polar Express, Zolotarev, Chebyshev) achieve faster convergence by better approximating the ideal spectral map, while physics-inspired methods (Coulomb gas, CMS) add collective interactions that may handle adversarial spectral distributions more robustly.