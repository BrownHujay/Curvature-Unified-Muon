# Polynomial design, temporal averaging, and oscillation dynamics for CUM optimizer V3

**The Newton-Schulz "cursed quintic" is not broken—it operates at the edge of chaos, and the research literature converges on a clear verdict: the oscillation is an implicit regularizer, not a bug.** Multiple independent theoretical frameworks—edge-of-stability dynamics, stochastic resonance, period-2 bifurcation theory, and anisotropic noise regularization—all predict that sub-unity singular value targets with mild deterministic oscillation should outperform exact orthogonalization. The Polar Express paper (Amsel et al., 2025) and CANS (Grishina et al., 2025) have already formalized optimal polynomial design via Remez/Chebyshev theory, but neither explores the researcher's key insight: that the *non-convergent* behavior itself provides value. Below is a comprehensive analysis of all seven research directions with specific mathematical tools, mechanisms, risk assessments, and implementation paths.

---

## 1. The polynomial design space is smaller and more tractable than it appears

The researcher's problem—optimizing degree-5 odd polynomial coefficients (a,b,c) for iterated SV mapping—sits at the intersection of three mature mathematical fields, each offering different tools.

**Minimax approximation via Remez algorithm.** The Chebyshev equioscillation theorem guarantees that the best uniform polynomial approximation to any continuous function on an interval is characterized by alternating maximum-error points. The Polar Express paper (Amsel, Persson, Musco, Gower, 2025, arXiv:2505.16932) directly solves this for Muon: at each NS step t, choose the degree-(2q+1) odd polynomial minimizing max_{σ∈[ℓ_t, u_t]} |p_t(σ) − sign(σ)|. They prove this greedy strategy is optimal and achieves **cubic convergence** for degree-5 polynomials. Their per-iteration polynomial adaptation converges to machine precision in **8 iterations** (24 matmuls) versus 20 for standard NS. The CANS paper (Grishina et al., 2025, arXiv:2506.10935) independently derives Chebyshev-optimal NS coefficients using the same Remez framework.

However, both Polar Express and CANS target sign(σ)=1 for all σ. The CUM researcher's "stable-0.88" polynomial targets a *different* function—a constant 0.88 rather than 1.0. **This requires a modified Remez algorithm** where the target is f(σ)=0.88 on (0,1), yielding fundamentally different optimal coefficients. The modified problem is: min_{a,b,c} max_{σ∈[ε,1]} |p⁵(σ) − 0.88|, subject to p(σ) = aσ + bσ³ + cσ⁵. This is straightforward to solve numerically.

**SOS/SDP optimization.** With only 3 free parameters (a,b,c) and one linear constraint from the fixed-point condition aτ + bτ³ + cτ⁵ = τ (reducing to 2 free parameters), the problem is trivially solvable via sum-of-squares programming or even grid search. The fixed-point constraint is linear in (a,b,c). Stability requires |a + 3bτ² + 5cτ⁴| < 1, also linear. Contraction on (0,1) can be certified as an SOS condition via MOSEK or SeDuMi. The iterated map p⁵ has degree 5⁵ = 3125, but sampling-based SDP approximations handle this easily.

**Spectral filtering theory provides the deepest insight.** The NS iteration *is* a spectral filter in the sense of regularization theory (Engl, Hanke, Neubauer, 1996). Tikhonov regularization applies G_λ(σ) = σ²/(σ²+λ); Landweber iteration applies G_k(σ) = 1−(1−ασ²)^k. The NS polynomial is a degree-5 spectral filter. The key revelation: **targeting σ*=0.88 rather than 1.0 is implicit Tikhonov-like regularization**—it shrinks large SVs while amplifying small ones, preventing over-amplification of gradient noise. Regularization theory (Hansen, 2010) shows the optimal filter strength depends on the noise level, which changes during training. This predicts the 0.88 target is correct for certain noise regimes but potentially suboptimal as training progresses and gradient SNR improves.

**Markov brothers' inequality constrains the design space.** For ‖p‖_{[0,1]} ≤ M, the derivative bound |p'(σ)| ≤ 25M on [0,1] limits how fast small SVs can grow (p'(0) = a) while maintaining stability at the fixed point. The "cursed quintic" maximizes a=3.4445 (fast SV inflation) at the cost of no convergent fixed point. Any stable-0.88 polynomial must sacrifice inflation speed for stability.

**Core equation for gradient-based coefficient optimization:**
```
∂L/∂a = Σ_{j=1}^{5} [Π_{i=j+1}^{5} p'(p^{i-1}(σ))] · p^{j-1}(σ)
```
where L = Var[p⁵(σ)] over the input SV distribution. This is trivially differentiable in JAX/PyTorch.

*Why the stable polynomial might fail:* The researcher's own finding that ALL monotone SV mappings fail suggests the oscillation structure—not just the target value—matters. A super-stable polynomial (p'(0.88)≈0) eliminates oscillation entirely, which may remove the implicit regularization benefit. **The optimal design point is likely a polynomial with a weakly unstable fixed point (1 < |p'(σ*)| < ~1.6) that oscillates just enough to regularize but not enough to destabilize.**

---

## 2. Temporal averaging of subroutine outputs has deep theoretical grounding

The researcher's discovery—that EMA-averaging NS outputs improves training—connects to at least five distinct theoretical frameworks, each predicting this should work and offering quantitative predictions.

**Two-timescale stochastic approximation (Borkar, 1997).** The NS-EMA system is precisely a two-timescale SA: gradient momentum updates on the fast timescale (learning rate α), NS output EMA on the slow timescale (decay rate β ≪ α). Borkar proved a.s. convergence when timescales are properly separated. Konda & Tsitsiklis (2004) showed both components achieve optimal √n convergence with Polyak-Ruppert averaging. Dalal et al. (2018) provided the first finite-sample analysis. **Key prediction:** The optimal EMA decay rate satisfies β ∝ (D/σ²)^{1/3}, where D is the rate of change of the ideal orthogonalized gradient and σ² is the NS truncation noise variance.

**Signal processing: EMA as IIR low-pass filter.** The EMA has transfer function H(z) = α/(1−(1−α)z⁻¹), with cutoff frequency f_c ≈ α·f_s/(2π). The NS output oscillation (period-2-like) has dominant frequency at ~f_s/2. An EMA with α=0.05 attenuates this by approximately **26 dB**—essentially eliminating the high-frequency oscillation while preserving the slowly-evolving signal component. Wiener filter theory shows the EMA is *not* optimal; a second-order IIR filter tuned to the oscillation frequency would provide sharper rejection. Different EMA rates per layer (whose gradients change at different rates) could further improve performance.

**Variance reduction via anti-correlation.** The finding that NS₂ and NS₅ outputs have anti-correlated oscillation patterns is critical. For anti-correlated variables with correlation ρ < 0, the variance of their blend reduces by factor 1/(1−ρ²) > 1 compared to independent averaging. This means **super-linear variance reduction**—better than the O(1/√N) from averaging independent samples. The optimal blend weight for NS₂ and NS₅ has closed form: α* = Σᵢ(1−φ⁵(σᵢ))(φ²(σᵢ)−φ⁵(σᵢ)) / Σᵢ(φ²(σᵢ)−φ⁵(σᵢ))², where φ^k denotes k-fold iteration.

**Polyak-Ruppert averaging theory predicts the improvement.** The classic result (Polyak & Juditsky, 1992): averaged iterates achieve the Cramér-Rao bound √n(θ̄_n − θ*) → N(0, H⁻¹ΣH⁻¹), optimal even with suboptimal step sizes. The NS output is a noisy estimate of the polar factor; EMA averaging achieves variance reduction ~1/effective_window without requiring knowledge of the error structure.

**Precedent: Shampoo and K-FAC already do this.** K-FAC maintains EMA of Kronecker factors L_t = β₂L_{t-1} + (1−β₂)a_ta_t^T, which are *processed* quantities (outer products, not raw gradients). Shampoo similarly EMAs Kronecker factor estimates. SOAP (Vyas et al., 2024) explicitly exploits the two-timescale structure with periodic eigendecomposition updates. **The NS-EMA extends this validated principle to the orthogonalization subroutine.**

**Novel prediction: TD(λ)-style geometric blending of ALL NS iterates.** Rather than blending just NS₂+NS₅, use exponentially weighted combination of all iterates: Z = (1−λ)Σ_{k=1}^{5} λ^{k-1} NS_k, mirroring TD(λ)'s multi-step returns. The optimal λ should relate to the spectral radius of the NS polynomial's derivative at the fixed point. This generalizes the researcher's within-step blending and should outperform any pair combination.

*Would temporal averaging help other subroutines?* Yes: EMA of Shampoo's eigenvectors during periodic updates could smooth transitions; EMA of power iteration estimates for eigenvalue computation could reduce oscillation in near-degenerate cases; EMA of K-FAC inverse factors could provide better conditioning.

---

## 3. Seven theoretical frameworks agree: oscillation is a feature

The question of whether NS oscillation is a "bug" or "feature" has a remarkably convergent answer from independent theoretical perspectives.

**Edge of Stability (Cohen et al., ICLR 2021).** Full-batch GD on neural networks self-regulates: the loss Hessian's maximum eigenvalue increases to exactly 2/η, then stabilizes there while the loss decreases non-monotonically. Arora, Li & Panigrahi (ICML 2022) proved this creates an implicit regularization effect—GD at EoS follows a deterministic flow on the minimum-loss manifold. The NS oscillation creates **optimizer-internal instability** analogous to EoS, potentially producing a dual-layer edge-of-stability effect. Damian et al. (ICLR 2023) showed that when sharpness exceeds 2/η, a self-correcting mechanism drives it back—oscillation between stable and unstable regimes is the mechanism.

**Period-2 bifurcation theory.** With |p'(σ*)| ≈ 1.58 > 1, the NS iteration is just past the first period-doubling bifurcation. By Sharkovskii's theorem, period-2 is the "mildest" non-fixed-point dynamics (existence of period-2 implies period-1 but NOT higher periods). The period-2 orbit satisfies p(p(σ))=σ, and its stability is determined by |p'(a)·p'(b)| for orbit points {a,b}. The empirical "period-2-like" behavior around 0.88 is consistent with a **stable period-2 orbit born from the bifurcation**. The Feigenbaum route to chaos (δ ≈ 4.669) predicts that increasing |p'(σ*)| further would lead to period-4, then period-8, then chaos—but the current coefficients sit in the **optimal period-2 regime**.

**Edge of Chaos (Goto et al., 2025, arXiv:2508.17655).** Recent work proves that optimization success probabilities become "dramatically high NEAR the edge of chaos"—the boundary between regular and chaotic dynamics. Weakly chaotic processes near regular dynamics avoid local minima traps. This is "essentially different from conventional approaches with random noises such as SA." The NS iteration with |p'(σ*)| ≈ 1.58 sits precisely at this edge.

**Stochastic resonance.** In nonlinear systems, an optimal non-zero noise level maximizes signal-to-noise ratio (Gammaitoni et al., Rev. Mod. Phys. 1998). The NS oscillation amplitude represents an "implicit noise level." A Nature Communications 2024 paper demonstrated that stochastic resonance neurons can reduce required network size by an order of magnitude. This predicts an **inverted-U curve**: oscillation amplitude vs. training performance, with the current coefficients near the peak.

**Anisotropic noise advantage (Zhu et al., ICML 2019).** Anisotropic noise aligned with loss curvature is superior to isotropic noise for escaping sharp minima. The escape efficiency is Tr(HΣ)—alignment between Hessian H and noise covariance Σ. NS oscillation is anisotropic by construction (different per-SV), bounded, and deterministic. It should be **more effective than Gaussian noise injection** for escaping sharp minima.

**Implicit ensemble effect.** Each training step with oscillating SVs sees a different effective optimizer, analogous to dropout creating an implicit ensemble of architectures (Gal & Ghahramani, ICML 2016). The crucial difference: NS oscillation is deterministic and anti-correlated across consecutive steps, enabling **O(1/N) variance reduction** from blending rather than O(1/√N) from independent random samples.

**PAC-Bayes flat minima theory.** Training with perturbation-resilient dynamics (as NS oscillation forces) biases toward solutions at flat minima (Dziugaite & Roy, UAI 2017). Networks that maintain training progress despite oscillating update directions must be at flat, perturbation-tolerant minima. Bishop (1995) proved noise injection during training is equivalent to Tikhonov regularization; NS oscillation is a structured version of this.

**Experimental predictions from this synthesis:**

- A sweep of polynomial coefficients parameterized by |p'(σ*)| from 0.9 (convergent) to 3.0 (chaotic) should show an inverted-U training performance curve peaking near the period-2 regime (|p'(σ*)| ∈ [1.2, 1.8])
- The Lyapunov exponent of the NS polynomial iteration should correlate with training performance: slightly positive but small is optimal
- **Adaptive oscillation scheduling**—starting with larger |p'(σ*)| (more exploration) and reducing toward convergent coefficients (more exploitation)—analogous to learning rate schedules

---

## 4. Per-layer spectral adaptation is well-motivated but the implementation should be simple

Transformer layers exhibit dramatically different gradient spectral properties, providing clear motivation for per-layer NS adaptation.

**Documented spectral differences across layers.** The "Small Singular Values Matter" paper (2024, arXiv:2410.17770) analyzed BERT, Pythia-410M, and LLaMA-8B: attention-output matrices remain near Marchenko-Pastur (random-like), while value matrices develop strong outlier singular values in deeper layers. Gate/up-projection matrices display outliers; down-projection does not. The "From GaLore to WeLore" paper (2024) categorizes layers as Low-Rank Components (LRCs) with heavy-tail SV distributions versus Non-Low-rank Components (N-LRCs). **Attention layers in first/last blocks carry richer gradient signals than middle MLP layers.**

**Condition number variation is extreme.** Yousefzadeh & O'Leary (arXiv:1908.02400) found weight matrix condition numbers varying between **2 and 2,652** across layers in a 12-layer network. Chen & Chow (2014) showed NS requires approximately 2× more iterations for ill-conditioned matrices. Turbo-Muon's AOL preconditioner (Boissin et al., Dec 2025) addresses this by data-dependent diagonal scaling before NS, tightening the SV spread to enter the quadratic convergence regime faster—achieving **up to 2.8× speedup** per layer.

**μP compatibility is confirmed.** Yang et al.'s Tensor Programs prescribe different learning rates per layer type based on width. Critically, Muon's NS orthogonalization naturally achieves the spectral norm control μP requires. Multiple papers (arXiv:2603.00541, arXiv:2601.01306) confirm μP-compatible HP transfer with Muon up to 3.7B parameters. Any per-layer NS adaptation must preserve this.

**Practical recommendation: per-layer NS iteration count or blend weight.** The optimal blend weight for NS₂+NS₅ per layer has the analytical formula α*(layer) = Σᵢ(1−φ⁵(σᵢ))(φ²(σᵢ)−φ⁵(σᵢ)) / Σᵢ(φ²(σᵢ)−φ⁵(σᵢ))², computable from the layer's SV statistics. Layers with well-conditioned gradients (attention output) → α → 1 (favor NS₂); layers with many small SVs (QKV projections in first/last blocks) → α → 0 (favor NS₅). Tracking a cheap EMA of per-layer spectral statistics (power iteration for σ_max, Frobenius norm for approximate SV sum) enables adaptive selection at negligible cost.

**SpecMuon (arXiv:2602.16167)** takes this further by decomposing gradients into singular modes and applying RSAV updates individually per spectral direction with adaptive step sizes per mode. This is the most aggressive spectral routing approach published.

| Layer type | Typical gradient rank | Condition number | Recommended NS treatment |
|---|---|---|---|
| Embedding | N/A (use AdamW) | — | Skip NS entirely |
| Q,K projections | Higher effective rank | High | Full NS₅; may benefit from per-step coefficients |
| V projection | Moderate rank | Moderate | Standard NS₅ |
| Attention output | Low rank (near random) | Very high | Fewer iterations; gradients carry less info |
| FFN up/gate | Strong low-rank structure | Moderate-high | Standard NS₅ |
| Output projection | N/A (use AdamW) | — | Skip NS entirely |

---

## 5. Shrinkage theory provides a principled framework for curvature recovery

The blend W_final = (1−α)·NS(M) + α·(M/‖M‖) is exactly a James-Stein shrinkage toward the equalized target. Multiple estimation theories give quantitative predictions for optimal α.

**James-Stein shrinkage.** The classic estimator θ̂_JS = ν + (1 − (d−2)σ²/‖X−ν‖²)(X − ν) dominates MLE in dimension ≥ 3 when shrinking toward ANY structured target ν. Here NS(M) is the target. The optimal intensity is **α* ≈ σ²_signal / (σ²_signal + σ²_noise)**, where σ²_signal measures true gradient curvature variation across SVs and σ²_noise reflects mini-batch noise. For typical training with moderate gradient SNR, α ∈ [0.02, 0.10]—the researcher's 5% is in the right ballpark.

**Donoho-Gavish optimal SV thresholding offers a superior alternative to uniform blending.** For matrix Y = X + σZ, the optimal hard threshold is τ* ≈ (4/√3)·√n·σ ≈ **2.309·√n·σ** (Gavish & Donoho, 2014). SVs above this threshold contain signal; below are noise. The BBP phase transition (Baik, Ben Arous, Péché, 2005) provides the exact detectability boundary: spike ℓ is detectable iff ℓ > 1 + √γ where γ = p/n. **Rather than blending uniformly with α, applying per-SV nonlinear shrinkage η*(y) = √(y² − 4nσ²) is theoretically superior.** This achieves asymptotic MSE of 2nrσ², improving over hard thresholding (3nrσ²) and soft thresholding (6nrσ²).

**SURE provides data-driven α tuning without ground truth.** Stein's Unbiased Risk Estimate (Stein, 1981) gives an unbiased MSE estimate: SURE(θ̂) = ‖Y − θ̂(Y)‖² − dσ² + 2σ²·div(θ̂(Y)). For the blend estimator, SURE is smooth in α and can be minimized by simple line search. Candès, Sing-Long & Trzasko (2013) extended SURE to spectral estimators of matrices, including cross-terms for SV interactions.

**Tweedie's formula gives the optimal per-SV correction.** The posterior mean E[θ|Y=y] = y + σ²·∇log p_Y(y) is exact, requiring only the score function of the marginal SV density. If many SVs cluster near the Marchenko-Pastur bulk, the score pushes estimates toward the bulk center (shrinkage); outlier SVs far from the bulk get minimal correction. This provides **nonlinear, per-SV curvature restoration** rather than uniform blending—theoretically optimal under the Gaussian noise model.

**Ledoit-Wolf analytical nonlinear shrinkage.** For large-dimensional covariance matrices, the oracle optimal shrunk eigenvalue is d*_i = λ_i / |1 − c − c·λ_i·m_F(λ_i)|², involving the Stieltjes transform m_F of the limiting spectral distribution. Ledoit & Wolf (2020) provide a closed-form formula using the Hilbert transform of the spectral density, capturing **96%+ of potential variance reduction** in ~20 lines of code.

**Key prediction: α should be adaptive.** The optimal blend depends on gradient SNR, which improves during training. Early training (high noise, low signal) → small α (more equalization); late training (low noise, high signal) → larger α (more curvature preservation). α should also scale with batch size: larger B → more signal → larger α is safe. A practical rule: **α ≈ min(1, B·‖∇L‖² / trace(Cov[∇ℓ]))**.

**Row-wise restoration is preferable to column-wise.** K-FAC's Kronecker structure F_W ≈ A ⊗ Γ decomposes curvature into input-side (A) and output-side (Γ). Amari et al. (AISTATS 2019) proved the Fisher is approximately **unit-wise block-diagonal**, supporting per-output-neuron curvature (row norms) as the primary structure. Row norms correspond to output neuron magnitudes—a form of per-neuron learning rate. Computational cost: O(mn) for norm computation, negligible.

---

## 6. Scaling evidence is strongly favorable but the polynomial may need adaptation

The most important scaling question—does CUM's advantage persist?—has a largely positive answer from empirical evidence at massive scale.

**Muon is validated at 1T parameters.** Kimi K2 (Moonshot AI, 2025, arXiv:2507.20534) trained a 1T-parameter MoE model (32B activated) on 15.5T tokens using Muon with zero loss spikes (via MuonClip). Moonlight (Liu et al., 2025, arXiv:2502.16982) trained 3B/16B MoE on 5.7T tokens, achieving **~2× computational efficiency** vs AdamW. Critically, **neither modified the NS polynomial coefficients or iteration count at scale**. The same "cursed quintic" (3.4445, −4.7750, 2.0315) with 5 iterations was used from 100M to 1T parameters.

**Scaling modifications were about stability, not spectral processing.** Moonlight added weight decay (essential), consistent update RMS scaling (s = 0.2√n), and Nesterov-style momentum. Kimi K2 added QK-Clip for attention logit stability. Essential AI (arXiv:2505.02222) confirmed muP-compatible HP transfer up to 3.7B with Muon maintaining ~48-52% compute reduction across scales.

**The 2× advantage appears as a multiplicative shift in scaling laws.** Chinchilla-style analysis models loss as L(N,D) = E + A/N^α + B/D^β. Muon affects the B/D^β term (data efficiency), effectively doubling each token's value. This multiplicative effect does NOT shrink with scale—it was consistent from 100M through 1T in Moonlight/K2 experiments.

**How NS oscillation changes with dimension.** The scalar polynomial dynamics are dimension-independent per-SV, but aggregate effects depend on matrix size. By CLT, the variance of the mean SV contribution scales as **1/√n** where n = min(d_in, d_out). At n=64 (1.2M model), fluctuations are ~12.5%; at n=4096 (1B model), ~1.6%. The oscillation amplitude per-SV is unchanged, but its training impact decreases. **This suggests CUM's blending modifications may matter more at small scale and less at large scale**, where individual SV oscillations average out naturally.

**Random matrix theory predictions.** At large dimensions, gradient SV distributions converge to Marchenko-Pastur for the noise bulk plus heavy-tailed signal spikes (Martin & Mahoney, 2021). The bulk edge at (1+√γ)σ becomes sharper with dimension. At 1B parameters with matrices ~4096×16384, the MP prediction is essentially exact for the noise portion. **This means the NS polynomial encounters increasingly bimodal input: a concentrated MP bulk plus well-separated signal spikes.** The polynomial optimized for small irregular spectra may not be optimal for this structure.

**Whether the 0.88 target changes with dimension.** Under μP, the learning rate scales as 1/√n, absorbing dimension dependence. The absolute SV value after NS is irrelevant because μP's lr scaling compensates. What matters is equalization quality, which is dimension-independent. **The 0.88 is an artifact of the non-convergent quintic, not an optimized target—it does not need to change with width.**

---

## 7. Five genuinely novel mathematical directions worth pursuing

### Free probability for predicting gradient momentum spectra

**Core idea.** If stochastic gradients from different mini-batches are modeled as freely independent random matrices (Voiculescu, 1991), the spectral distribution of momentum m_t = β·m_{t-1} + (1−β)g_t can be computed via additive free convolution using the R-transform: R_{A+B}(z) = R_A(z) + R_B(z). Pennington & Bahri (ICML 2017) used this to compute Hessian spectra; Pennington, Schoenholz & Ganguli (2018) used free multiplicative convolution for deep network Jacobians. **Application:** Predict μ_G analytically, then optimize NS polynomial for E_{σ~μ_G}[|p⁵(σ) − target|²] rather than worst-case. *Risk:* Free independence is asymptotic; gradient temporal correlations from momentum may violate it. *Cost:* O(100) scalar iterations for Stieltjes transform, then standard optimization for coefficients—no per-step overhead.

### Krylov subspace connection to NS polynomial optimality

**Core idea.** NS iterations on matrix X implicitly construct vectors in span{X, X(X^TX), X(X^TX)², ...}—a Krylov subspace of the Gram matrix G = X^TX. Lanczos/Arnoldi algorithms produce the *optimal* polynomial approximation from this subspace in a given norm (Saad, 1992; Güttel, 2013). The Krylov-optimal polynomial adapts to the actual spectral distribution via Ritz values, potentially outperforming fixed coefficients. Casulli et al. (2023) explicitly applied low-memory Krylov methods to the matrix sign function. **The CANS framework already bridges this gap** by using Chebyshev theory for NS optimization. *Risk:* Building the actual Krylov basis requires orthogonalization, exactly what NS avoids. But the offline insight—using Krylov analysis on representative gradient matrices to find better fixed coefficients—is zero-cost at runtime.

### Wasserstein gradient flow perspective on SV equalization

**Core idea.** View the empirical SV distribution μ_k = (1/n)Σδ_{σᵢ^(k)} at NS step k as a point in Wasserstein space P₂(ℝ). Each NS step pushes μ_k via the polynomial map, giving pushforward p#μ_k. Design p(σ) to minimize the **Wasserstein-2 distance** W₂(p#μ, δ_{target}) rather than pointwise error. For 1D distributions, W₂ = (∫|F⁻¹_μ(t) − F⁻¹_ν(t)|² dt)^{1/2}. This accounts for the entire distribution shape, not just pointwise behavior. The JKO scheme (Jordan, Kinderlehrer, Otto, 1998) provides the variational framework. *Risk:* The pushforward of a polynomial is not a true gradient flow step; the 1D Wasserstein distance loses eigenvector structure information. *Cost:* O(n log n) for sorting + O(n) for W₂ computation; polynomial optimization is standard.

### Ergodic theory of the NS polynomial iteration

**Core idea.** Compute the invariant measure of x_{n+1} = p(x_n) via Ulam's method (discretize interval into bins, build Markov transition matrix, find stationary distribution). The Lyapunov exponent λ = lim(1/n)Σlog|p'(x_k)| classifies dynamics: λ < 0 → attracting cycle, λ > 0 → chaos, λ ≈ 0 → edge of chaos. For the cursed quintic, computing the exact period-2 orbit (solving p(p(σ))=σ) and its stability (|p'(a)·p'(b)| < 1?) would definitively characterize the oscillation. **Thermodynamic formalism** connects topological pressure P(t) = sup_μ{h_μ + t∫log|p'|dμ} to the tradeoff between entropy (exploration) and expansion rate (exploitation). *Implementation:* Ulam's method requires O(K²) for K-bin discretization—entirely feasible as a one-time offline analysis. Lyapunov exponent: O(N) scalar iterations.

### QSVT polynomial design insights

**Core idea.** Quantum Singular Value Transformation (Gilyén et al., STOC 2019) provides a complete characterization of achievable polynomial transformations of singular values. Tang & Tian (2023) showed all QSVT polynomial constructions reduce to truncated Chebyshev series. For the sign function: sign(x) ≈ (4/π)[T₁(x) − T₃(x)/3 + T₅(x)/5 − ...]. Truncating to degree 5 with only odd terms gives specific coefficients that are the Chebyshev-optimal approximation to sign—directly comparable to the NS quintic. **The truncated Chebyshev coefficients provide a principled starting point** for coefficient optimization, different from both the "cursed quintic" and the "stable-0.88" polynomial. *Risk:* QSVT's polynomial design is for single-shot application, not iterated composition. The boundedness constraint |P(x)| ≤ 1 is more restrictive than NS needs. *Cost:* Purely theoretical—insights inform coefficient choice with zero runtime cost.

---

## Synthesis and highest-priority experimental directions

The research literature converges on several actionable conclusions that go beyond what the researcher's 60 experiments have explored:

**The bifurcation diagram sweep is the single highest-value experiment.** Parameterize the polynomial family by |p'(σ*)| (oscillation strength) while fixing p(σ*)=σ* (fixed-point constraint). Sweep from 0.9 (convergent) through 1.0, 1.2, 1.5, 1.8, 2.5, 3.0. Theory strongly predicts an inverted-U with optimal performance in the period-2 regime (|p'(σ*)| ∈ [1.2, 1.8]). This directly tests whether oscillation is a feature vs. bug.

**SURE-optimized adaptive α for curvature recovery.** Rather than fixing α=5%, use SURE to select α from data at each step: evaluate SURE(α) for 5-10 candidate values and pick the minimum. SURE requires only observed quantities—no ground truth—and adds negligible overhead. This replaces guesswork with provably optimal shrinkage.

**Per-SV nonlinear shrinkage dominates uniform blending.** Replace (1−α)·NS(M) + α·(M/‖M‖) with Donoho-Gavish optimal shrinkage: for each SV σᵢ, compute η*(σᵢ) = √(σᵢ² − 4nσ²_noise) if above threshold, else clip to the NS-equalized value. This treats signal and noise SVs differently, which theory predicts is strictly superior.

**TD(λ)-style geometric blending of all NS iterates.** Replace the NS₂+NS₅ blend with Z = (1−λ)Σ_{k=1}^{5} λ^{k-1} NS_k. Sweep λ ∈ {0.3, 0.5, 0.7, 0.9}. This generalizes within-step blending and is predicted to dominate any pair combination by TD theory.

**Adaptive oscillation scheduling.** Start training with the cursed quintic (high oscillation → exploration) and gradually transition toward more convergent coefficients (lower |p'(σ*)| → exploitation). This mirrors learning rate schedules and simulated annealing temperature, both well-established in optimization theory.

**Per-layer NS iteration count based on tracked spectral statistics.** Maintain a cheap EMA of each layer's σ_max/σ_eff ratio. Layers with high condition number get NS₅; layers with low condition number get NS₃ or NS₂. This is the simplest per-layer adaptation and follows directly from Chen & Chow's (2014) convergence analysis showing ill-conditioned matrices need approximately 2× more iterations.

The mathematical toolkit for understanding and improving CUM is rich. The key intellectual contribution of this work—that non-convergent iteration with structured oscillation and temporal averaging outperforms exact methods—is validated by dynamical systems theory, regularization theory, and stochastic approximation theory. The path forward is clear: characterize the oscillation precisely via bifurcation analysis, optimize the polynomial via Chebyshev/Remez tools adapted for sub-unity targets, and replace heuristic curvature recovery with principled shrinkage estimators.