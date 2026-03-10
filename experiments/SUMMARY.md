# CUM Optimizer — Experiment Summary

## Benchmark Baseline
- **Muon NS=5:** val_loss ≈ 1.515 (varies ±0.005 per run due to torch.compile)
- **NEW Best:** 12v2 TD(λ=0.5) +temporal d=-1.0 → val_loss = 1.4993 (-0.018 vs Muon)
- **Previous best:** v5 save@2 b=0.15 → val_loss ≈ 1.508 (-0.007 to -0.011 vs Muon)

## Results Table

| Version | Name | Val Loss | vs Muon | vs v5 | Status |
|---------|------|----------|---------|-------|--------|
| v1 | Pre-NS Factored Precond | 1.5187 | -0.0003 | +0.0110 | FAILED |
| v2 | Post-NS Row/Col Scaling | ~1.53 | +0.01 | +0.02 | FAILED |
| v3 | Soft NS (Raw Blend) | 1.5146 | -0.0044 | +0.0069 | SUCCESS (small) |
| 3b | NS Step Reduction | 1.5439 | +0.0249 | — | FAILED |
| v4 | Stacked Innovations | 1.5193 | +0.0003 | — | FAILED |
| **v5** | **Multi-Resolution NS** | **1.5077** | **-0.0113** | **baseline** | **BEST** |
| 5b | v5 Fine-Tuning (b=0.2) | 1.5113 | -0.0077 | +0.0036 | b=0.15 confirmed optimal |
| v6 | Adaptive Spectral Blend | — | — | — | PENDING |
| v7 | Orthogonal Feedback Loop | — | — | — | PENDING |
| v8 | Multi-Scale Curvature Blend | — | — | — | PENDING |
| v9 | Dampened Late-Stage NS | 1.5101 | -0.0089 | +0.0024 | PARTIAL SUCCESS |
| v10 | Dampened NS + Multi-Res | ~1.84 | terrible | — | FAILED (3.6x slower too) |
| v11 | Second-Moment Grafting | 1.5126 | -0.0064 | +0.0049 | FAILED |
| v12 | Gradient Difference Momentum | 1.5157 | -0.0033 | +0.0080 | FAILED |
| 2v1 | Randomized Top-k Curvature | 1.5296 | +0.0091 | — | FAILED |
| 2v2 | SVD Orthogonalization | 1.5307 | +0.0117 | — | FAILED (key discovery) |
| 3v1 | Warm-Started NS | ~2.10* | +0.32* | — | FAILED (killed early) |
| 3v2 | Cayley Retraction | 1.8465 | +0.3219 | — | FAILED |
| 3v3 | Directional Momentum | 1.5250 | +0.0004 | +0.0078 | FAILED (tied Muon) |
| 4v2 | SODA (Spectral Outlier Dampen) | 2.3401 | +0.83 | — | FAILED (catastrophic) |
| 4v3 | WGASU (Weight-Geometry) | 2.3246 | +0.81 | — | FAILED (catastrophic) |
| 5v1 | Lie Algebra Momentum + NS | 3.2463 | +1.73 | — | FAILED (catastrophic) |
| 5v2 β=3 | Soft EQ (tanh β=3) | 1.6375 | +0.12 | — | FAILED |
| 5v2 β=7 | Soft EQ (tanh β=7) | 1.5875 | +0.07 | — | FAILED |
| **5v3 p=8** | **Schatten-p Descent p=8** | **1.5202** | **+0.005** | **+0.013** | **CLOSEST non-NS** |
| 5v3 p=32 | Schatten-p Descent p=32 | 1.5308 | +0.016 | — | FAILED |
| 5v4 | Adaptive Schatten-p | 1.5309 | +0.016 | — | FAILED |
| 5v5 | SVD+NS Poly (diagnostic) | 1.5198 | +0.002 | — | DIAGNOSTIC (confirms NS=scalar SV) |
| **5v6** | **SVD NS Blend s2 b=0.25** | **1.5048** | **-0.017** | **-0.013** | **NEW BEST** |
| 5v6 | SVD NS Blend s2 b=0.15 | 1.5040-55 | -0.010-0.016 | — | BEST (consistent) |
| 5v6 | SVD Tilt ε=0.1 | 1.5244 | +0.011 | — | FAILED |
| **Series 6: Deep Research V1 Directions (H100 GPU, batch=32, 2000 steps)** |||||
| 6v1 | Polar Express (Remez std) | 1.5204 | +0.009 | — | FAILED (converges to 1.0) |
| 6v1 | Polar Express (blend) | 1.5335 | +0.022 | — | FAILED |
| 6v2 | PolarGrad NS | 1.5473 | +0.036 | — | FAILED |
| 6v2 | PolarGrad blend | 1.5485 | +0.037 | — | FAILED |
| 6v3 | PSGD Kron | 2.3934 | +0.88 | — | FAILED (catastrophic) |
| 6v4 | Dion full rank | 1.5328 | +0.021 | — | FAILED (exact polar = 1.0) |
| 6v4 | Dion r=32 | 1.6440 | +0.13 | — | FAILED (low-rank too lossy) |
| 6v5 | Halley 3iter | 1.5721 | +0.060 | — | FAILED (exact polar = 1.0) |
| 6v5 | QDWH 3iter | CRASHED | — | — | FAILED (linalg.solve ill-cond) |
| 6v6 | MARS γ=0.5 | 2.4905 | +0.98 | — | FAILED (catastrophic) |
| 6v6 | MARS γ=1.0 | 2.5023 | +0.99 | — | FAILED (catastrophic) |
| 6v7 | Warm-Started NS 2step | 3.0052 | +1.49 | — | FAILED (catastrophic) |
| 6v7 | Warm-Started NS hybrid | NaN | — | — | FAILED (diverged) |
| 6v8 | Optimal SV Shrinkage (hard) | 2.1458 | +0.63 | — | FAILED (zeroing SVs kills) |
| 6v8 | Optimal SV Shrinkage (blend) | 1.5191 | +0.007 | — | FAILED (closest non-NS) |
| 6v9 | Weighted Procrustes (mag) | 1.9280 | +0.42 | — | FAILED (SV ordering hurts) |
| 6v9 | Weighted Procrustes (decay) | 1.9406 | +0.43 | — | FAILED |
| **Series 7: Deep Research V2 — Huber SV Mapping (H100 GPU, batch=64, 1000 steps)** |||||
| 7v1 | Huber α=0.1 c=0.88 | 1.5622 | +0.027 | — | FAILED (closest monotone) |
| 7v1 | Huber α=0.3 c=0.85 | 1.7582 | +0.22 | — | FAILED |
| 7v1 | Huber α=0.3 c=0.88 | 1.7513 | +0.22 | — | FAILED |
| 7v1 | Huber α=0.3 c=0.92 | 1.7509 | +0.22 | — | FAILED |
| 7v1 | Huber α=0.5 c=0.88 | 2.0170 | +0.48 | — | FAILED |
| 7v1 | Huber α=0.7 c=0.88 | 2.2093 | +0.67 | — | FAILED |
| 7v1 | Smooth Huber α=0.3 c=0.88 | 1.7818 | +0.25 | — | FAILED |
| 7v1 | Power α=0.3 c=0.88 | 1.8009 | +0.27 | — | FAILED |
| 7v1 | Scheduled α 0.5→0.1 c=0.88 | 1.8555 | +0.32 | — | FAILED |
| 7v1 | Scheduled α 0.3→0.05 c=0.88 | 1.6912 | +0.16 | — | FAILED |

## Key Learnings

1. **NS is Muon's core strength AND weakness.** Equalizes directions but destroys curvature. Winning approach = partially recover curvature.
2. **Pre-NS modifications distort direction.** NS locks in damage. Modify AFTER NS or use NS intermediate.
3. **Post-NS modifications add noise unless signal is clean.** Raw gradient = noisy. NS intermediate = denoised.
4. **Don't stack innovations.** They interfere. Refine one core approach.
5. **NS steps = 5 sacred.** Can't reduce without quality loss.
6. **Weight decay kills Muon/CUM.** wd=0.01 dominates updates 12x.
7. **Gradient centralization hurts transformers.** Row means carry useful info.
8. **Coherence LR scaling conflicts with LR schedules.** Causes overshoot.
9. **b=0.15 optimal for save@2.** Higher adds noise.
10. **NS approximation error is a FEATURE.** Exact SVD polar (SVs=1.0) is WORSE than NS₅ (SVs≈0.877). Sub-unity SVs provide implicit regularization.
11. **Don't replace NS.** NS₅ > exact SVD polar factor.
12. **Modifying NS input barely helps.** NS is robust to input perturbations.
13. **torch.compile ~15-20% speedup** on model fwd+bwd only. Doesn't help NS.
14. **NS dominates everything.** Changing the input (directional momentum, gradient diffs, orth feedback) doesn't affect the output. NS is extremely robust to input preprocessing.
15. **The ONLY working strategy is intercepting NS mid-iteration** (v5's multi-resolution blend). All other approaches (pre-NS, post-NS, alternative orthogonalization, warm-start, manifold-aware direction) fail or tie.
16. **Schatten-p power mapping is the right mathematical framework** for understanding SV equalization. Power function σ^{1/(p-1)} lifts small SVs much better than tanh. p=8 (α=1/7) gives closest non-NS result to Muon (+0.005).
17. **More equalization isn't always better with SVD**: p=8 beats p=32. The NS polynomial's SPECIFIC oscillating SV mapping may provide beneficial properties beyond simple equalization.
18. **Lie algebra momentum is catastrophic**: projecting into so(m) discards the symmetric (scaling) component which carries most optimization signal.
19. **SVD-based SV manipulation is MORE PRECISE than matrix NS**: Computing exact SVs via SVD and applying the polynomial as a scalar function avoids floating point error accumulation across 5 matrix iterations. 5v6 SVD ns_blend consistently beats v5's matrix-level blend.
20. **5v6 SVD ns_blend is the new BEST approach**: Consistently beats Muon by 0.010-0.017. The SVD framework also enables exploration of custom SV curves beyond what NS iteration can express.

21. **ALL methods converging to SVs=1.0 lose to Muon**: Polar Express, Dion, Halley, PolarGrad — confirmed that exact polar factor is strictly worse. The 0.88 contraction IS essential.
22. **Monotone SV mappings cannot match NS**: Any function preserving SV ordering (larger input → larger output) leaves too much spread. NS's oscillating non-monotone map achieves near-perfect equalization that no monotone function can. The Huber theory from DR v2 was wrong.
23. **NS's oscillation IS the feature**: The polynomial doesn't converge (p(1)=0.701≠1.0). It scatters SVs across [0.68, 1.12]. The v5/5v6 blend works by CANCELING this oscillation (even/odd iterates are anti-correlated), producing a near-constant ~0.88. This is accidental but effective.
24. **Variance reduction (MARS) catastrophic with NS**: NS(g_t - g_{t-1}) on tiny gradient differences produces random orthogonal matrices that dominate the update.
25. **Warm-starting NS is fundamentally broken**: Starting NS from previous polar factor (already near-orthogonal) means NS barely changes it — you repeat last step's direction.
26. **Learned preconditioners (PSGD Kron) diverge**: Lie group Kronecker learning needs careful tuning beyond simple drop-in.
27. **Hard SV thresholding kills training**: Zeroing SVs below noise floor removes useful gradient directions.
28. **Shrinkage-NS blend is closest alternative**: Donoho-Gavish blended with NS₅ at +0.007 vs Muon — respects both denoising and equalization, but still loses.
29. **SVD-based methods 5x slower on GPU**: SVD isn't parallelizable like matmuls. On H100: SVD methods ~140s vs Muon ~30s. Matrix-iteration methods are the speed path.
30. **The only winning strategy remains NS intermediate blending** (v5/5v6). 40+ experiments across 7 series confirm nothing else works.

| **Series 8: Extended NS Iterate Blending (A100 GPU, batch=32, 2000 steps)** |||||
| 8v1 | Two-point s2+NS₈ b=0.15 | 1.5203 | -0.0015 | — | NOISE (flips ±0.005) |
| 8v1 | Two-point s3+NS₇ b=0.15 | 1.5193 | -0.0025 | — | NOISE (flips ±0.005) |
| 8v1 | Three-point NS₁+NS₃+NS₅ | 1.5225 | +0.0106 | — | FAILED (NS₁ noise late) |
| 8v1 | Geometric SV blend s2 b=0.15 | 1.5087 | -0.0032 | — | TIES 5v6 (same SVD cost) |
| **Series 9: General Intermediate Iterate Blending (A100 GPU, batch=32, 2000 steps)** |||||
| 9v1 | Dual-momentum pre_ns | 1.5294 | +0.0045 | — | FAILED (NS washes out) |
| **8v1** | **Input blend β=0.5 α=0.15** | **1.5139** | **-0.0110** | — | **BEST PRACTICAL (48s)** |
| **Series 10: Replication & Noise Analysis (A100 GPU, batch=32, 2000 steps)** |||||
| 8v1 | Two-point s2+NS₈ repl. | 1.5199 | +0.0045 | — | FAILED (8a was noise) |
| 8v1 | Two-point s3+NS₇ repl. | 1.5200 | +0.0047 | — | FAILED (8a was noise) |
| 8v1 | Geom SV repl. | 1.5107 | -0.0073 | — | Consistently 2nd (arithmetic > geometric) |
| 8v1 | Three-point repl. | 1.5198 | +0.0018 | — | Muon-tier, always last |
| **8v1** | **Input-blend repl.** | **1.5107** | **-0.0103** | — | **CONFIRMED: ties 5v6, 4.4x faster** |
| 9v1 | Dual-momentum pre_ns repl. | 1.5164 | -0.0046 | — | REVISED: noise (±0.005), not harmful |

| **8v1** | **Combined (iterate+input)** | **1.5008** | **-0.0191** | — | **NEW RECORD? Needs replication** |
| 8v1 | Scheduled three-point | 1.5173 | -0.0027 | — | Marginal, not worth complexity |
| 8v1 | Input-blend (run 3) | 1.5237 | +0.0037 | — | REGRESSED — instability revealed |
| **8v1** | **Combined (replication)** | **1.5062** | **-0.0162** | — | **CONFIRMED (2 runs: -0.019, -0.016)** |
| **Series 11: Learned Polynomial + Adaptive + Un-Equalization (A100 GPU, batch=32, 2000 steps)** |||||
| 11v1 | Stable-0.88 basic (2.0, -1.94, 0.84) | 1.5671 | +0.0448 | — | FAILED (catastrophic — weak equalization) |
| 11v1 | Stable-0.88 combined | 1.5544 | +0.0321 | — | FAILED (can't fix fundamental deficit) |
| **8v1** | **Combined (3rd run)** | **1.5091** | **-0.0122** | — | **CONFIRMED (3 runs: mean -0.016±0.004)** |
| **11v3** | **Un-equalization α=0.05** | **1.5077** | **-0.0136** | — | **PROMISING: -0.0014 vs combined. Needs replication** |
| 11v2 | Adaptive blend s=1.0 | 1.5113 | -0.0101 | — | FAILED (+0.002 vs combined, adapting amounts doesn't help) |
| **Series 11c: Scale Test (124M params, WikiText-103, A100 GPU, 2000 steps)** |||||
| **8v1** | **Combined (124M)** | **4.1393** | **-0.0144** | — | **HOLDS AT SCALE (same as 1.2M mean -0.016)** |
| **Series 11d: Generalization Test (1.2M MicroGPT, A100 GPU, 2000 steps)** |||||
| — | SmoothedAdam α=0.15 | 1.6080 | +0.0039 vs AdamW | — | FAILED (temporal avg hurts Adam) |
| — | SmoothedAdam α=0.30 | 1.6162 | +0.0122 vs AdamW | — | FAILED (more smoothing = more damage) |

31. **Within-run rankings are reliable; cross-run absolutes are not.** torch.compile causes ±0.006 shifts but ordering is preserved.
32. ~~**Input-blend ties 5v6 at matrix-path speed.**~~ REVISED: input-blend unstable (flipped to +0.004 in run 3). Combined mode more promising.
33. **Input-blend has trajectory dominance early** but can crash late-game. Not as stable as initially claimed.
34. **Modifying NS input is irrelevant, not harmful.** 9v1 pre_ns flips ±0.005 vs Muon across runs. NS Jacobian contracts perturbations to noise.
35. **Double oscillation cancellation (combined mode) may be the real breakthrough.** Within-step iterate blend + across-step temporal blend = -0.019 vs Muon at matrix speed. Systematic late-game acceleration. ONE RUN — needs replication.
36. **Smart correction weighting fails.** Magnitude-based (adaptive-res) and agreement-based (cosine-gated) both Muon-tier. Intelligent weighting breaks statistical cancellation.
37. **Simple averaging beats complex weighting.** Combined = two layers of simple blend. Adaptive/cosine = one layer of complex logic. Simple wins.
38. **Input-blend mean: -0.005±0.006 vs Muon** (4 runs). Real but noisy. Not the -0.010 initially claimed.
39. **Combined mode CONFIRMED across 2 runs**: -0.019 and -0.016 vs Muon. Delta between runs (0.003) is within noise. Combined is real.
40. **Stable-0.88 polynomial catastrophically fails**: +0.045 basic, +0.032 combined vs Muon. The oscillation isn't the feature — AGGRESSIVE EQUALIZATION is. Large polynomial coefficients are needed to pull small SVs up fast enough (σ=0.1→0.34 in one step). Those same large coefficients cause oscillation (|p'|>1 at fixed point). You can't decouple them with degree-5 in 5 steps. Blending is Phase 2: extracts equalization benefit while canceling oscillation cost.
41. **Revised learning 23**: oscillation is not "the feature" but the unavoidable price of aggressive equalization. The feature is the equalization itself.
42. **Per-layer adaptive blend fails**: adapting the AMOUNT of uniform blending per layer (+0.002 vs combined). Even "smart amount" adaptation doesn't beat fixed weights. Combined's uniform blend is already near-optimal.
43. **5% row-wise un-equalization shows signal**: -0.0014 vs combined, with characteristic "behind early → ahead late" trajectory (row-norm EMA warmup). If real, this is a third axis: (1) within-step oscillation cancel, (2) across-step oscillation cancel, (3) post-blend curvature recovery. ONE RUN — needs replication.
44. **Combined mean across 3 runs: -0.016 ± 0.004 vs Muon.** Runs: -0.019, -0.016, -0.012. Solidly confirmed.
45. **Combined holds at 124M params.** -0.014 vs Muon on GPT-2 Small / WikiText-103. Same late-game acceleration pattern (behind early, ahead late). Deep research predicted advantage might vanish at scale — it didn't.
46. **Temporal averaging does NOT generalize to Adam.** SmoothedAdam worse than AdamW (+0.004 at α=0.15, +0.012 at α=0.30). Adam has no structured oscillation to cancel. Blending without oscillation = sluggishness. The principle is NS-specific.
47. **The mechanism is oscillation cancellation, not generic smoothing.** Confirmed by: (a) stable polynomial fails (no oscillation = no equalization), (b) SmoothedAdam fails (no oscillation = blending hurts), (c) combined succeeds only with standard oscillating polynomial.

| **Series 12: Bifurcation Sweep, TD(λ), Adaptive Scheduling (A100 GPU, batch=32, 2000 steps)** |||||
| 12v1 | d=-1.0 combined (bifurcation boundary) | 1.5033 | -0.0152 | — | MARGINAL (-0.001 vs combined, within noise) |
| 12v1 | d=-1.4 combined | 1.5133 | -0.0052 | — | WORSE than standard combined (noise) |
| 12v1 | d=-1.58 combined (≈standard) | 1.5055 | -0.0088 | — | Matches combined (expected) |
| 12v1 | d=-2.0 combined | 1.5102 | -0.0042 | — | Worse with more oscillation |
| 12v1 | d=-2.8 combined | 1.5212 | +0.0068 | — | FAILED (too much oscillation hurts even with blending) |
| **12v2** | **TD(λ=0.5) +temporal** | **1.5011** | **-0.0156** | — | **NEW BEST: -0.004 vs combined (same run)** |
| 12v2 | TD(λ=0.3) +temporal | 1.5062 | -0.0106 | — | Ties combined (too concentrated on NS₅) |
| 12v2 | TD(λ=0.5) no-temporal | 1.5087 | -0.0110 | — | Multi-iterate alone ≈ input-blend level |
| 12v2 | TD(λ=0.7) +temporal | 1.5093 | -0.0104 | — | Too much early iterate noise |
| 12v2 | TD(λ=0.9) +temporal | 1.5103 | -0.0094 | — | Near-uniform: even more noise |
| 12v3 | Adaptive sched -1.8→-1.0 | 1.5120 | -0.0106 | — | FAILED (worse than constant combined) |
| 12v3 | Adaptive sched -2.2→-1.2 | 1.5154 | -0.0072 | — | FAILED (starting strong oscillation hurts) |
| 12v1 | d=-1.4 combined (re-run) | 1.5040 | -0.0203 | — | d=-1.4 beat d=-1.0 this time; optimal in [-1.0, -1.4] |
| 12v1 | d=-1.0 combined (re-run) | 1.5051 | -0.0192 | — | Confirmed real: consistently beats standard combined |
| **12v2** | **TD(λ=0.5) +temporal d=-1.0** | **1.4993** | **-0.0182** | — | **ALL-TIME BEST: sub-1.50, -0.009 vs combined** |
| 12v2 | TD(λ=0.5) +temporal (std, run 2) | 1.5050 | -0.0124 | — | TD(λ) replicated: -0.004 vs combined again |

48. **TD(λ) multi-iterate blending beats two-point blending.** λ=0.5 +temporal beat combined by -0.004 in same run (replicated across 2 runs). NS₃ and NS₄ carry useful oscillation-cancellation info that two-point (NS₂+NS₅) discards. More samples of oscillatory process → better cancellation (Nyquist principle).
49. **Within-step and across-step cancellation are orthogonal channels.** TD(λ=0.5) alone: -0.011 vs Muon. TD(λ=0.5) + temporal EMA: -0.012 to -0.016. The ~0.005 temporal contribution stacks on top of multi-iterate blending.
50. **Weaker polynomial oscillation is real with blending.** d=-1.0 to -1.4 range consistently beats standard d=-1.58 combined. Optimal in [-1.0, -1.4]. d=-2.8 catastrophic. When iterate blending handles cancellation, the polynomial should minimize oscillation while maximizing equalization speed.
51. **Oscillation scheduling is a dead end.** Annealing polynomial dynamics over training doesn't help. Optimal dynamics should be constant — there's no "explore more early" benefit in NS polynomial space.
52. **λ=0.5 is the TD(λ) sweet spot.** λ=0.3 too concentrated on final iterate (similar to two-point). λ=0.7+ lets too much early iterate noise through. λ=0.5 balances useful middle iterates (NS₃: 13%, NS₄: 26%) with noise suppression.
53. **TD(λ) + weak polynomial effects are additive.** Combining λ=0.5 +temporal with d=-1.0 yielded 1.4993 (-0.018 vs Muon, -0.009 vs combined). First sub-1.50 result. TD(λ) contributed ~-0.004, d=-1.0 contributed ~-0.006, total ~-0.009 vs combined. Three layers of oscillation management: (1) weaker polynomial oscillation, (2) multi-iterate within-step cancellation, (3) temporal across-step EMA.

| **Series 13: Minimax-Optimal Polynomials from Theory (A100 GPU, batch=32, 2000 steps)** |||||
| 13v1 | Minimax basic (a=2.68) | 1.5304 | +0.023 vs combined | — | FAILED (too gentle) |
| 13v1 | Minimax combined (a=2.68) | 1.5253 | +0.017 vs combined | — | FAILED (barely beats Muon) |
| 13v1 | Minvar combined (a=2.23) | 1.5444 | +0.020 | — | FAILED (weakest coefficients = worst) |
| 13v1 | L2 combined (a=2.67) | 1.5302 | +0.006 | — | FAILED |
| 13v1 | Minimax td (full recipe) | 1.5222 | -0.0003 | — | FAILED (Muon-tier with full treatment) |
| **12v2** | **Final recipe (3rd replication)** | **1.4992** | **-0.0233** | — | **CONFIRMED (3 runs: 1.4993, 1.4993, 1.4992)** |

54. **Coefficient magnitude trumps equalization quality.** The `a` coefficient (linear term) controls small-SV inflation speed. a≥~3.0 is non-negotiable. Minimax-optimal polynomials (a=2.68) achieve near-perfect equalization (Var[p⁵]≈0) but inflate small SVs 22% slower per step. After 5 iterations this compounds fatally. The hierarchy: (1) fast inflation (large a), (2) oscillation management (blending), (3) equalization quality (least important — 17x worse equalization still wins).
55. **The bifurcation family constraint is structural, not arbitrary.** Polynomials parameterized by p'(σ*) with fixed point at σ*=0.868 are FORCED to have large enough coefficients (a≥~3.0 for useful d range). Unconstrained optimization finds "gentle" polynomials that equalize by being weak, not by being aggressive-then-canceling. The constraint preserves the property that matters most.
56. **Dose-response: a=2.0 (+0.045), a=2.23 (+0.020), a=2.68 (+0.017 vs combined), a=3.15 (-0.018 vs Muon).** Clear monotonic relationship between linear coefficient magnitude and performance. No sweet spot at intermediate values.

| **Series 14: Deep Research V4 Mathematical Frameworks (A100 GPU, batch=32, 2000 steps)** |||||
| 14v1 | Ruiz5+NS3 basic | 1.5388 | +0.017 | — | FAILED (NS robust to input preprocessing) |
| 14v1 | Ruiz5+NS5 basic | 1.5246 | +0.003 | — | FAILED (Ruiz adds cost, no benefit) |
| 14v1 | Frame η=2.5 7step | 2.1400 | +0.618 | — | FAILED (catastrophic — Paradigm A dead) |
| 14v1 | Polar Express 5step (hand-tuned) | 1.5396 | +0.018 | — | FAILED (coefficients too gentle) |
| 14v1 | Ruiz5+NS3 combined | 1.5320 | +0.010 | — | FAILED |
| 14v1 | Ruiz5+NS5 combined | 1.5130 | -0.009 | — | FAILED (worse than combined alone) |

57. **Ruiz pre-conditioning is irrelevant to NS.** Diagonal row/col scaling before NS doesn't help — NS is robust to input preprocessing (reconfirms learning 14). Adding Ruiz just increases cost.
58. **Frame potential gradient flow is catastrophic in practice.** The cubic iteration σ→σ(1−η(σ²−c²)) with aggressive η is unstable with real gradients (+0.62 vs Muon). Six mathematical fields converging to the same iteration doesn't make it practically useful. Proves direct spectral targeting (Paradigm A) is dead.
59. **Hand-tuned step-adaptive coefficients fail when they're too gentle.** Polar Express with progressively gentler coefficients (a: 3.44→2.30) hit the coefficient magnitude issue from Series 13. Later steps need less correction but early steps need MORE aggression, not less.

| 14v1 | Polar Remez 5step | 1.5229 | +0.007 | — | FAILED (Remez-optimal still loses to NS) |
| 14v1 | Polar Remez 3step | 1.5409 | +0.025 | — | FAILED |

60. **Remez-optimal step-adaptive coefficients still lose.** Even with minimax-optimal polynomials computed via differential_evolution for each step's spectral interval, Polar Express (+0.007 vs Muon) can't beat NS's fixed quintic. The cursed quintic's specific oscillation dynamics are uniquely effective.
61. **Every alternative polynomial/iteration across 14 series has failed.** Frame potential, Ruiz, Polar Express (hand-tuned and Remez), minimax, minvar, L2, stable-0.88, Chebyshev sign — none beat standard NS. The cursed quintic is load-bearing.

| **Series 15: Universal Muon / Per-Head Orthogonalization (A100 GPU, batch=32, 2000 steps)** |||||
| Universal all s1d=0.015 | NS all 2D + norm 1D | 1.5280 | +0.007 | — | FAILED (NS on embeddings hurts) |
| Universal 2D + AdamW 1D | NS all 2D, AdamW 1D | 1.5288 | +0.007 | — | FAILED (confirms embeddings need Adam) |
| **PerHead 4 slices** | **Per-head QKV orthogonalization** | **1.4991** | **-0.025** | — | **NEW RECORD? Needs replication** |
| PerHead 3 slices | Per-Q/K/V orthogonalization | 1.5067 | -0.018 | — | Strong (matches blending recipe) |
| PerHead 12 slices | Per-Q/K/V/head orthogonalization | 1.5077 | -0.017 | — | Too fine-grained |

62. **NS on embeddings hurts.** Embeddings are lookup tables, not linear transforms. Orthogonalizing their gradients doesn't have the same geometric meaning as weight matrices. The Muon+AdamW split is structurally correct.
63. **Per-head orthogonalization gives -0.025 vs Muon (1 run).** Splitting QKV gradient into per-head slices (4 × 96×128) before NS respects multi-head structure. Different heads learn different features with different spectral profiles; mixing them in one NS forces suboptimal shared equalization. Head separation (4 slices) > Q/K/V separation (3 slices) > per-Q/K/V-per-head (12 slices, too fine).
64. **Per-head is orthogonal to iterate blending.** Per-head changes WHAT gets orthogonalized. Blending changes HOW oscillation is managed. They should stack. Potential combined: -0.034 vs Muon.

## Untested Ideas
- **CANS convergent coefficients** — polynomial coefficients that sum to 1.0 and actually converge, but targeting ~0.88 instead of 1.0
- **SDP-optimized polynomial** — sample gradient SV distributions, solve for coefficients minimizing training loss directly via semidefinite programming
- v6 (Adaptive Spectral Blend) — per-layer adaptive blend via spectral divergence
