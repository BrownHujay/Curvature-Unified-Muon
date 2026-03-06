# CUM Optimizer — Experiment Summary

## Benchmark Baseline
- **Muon NS=5:** val_loss ≈ 1.515 (varies ±0.005 per run due to torch.compile)
- **NEW Best:** 5v6 SVD ns_blend s2 b=0.25 → val_loss ≈ 1.505 (-0.013 to -0.017 vs Muon)
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

## Untested Ideas
- v6 (Adaptive Spectral Blend) — per-layer adaptive blend via spectral divergence
- v8 (Multi-Scale Curvature Blend) — three-point NS₁/NS₃/NS₅ blend
- **NS polynomial tuning** — design (a,b,c) coefficients with STABLE fixed points at different values. Current coefficients have an unstable fixed point at 0.868 — SVs oscillate, don't converge. Stable coefficients might behave very differently.
- **Per-SV-group treatment** — instead of uniform NS, apply different convergence rates to top-k vs bottom SVs during iteration
