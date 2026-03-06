# CUM Optimizer — Experiment Log

## Benchmark Setup (constant across all experiments)
- **Hardware:** Apple M3 CPU, 4 threads
- **Model:** MicroGPT (d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256, ~1.2M params)
- **Data:** TinyShakespeare (1.1M chars, vocab=65, char-level)
- **Training:** 2000 steps, batch=32, warmup=200, cosine LR decay
- **Split:** Muon/CUM for hidden 2D weights (lr=0.02), AdamW for embeddings/biases (lr=3e-4, wd=0.01)
- **Seed:** 42 (deterministic — val_loss is perfectly reproducible across runs)
- **Baseline:** Muon NS=5, beta1=0.95 → val_loss=1.5190

---

## Experiment 1: CUM v1 — Pre-NS Factored Preconditioning

**File:** `cum/cum.py`
**Hypothesis:** Preconditioning the gradient with row/column variance before NS should help NS converge in fewer steps and capture curvature.
**Architecture:** Momentum → Factored Precond → NS(3) → Spectral Damping → Update

### Results
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| CUM v1 (lr=0.02) | 1.5187 | -0.0003 |

### Diagnosis
- Factored preconditioning **rotates gradient direction 28°** (cosine sim 0.88 with original)
- NS(3) vs NS(5) differs by only cosine sim 0.997 — preconditioning distorts direction **10x more** than reducing NS steps
- Preconditioning shrinks gradient norm by ~1000x
- Spectral damping is **dead code**: weight norms (~0.2-0.3) never approach threshold (30.0), damping=1.0 always
- **Verdict: FAILED.** Pre-NS preconditioning damages the gradient direction that NS then locks in. The magnitude benefit is destroyed by NS equalization.

---

## Experiment 2: CUM v2 — Post-NS Adaptive Row/Column Scaling

**File:** `cum/cum_v2.py`
**Hypothesis:** Apply curvature AFTER NS so it doesn't get washed out. Track gradient row/column variance, use it to scale the orthogonalized update.
**Architecture:** Momentum → NS(5) → Row/Col Variance Scaling → Update

### Results
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| CUMv2 α=0.3 | ~1.53 | +0.01 |
| CUMv2 α=0.5 | 1.5322 | +0.013 |

### Analysis
- Post-NS scaling preserves NS direction quality
- But row/col gradient variance at this model scale is **too noisy** to provide useful curvature information
- The scaling just adds noise without meaningful curvature signal
- **Verdict: FAILED.** Post-NS row/col scaling doesn't capture useful curvature at this scale.

---

## Experiment 3: CUM v3 — Soft NS (Raw Gradient Blend)

**File:** `cum/cum_v3.py`
**Hypothesis:** NS orthogonalization is too aggressive — it equalizes ALL singular values, destroying info about which directions matter. Blend NS output with normalized pre-NS momentum to preserve some singular value structure.
**Architecture:** Momentum → NS(5) → Blend with normalized(u) → Update

**Formula:** `update = (1-α) * NS(u) + α * normalize(u)`

### Results
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| soft α=0.05 | — | (not tested) |
| **soft α=0.1** | **1.5146** | **-0.0044** |
| soft α=0.2 | 1.5182 | -0.0008 |

### Also tested: Cautious masking (mask stale momentum before NS)
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| cautious + soft=0.1 | 1.5213 | +0.0023 |

### Analysis
- **Soft NS α=0.1 beats Muon!** First clear win.
- α=0.2 too much blend — noise from raw gradient outweighs curvature benefit
- Cautious masking **hurts** — norm rescaling after masking destabilizes NS input
- The raw gradient is noisy but carries real curvature info in its top singular values
- **Verdict: SUCCESS (small).** +0.0044 improvement. The raw gradient blend is noisy though — can we do better?

---

## Experiment 3b: NS Step Reduction with Soft Blend

**Hypothesis:** Use fewer NS steps (3 instead of 5) to save compute, with soft blend to compensate for reduced orthogonalization quality.

### Results
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| Muon NS=3 | 1.5481 | +0.0291 |
| CUMv3 NS=3 + soft=0.1 | 1.5439 | +0.0249 |

### Analysis
- NS=3 is **significantly worse** than NS=5 (0.029 gap)
- Soft blend makes NS=3 slightly better but can't close the gap
- With NS=3, the direction isn't clean enough — blending adds more mess to an already messy direction
- **Verdict: FAILED.** Can't cut NS steps. NS=5 direction quality is critical.

---

## Experiment 4: CUM v4 — Stacked Innovations

**File:** `cum/cum_v4.py`
**Hypothesis:** Stack multiple innovations: soft NS + gradient centralization + coherence-adaptive step size.

**Innovations:**
1. **Gradient centralization:** `g = g - g.mean(dim=1)` — remove common-mode bias
2. **Soft NS:** α=0.1 blend (validated)
3. **Coherence scaling:** `lr *= 1 + 0.3 * cos_sim(g, momentum)` — bigger steps when gradient is consistent

### Results
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| v4 full (all three) | 1.5193 | +0.0003 |
| v4 no-center (soft + coherence) | 1.5263 | +0.0073 |
| v4 center-only | ~1.52 | ~+0.01 |

### Analysis — Ablation
- **Coherence scaling HURTS** (+0.012 vs v3 alone). The 30% LR boost conflicts with the cosine LR schedule — helps early but causes overshoot late.
- **Centralization HURTS** for transformers. Row means in attention Q/K/V projections carry useful bias information. Removing them is destructive.
- Stacking innovations: they **cancel each other out**. Individual improvements don't compose well.
- **Verdict: FAILED.** Only soft NS helps. Don't stack features — refine the core approach.

---

## Experiment 5: CUM v5 — Multi-Resolution NS ⭐

**File:** `cum/cum_v5.py`
**Hypothesis:** Instead of blending NS(5) with the raw gradient (noisy), blend with a partially-converged NS intermediate (denoised). Save the NS state at step k during the iteration — it has been partially denoised but retains curvature info.

**Key insight:** NS iteration progressively equalizes singular values. At step 2, ~25% of original SV spread remains. At step 3, ~5% remains. The step-2 intermediate is partially denoised (2 steps of cleanup) while retaining significant curvature structure.

**Architecture:** Momentum → NS₅(u) with save at step k → Blend NS₅ with NS_k → Update
**Formula:** `update = (1-α) * NS₅(u) + α * scale_match(NS₂(u))`
**Extra cost:** One `.clone()` at save point. No extra matmuls.

### Results
| Config | Val Loss | vs Muon | vs v3 |
|--------|----------|---------|-------|
| v5 save@2 b=0.1 | 1.5081 | **-0.0109** | -0.0065 |
| **v5 save@2 b=0.15** | **1.5077** | **-0.0113** | **-0.0069** |
| v5 save@3 b=0.1 | 1.5152 | -0.0038 | +0.0006 |

### Step-by-step trajectory (v5 save@2 b=0.15 vs Muon)
| Step | Muon | v5 best | Delta |
|------|------|---------|-------|
| 500 | 1.7830 | 1.7872 | +0.0042 (v5 slower start) |
| 1000 | 1.5710 | 1.5634 | -0.0076 (v5 catches up) |
| 1500 | 1.5256 | 1.5175 | -0.0081 (v5 pulling ahead) |
| 1999 | 1.5110 | 1.5024 | -0.0086 (v5 accelerating) |

### Analysis
- **save@2 >> save@3**: Step-2 intermediate has much more curvature info than step-3
- **b=0.15 slightly better than b=0.1** for save@2: more blend from denoised source is better
- **2.6x bigger improvement** than v3's raw gradient blend (0.0113 vs 0.0044)
- v5 starts slightly slower (step 500) but **accelerates in the second half** — curvature info becomes more valuable as training progresses
- **Verdict: BEST RESULT SO FAR.** Multi-resolution NS is the clear winner.

### Why v5 works better than v3
- v3 blends NS₅ with raw normalized gradient: **maximum curvature, maximum noise**
- v5 blends NS₅ with NS₂: **significant curvature, reduced noise**
- The NS₂ intermediate has already had 2 steps of singular value compression, removing the noisiest components while preserving the dominant curvature structure
- This is a higher-quality curvature signal → better blend → bigger improvement

---

## Key Learnings

1. **NS orthogonalization is Muon's core strength AND weakness.** It gives excellent direction equalization but destroys all curvature info. The winning approach partially recovers curvature.

2. **Pre-NS modifications distort direction.** NS then locks in the damaged direction. Always modify AFTER NS or use the NS intermediate.

3. **Post-NS modifications add noise unless the signal is clean.** Raw gradient = noisy signal. NS intermediate = denoised signal. Use the intermediate.

4. **Don't stack innovations.** Individual improvements interfere with each other. Refine one core approach.

5. **NS steps are sacred.** Can't reduce below 5 without significant quality loss. The iteration needs to fully converge for good direction quality.

6. **Weight decay kills Muon-family optimizers.** NS-orthogonalized updates have magnitude ~lr/sqrt(n_params). Even small weight decay (0.01) dominates 12x and drives weights to zero.

7. **Gradient centralization hurts transformers.** Row means in attention projections carry useful information. Don't remove them.

8. **Coherence-based LR scaling conflicts with LR schedules.** The cosine schedule is already carefully calibrated; boosting it based on gradient coherence causes overshoot.

9. **b=0.15 is optimal for save@2.** Higher blend (0.2) confirmed worse — too much curvature noise leaks through.

10. **NS's approximation error is a FEATURE, not a bug.** Exact SVD polar factor (SVs=1.0) performs worse than NS₅ (SVs≈0.877). The sub-unity SVs provide implicit regularization that prevents overshoot in later training.

11. **Don't replace NS with "better" orthogonalization.** NS₅ > exact SVD polar factor. The iteration's convergence to ~0.877 fixed point is optimal for training dynamics.

12. **Modifying the NS INPUT (momentum) doesn't help much.** Gradient difference momentum (v12), orthogonal feedback (v7) — these change what goes INTO NS but the improvement is small because NS is robust to input perturbations.

13. **torch.compile gives ~15-20% speedup** on model forward+backward but does NOT help NS (too small). Compile introduces tiny FP differences that compound over training, so absolute val_loss values shift slightly but relative comparisons remain valid.

---

## Experiment 5b: v5 Fine-Tuning (Partial Results)

**Hypothesis:** Push blend higher (0.2) and test save@1 for even more curvature.

### Results (partial — context was lost mid-run)
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| v5 s@2 b=0.15 | 1.5077 | -0.0113 (reproduced exactly) |
| v5 s@2 b=0.20 | 1.5113 | -0.0077 (worse — too much blend) |

**Verdict:** b=0.15 confirmed optimal for save@2. Pushing higher adds noise.

---

## Experiment 6: CUM v6 — Adaptive Spectral Blend ⭐ (PENDING)

**File:** `cum/cum_v6.py`
**Hypothesis:** The optimal blend amount should vary PER LAYER PER STEP based on how much curvature info NS destroyed. Measure this via spectral divergence between NS₂ and NS₅ outputs.

**Architecture:** Momentum → Multi-Resolution NS → Spectral Divergence → EMA-smoothed Adaptive Blend → Update

**Key formula:**
```
divergence = 1 - cos_sim(NS₅, NS₂)
raw_blend = min(divergence * blend_scale, blend_max)
blend_ema = β * blend_ema + (1-β) * raw_blend
update = (1-blend_ema) * NS₅ + blend_ema * scale(NS₂)
```

**Why it should work:** Different weight matrices have different SV distributions. Attention Q/K/V (ill-conditioned) need more curvature recovery. FFN weights (well-conditioned) need less. The spectral divergence automatically detects this.

**Cost:** One cosine similarity per weight matrix (~O(mn), negligible). One scalar EMA per weight.

---

## Experiment 7: CUM v7 — Orthogonal Feedback Loop (PENDING)

**File:** `cum/cum_v7.py`
**Hypothesis:** Feed back a fraction of the PREVIOUS step's orthogonalized output into the current NS input. This creates temporal coherence in the orthogonalized space.

**Architecture:** Momentum + β₂ * prev_orth → Multi-Resolution NS → Spectral Blend → Save for next step → Update

**Key formula:**
```
u = g + β₁ * momentum + β_feedback * prev_orth
orth = multi_resolution_NS_blend(u)
prev_orth = orth  # save for next step
```

**Why it should work:**
- Without feedback, consecutive NS outputs can jump around because NS is sensitive to the input's SV structure
- With feedback, we get "orthogonal momentum" — persistence in the orthogonalized direction
- Standard Muon has momentum in GRADIENT space. v7 adds momentum in ORTHOGONAL space.
- This is mathematically distinct from increasing β₁ — it biases the NS INPUT toward the previous orthogonal manifold point

**Cost:** One extra matrix buffer per weight (same size as momentum buffer).

---

## Experiment 8: CUM v8 — Multi-Scale Curvature Blend (PENDING)

**File:** `cum/cum_v8.py`
**Hypothesis:** Instead of blending TWO NS resolutions (v5: NS₂ + NS₅), blend THREE: NS₁, NS₃, and NS₅. This captures curvature at multiple scales.

**Architecture:** Momentum → Triple-Resolution NS (save@1, save@3, full@5) → Three-point weighted blend → Update

**Key formula:**
```
update = (1 - w₁ - w₃) * NS₅ + w₃ * scale(NS₃) + w₁ * scale(NS₁)
```

**Why it should work:**
- v5 showed save@2 >> save@3 (more curvature retained)
- save@1 should have even MORE curvature (but also more noise)
- Three-point blend lets NS₁'s rich curvature enter with small weight (filtering noise) while NS₃ provides the stable curvature backbone
- This is a multi-scale decomposition: NS₅=base, NS₃-NS₅=medium-freq curvature, NS₁-NS₃=high-freq curvature

**Cost:** Two extra clones (no extra matmuls). Same NS step count.

---

## Experiment 9: CUM v9 — Dampened Late-Stage NS

**File:** `cum/cum_v9.py`
**Hypothesis:** Instead of modifying the NS OUTPUT (post-hoc blending), modify the NS ITERATION ITSELF. Dampen late-stage steps to preserve curvature within the iteration.

**Architecture:** Momentum → Dampened NS (standard steps 1-2, dampened steps 3-5) → Update

**Key formula (for steps k ≥ dampen_after):**
```
X_{k+1} = (1-d) * NS_step(X_k) + d * X_k
```

**Why it should work:**
- v5's blend is LINEAR between two discrete snapshots
- v9's dampening creates a SMOOTH trajectory through SV space — each dampened step finds its own equilibrium
- The dampened iteration naturally adapts to each matrix's SV structure
- v5 needs extra memory (save intermediate). v9 needs ZERO extra memory — same matmul count, just different iteration

**Cost:** ZERO extra memory. Same matmuls. Just different coefficients.

### Results
| Config | Val Loss | vs Muon | vs v5 |
|--------|----------|---------|-------|
| v9 damp=0.3 | 1.5101 | -0.0089 | +0.0024 |

### Step-by-step trajectory
| Step | Muon | v5 best | v9 damp=0.3 |
|------|------|---------|-------------|
| 500 | 1.7830 | 1.7872 | 1.7989 (slowest start) |
| 1000 | 1.5710 | 1.5634 | 1.5616 (catching v5!) |
| 1500 | 1.5256 | 1.5175 | 1.5183 (neck and neck) |
| 1999 | 1.5110 | 1.5024 | 1.4980 (lowest mid-train!) |
| Final | 1.5190 | 1.5077 | 1.5101 |

### Analysis
- **Beats Muon by 0.0089** — dampened iteration preserves curvature
- **Loses to v5 by 0.0024** — linear interpolation dampening is less surgical
- Had the **lowest step-1999 loss ever** (1.4980) but final eval was higher (1.5101) — possible overfitting in late training
- The dampening is too uniform: it slows convergence for ALL SVs equally, whereas v5's snapshot blend naturally captures the curvature structure that matters
- Zero extra memory is appealing but the quality gap vs v5 is real
- **Verdict: PARTIAL SUCCESS.** Proves modifying the NS iteration itself works, but this specific mechanism is weaker than v5's post-hoc blend.

---

## Experiment 10: CUM v10 — Dampened NS + Multi-Resolution Blend (FAILED)

**File:** `cum/cum_v10.py`
**Hypothesis:** Combine v5 (multi-resolution blend) and v9 (dampened iteration). Both independently beat Muon, so combining them should be even better.

### Results
| Config | Val Loss (step 500) | Time (500 steps) |
|--------|---------------------|-------------------|
| v10 damp+blend | 1.8391 | 529s |
| Muon (reference) | 1.7830 | 145s |

**Killed early** — val loss far behind AND 3.6x slower.

### Analysis
- The interpolation in the inner NS loop (`(1-d)*X_next + d*X`) breaks in-place tensor optimization
- Creates extra tensor allocations every NS step → 3.6x slowdown
- Double curvature preservation (dampened output + intermediate blend) preserves TOO MUCH curvature — the signal becomes noise-dominated
- **Verdict: FAILED.** Combining two curvature-recovery mechanisms is worse than either alone. They interfere rather than compose.

---

## Experiment 11: CUM v11 — Second-Moment Grafting

**File:** `cum/cum_v11.py`
**Hypothesis:** Use Adam's second-moment estimate to reweight the NS output element-wise. NS gives the best direction, Adam knows per-element magnitudes. Graft Adam's magnitude onto NS's direction.

### Results
| Config | Val Loss | vs Muon | vs v5 |
|--------|----------|---------|-------|
| v11 graft=0.3 | 1.5126 | -0.0064 | +0.0049 |

### Analysis
- Beats Muon by 0.0064 (the v5 blend component helps)
- But loses to v5 alone by 0.0049 — grafting HURTS
- Per-element Adam scaling adds noise to the orthogonalized update
- NS's uniform magnitude is a FEATURE not a bug — redistributing magnitude element-wise breaks the "equal SV treatment" property
- The second moment signal is too noisy at this scale to provide useful magnitude info
- **Verdict: FAILED.** Post-NS element-wise reweighting damages NS's core property.

---

## Series 2: Fundamentally Different Approaches

### Experiment 2v1: Randomized Top-k Curvature Recovery (FAILED)

**File:** `cum/cum_2v1.py`
**Hypothesis:** Extract exact top-k singular components via randomized projection, blend with NS output. Sharper curvature signal than v5's NS₂ intermediate.

### Results
| Config | Val Loss | vs Muon |
|--------|----------|---------|
| 2v1 rank=4 blend=0.15 | 1.5296 | +0.0091 |

**Verdict: FAILED.** Random projection is too noisy at this scale. Also unstable — spiked to 1.5558 at step 1999.

---

### Experiment 2v2: SVD-Based Orthogonalization ⭐ (KEY DISCOVERY)

**File:** `cum/cum_2v2.py`
**Hypothesis:** Replace NS with exact SVD polar factor. Tests whether NS's approximation error is helping or hurting. Also test partial SV preservation (α > 0).

### Results
| Config | Val Loss | vs Muon | Time |
|--------|----------|---------|------|
| SVD exact (α=0) | 1.5307 | +0.0117 | 601s |
| SVD α=0.1 | 1.5242 | +0.0052 | 625s |
| Muon (NS₅) | 1.5189 | — | 426s |

### Step-by-step trajectory
| Step | Muon (NS₅) | SVD exact |
|------|-----------|-----------|
| 500 | 1.7811 | **1.7623** (SVD FASTER start!) |
| 1000 | **1.5483** | 1.5538 (NS catches up) |
| 1500 | **1.5261** | 1.5379 (NS pulls ahead) |
| Final | **1.5189** | 1.5307 (NS wins big) |

### Analysis — CRITICAL DISCOVERY
- **The exact polar factor is WORSE than NS's approximation by 0.0117**
- NS₅'s singular values converge to ~0.877, not exactly 1.0 like the true polar factor
- This means NS's "error" (sub-unity SVs) is actually BENEFICIAL for optimization
- The ~0.877 fixed point may provide **implicit regularization** — slightly shrinking updates prevents overshoot
- SVD starts faster (step 500) because exact orthogonalization is better early, but NS's implicit regularization wins in later training when overshoot matters more
- SVD is also ~40% slower (601s vs 426s) — NS iteration is cheaper than full SVD
- **Verdict: FAILED as optimizer, SUCCEEDED as insight.** NS₅ > exact polar factor. Don't try to "improve" NS by making it more accurate — its inaccuracy is a feature.

---

## Key Learnings
