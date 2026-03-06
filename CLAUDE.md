# CUM Optimizer — Claude Instructions

## Project Overview
CUM (Curvature-Unified Muon) is a research optimizer that extends Muon with curvature recovery.
Current best: **v5 (Multi-Resolution NS)** — beats Muon by 0.0113 val_loss.

## Experiment Protocol

### Before implementing
- Check `experiments/SUMMARY.md` for what's been tried and key learnings
- Check individual `experiments/vN.md` files for detailed results on specific versions
- Don't repeat failed approaches (pre-NS modifications, stacking features, replacing NS with "better" orthogonalization)

### When implementing a new version
1. Create the optimizer file in `cum/cum_vN.py` - N as in which version it is
2. Export it from `cum/__init__.py`
3. Create `experiments/vN.md` with: hypothesis, architecture, key formula, cost analysis - N as in which version it is
4. Run **max 3-4 configs** per benchmark (always include Muon baseline + v5 best)

### After testing
1. Fill in results table and analysis in `experiments/vN.md`
2. Update `experiments/SUMMARY.md` with one-line result
3. Update verdict: SUCCESS / PARTIAL SUCCESS / FAILED with brief reason
4. If a key learning was discovered, add it to the Key Learnings section in SUMMARY.md

### Logging format for each experiment file (`experiments/vN.md`)
```markdown
# Experiment N: CUM vN — [Name]

**File:** `cum/cum_vN.py`
**Status:** PENDING / TESTED / FAILED / SUCCESS

## Hypothesis
[What we're testing and why]

## Architecture
[Pipeline diagram]

## Key Formula
[Core math]

## Cost
[Extra memory, extra compute vs Muon]

## Results
| Config | Val Loss | vs Muon | vs v5 |
|--------|----------|---------|-------|

## Trajectory (if tested)
| Step | Muon | vN |
|------|------|----|

## Analysis
[What happened and why]

## Verdict
[One-line takeaway]
```

## Critical Rules
- **Never run >4 configs in one benchmark.** Always include Muon + v5 as baselines.
- **Never use weight decay** with Muon/CUM (wd=0.01 dominates updates 12x)
- **NS steps = 5 always.** Can't reduce without quality loss.
- **Don't stack features** — they interfere. Refine one core approach.
- **NS "error" is beneficial** — don't try to make NS more accurate (SVD polar is WORSE)
- **Use `python3 -u`** for unbuffered output when running benchmarks
- **Use `tee`** to save benchmark output to `benchmarks/tier0/results_*.txt`

## Benchmark Setup (constant)
- Hardware: Apple M3 CPU, 4 threads
- Model: MicroGPT (d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256, ~1.2M params)
- Data: TinyShakespeare (1.1M chars, vocab=65, char-level)
- Training: 2000 steps, batch=32, warmup=200, cosine LR decay, seed=42
- Split: Muon/CUM for hidden 2D weights (lr=0.02), AdamW for embeddings/biases (lr=3e-4)
- Baseline: Muon NS=5, beta1=0.95 → val_loss=1.5190

## File Structure
- `cum/` — optimizer implementations
- `cum/newton_schulz.py` — NS variants (standard, multi-resolution, dampened)
- `benchmarks/tier0/` — benchmark scripts and raw results
- `experiments/` — experiment logs (one MD per version + summary)
- `CUM_BUILD_GUIDE.md` — architecture documentation
