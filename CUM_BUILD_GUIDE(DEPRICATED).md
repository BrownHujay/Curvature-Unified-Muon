# CUM: Curvature-Unified Muon

## Build Guide & Technical Specification

**Version:** 0.1.0-alpha  
**Authors:** Kota  
**Date:** February 2026

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Background: What CUM Builds On](#2-background-what-cum-builds-on)
3. [The CUM Algorithm](#3-the-cum-algorithm)
4. [Implementation Guide](#4-implementation-guide)
5. [GPU Optimization & Memory Layout](#5-gpu-optimization--memory-layout)
6. [Time Complexity Analysis](#6-time-complexity-analysis)
7. [Benchmark Suite Design](#7-benchmark-suite-design)
8. [Graphing & Evaluation Pipeline](#8-graphing--evaluation-pipeline)
9. [Known Risks & Mitigations](#9-known-risks--mitigations)
10. [Hyperparameter Guide](#10-hyperparameter-guide)
11. [File Structure & Build Instructions](#11-file-structure--build-instructions)

---

## 1. Motivation & Problem Statement

### What exists today

| Optimizer | Strengths | Weaknesses |
|-----------|-----------|------------|
| AdamW | Universal, well-understood, per-element adaptivity | Ignores matrix structure, O(2mn) extra memory for moments |
| Muon | Matrix-aware, spectral-norm control, memory-efficient (1 buffer), fast convergence on hidden layers | Only works on 2D params, no curvature info (treats all singular directions equally), needs 5 NS steps |
| MuonClip | Muon + stability at scale via QK-clip | Post-hoc band-aid, breaks orthogonality on clip, hard threshold, only monitors attention layers |
| SOAP | Full spectral whitening, best per-step quality | O(m² + n²) extra memory per layer, expensive eigenvector updates |
| COSMOS | Hybrid SOAP+Muon by eigensubspace | Complex, multiple hyperparameters, still carries SOAP's memory for top subspace |

### The gap CUM fills

CUM injects **cheap curvature information** into Muon's pipeline via factored preconditioning (O(m+n) memory) while replacing MuonClip's post-hoc clipping with **smooth per-layer spectral control**. The result:

- **Faster convergence** than Muon/MuonClip (curvature-informed directions)
- **Cheaper per-step compute** than Muon (fewer Newton-Schulz iterations due to tighter singular value spread)
- **Better stability** than MuonClip (smooth spectral damping instead of hard clipping)
- **Negligible memory overhead** vs Muon (O(m+n) per layer, <0.1% of weight size)

---

## 2. Background: What CUM Builds On

### 2.1 Muon Core Mechanics

Muon operates on 2D weight matrices W ∈ ℝᵐˣⁿ in hidden layers:

```
1. g = ∇L(W)                          # compute gradient
2. m ← β₁·m + (1 - β₁)·g             # momentum accumulation
3. u = nesterov_lookahead(g, m, β₁)   # Nesterov variant
4. X = NS_orthogonalize(u, steps=5)   # Newton-Schulz → polar factor
5. W ← W - η·scale·X - λ·W           # update + weight decay
```

The Newton-Schulz (NS) iteration approximates the polar factor of a matrix (the nearest orthogonal matrix). Given M = UΣVᵀ, the polar factor is UVᵀ. NS achieves this via:

```
X₀ = M / ‖M‖_F
For k = 1..N:
    A = X_k · X_kᵀ
    X_{k+1} = a·X_k + (b·A + c·A²)·X_k
```

With coefficients (a, b, c) = (3.4445, -4.775, 2.0315), 5 iterations suffice for transformer training.

**Key insight:** NS convergence rate depends on the **spread of singular values** in the input. If all singular values of X₀ are clustered near 1, convergence is fast (2-3 steps). If they span orders of magnitude, 5+ steps are needed.

### 2.2 MuonClip's QK-Clip

After the Muon update, MuonClip checks attention logits:

```
scores = (X @ W_q.T) @ (X @ W_k.T).T / √d_k
if max(scores) > τ:
    η_clip = τ / max(scores)
    W_q *= η_clip^α
    W_k *= η_clip^(1-α)
```

Problems:
- **Reactive, not preventive** — instability has already occurred when clip triggers
- **Breaks orthogonality** — the weight rescaling destroys the carefully computed orthogonal update
- **Scope-limited** — only monitors QK attention scores, blind to instabilities elsewhere
- **Discontinuous** — hard threshold creates a non-smooth optimization landscape

### 2.3 Why Factored Preconditioning Works

Adam maintains per-element second moments: v_ij = EMA(g²_ij). This requires O(mn) memory per layer.

SOAP/Shampoo maintains full Kronecker factor estimates: L = EMA(GGᵀ) ∈ ℝᵐˣᵐ and R = EMA(GᵀG) ∈ ℝⁿˣⁿ. This requires O(m² + n²) per layer.

**CUM's factored approach**: maintain row-wise and column-wise squared gradient norms:

```
r_i = EMA(Σⱼ g²_ij)    # how "active" is row i?     → O(m)
c_j = EMA(Σᵢ g²_ij)    # how "active" is column j?  → O(n)
```

This is equivalent to the diagonal of Adafactor's factored second-moment estimate. It captures which rows and columns of the weight matrix are receiving the most gradient signal, without tracking individual elements.

When we scale the momentum by 1/√r and 1/√c before orthogonalization, we:
1. **Compress the singular value spread** → fewer NS steps needed
2. **Amplify underexplored directions** → better convergence (same insight as Adam)
3. **Dampen exploding directions** → implicit stability (prevents QK-score blowup at the source)

---

## 3. The CUM Algorithm

### 3.1 Pseudocode

```
Algorithm: CUM (Curvature-Unified Muon)

Input:
    W ∈ ℝᵐˣⁿ         — weight matrix (2D hidden layer parameter)
    η                   — learning rate
    β₁ = 0.95           — momentum coefficient
    β₂ = 0.99           — second moment EMA coefficient
    λ                   — weight decay coefficient
    ns_steps = 3        — Newton-Schulz iterations (reduced from Muon's 5)
    ε = 1e-7            — numerical stability constant
    σ_max = 30.0        — spectral norm soft ceiling
    α_damp = 0.1        — spectral damping strength

State (initialized once, per layer):
    m ∈ ℝᵐˣⁿ  ← 0     — momentum buffer
    r ∈ ℝᵐ    ← 0     — row-wise second moment EMA
    c ∈ ℝⁿ    ← 0     — column-wise second moment EMA
    v ∈ ℝⁿ    ← rand  — power iteration vector (unit norm)

────────────────────────────────────────────────────────────
Per step t:
────────────────────────────────────────────────────────────

# ── Phase 1: Gradient & Momentum ──
g = ∇L(W)                                          # [O(mn)]
m ← β₁ · m + (1 - β₁) · g                         # [O(mn)]
u = g + β₁ · m           # Nesterov lookahead       [O(mn)]

# ── Phase 2: Factored Preconditioning ──
r ← β₂ · r + (1 - β₂) · sum(g², axis=1)           # [O(mn)] row norms
c ← β₂ · c + (1 - β₂) · sum(g², axis=0)           # [O(mn)] col norms

r̂ = r / (1 - β₂ᵗ)                                  # [O(m)]  bias correct
ĉ = c / (1 - β₂ᵗ)                                  # [O(n)]  bias correct

D_r = 1 / (√(r̂) + ε)                               # [O(m)]  row scale
D_c = 1 / (√(ĉ) + ε)                               # [O(n)]  col scale

ũ = D_r[:, None] * u * D_c[None, :]                 # [O(mn)] broadcast

# ── Phase 3: Newton-Schulz Orthogonalization ──
X = ũ / (‖ũ‖_F + ε)                                # [O(mn)] normalize
transpose = (m > n)
if transpose: X = Xᵀ                                # work with wider matrix

for i in 1..ns_steps:                                # 3 iterations
    A = X @ Xᵀ                                      # [O(k²p)] where k=min(m,n), p=max(m,n)
    B = b·A + c·(A @ A)                             # [O(k³)]
    X = a·X + B @ X                                 # [O(k²p)]

if transpose: X = Xᵀ
orth = X

# ── Phase 4: Spectral Norm Control ──
v ← W @ (Wᵀ @ v)                                   # [O(mn)] one power iteration step
v ← v / (‖v‖ + ε)                                   # [O(n)]  normalize
σ_est = ‖W @ v‖                                     # [O(mn)] estimate σ_max(W)

damping = 1.0 / (1.0 + α_damp · max(0, σ_est - σ_max))  # smooth sigmoid-like decay

# ── Phase 5: Weight Update ──
scale = √(max(1, m/n))                              # Muon's aspect ratio scaling
W ← W - η · damping · scale · orth - λ · W          # [O(mn)]
```

### 3.2 Mathematical Justification

**Why precondition before orthogonalization (not after)?**

If you precondition after orthogonalization, you're distorting the already-orthogonal update. The singular values of the final update would be the preconditioning scales, which defeats the purpose of orthogonalization.

By preconditioning before, you change **which orthogonal matrix** NS converges to. Let G be the raw momentum and P = diag(D_r) · G · diag(D_c) be the preconditioned version. Their SVDs are different:

```
G = U_G · Σ_G · V_Gᵀ    → polar factor = U_G · V_Gᵀ
P = U_P · Σ_P · V_Pᵀ    → polar factor = U_P · V_Pᵀ
```

The preconditioned polar factor U_P · V_Pᵀ encodes curvature information in its singular vectors — directions where the gradient has been consistently small get amplified in the preconditioned space, which rotates the orthogonal update to explore those directions more aggressively. This is the core mechanism by which CUM achieves faster convergence.

**Why does preconditioning reduce NS steps?**

The NS iteration converges as φᴺ(σ) → 1 for all singular values σ. The rate of convergence depends on how far the initial singular values are from 1. The factored preconditioning acts like a normalization: by scaling rows and columns by the inverse of their average gradient magnitude, we're compressing the singular value spectrum of the input. Empirically, this should tighten the ratio σ_max/σ_min by 2-5x, allowing NS to converge in 3 steps instead of 5.

**Why smooth damping instead of hard clipping?**

Hard clipping creates a discontinuity in the optimization landscape. When QK-Clip activates, the effective learning rate drops discontinuously, which can cause oscillation around the clip boundary. Smooth damping via:

```
damping = 1 / (1 + α · max(0, σ_est - σ_target))
```

creates a continuous, differentiable modulation. As σ_est approaches σ_target from below, damping ≈ 1 (no effect). As σ_est grows past σ_target, damping smoothly decreases toward 0. This is analogous to soft-capping but applied to the spectral norm of the weight matrix itself, not to downstream attention scores.

### 3.3 What CUM Does NOT Handle

CUM, like Muon, is **only for 2D hidden layer weights**. The following parameters must use a separate optimizer (AdamW recommended):

- Embedding matrices (input layer)
- Output projection / classifier head
- 1D parameters: biases, LayerNorm/RMSNorm gains
- Any parameter with ndim ≠ 2

The recommended setup is a hybrid optimizer:

```python
param_groups = [
    {"params": hidden_2d_weights, "optimizer": "CUM", ...},
    {"params": everything_else,   "optimizer": "AdamW", ...},
]
```

---

## 4. Implementation Guide

### 4.1 Core Implementation Structure

```
cum_optimizer/
├── __init__.py
├── cum.py                  # Main optimizer class
├── newton_schulz.py        # NS iteration (separate for profiling)
├── factored_precond.py     # Factored preconditioning logic
├── spectral_control.py     # Power iteration + damping
├── hybrid.py               # CUM + AdamW hybrid wrapper
└── utils.py                # Helpers (aspect ratio scaling, etc.)
```

### 4.2 Key Implementation Details

#### 4.2.1 The CUM Class

Subclass `torch.optim.Optimizer`. Each param group contains:

```python
defaults = {
    "lr": 0.02,
    "beta1": 0.95,
    "beta2": 0.99,
    "weight_decay": 0.01,
    "ns_steps": 3,
    "eps": 1e-7,
    "sigma_max": 30.0,
    "alpha_damp": 0.1,
    "nesterov": True,
}
```

State per parameter (stored in `self.state[p]`):

```python
state = {
    "step": 0,                          # int
    "momentum_buffer": torch.zeros_like(p),  # ℝᵐˣⁿ
    "row_var": torch.zeros(p.shape[0]),       # ℝᵐ
    "col_var": torch.zeros(p.shape[1]),       # ℝⁿ
    "power_iter_v": torch.randn(p.shape[1]),  # ℝⁿ (initialized random, normalized)
}
```

#### 4.2.2 Newton-Schulz Implementation

This must be a standalone function, not a method, for `torch.compile` compatibility:

```python
@torch.compile
def newton_schulz_orthogonalize(G: Tensor, steps: int = 3, eps: float = 1e-7) -> Tensor:
    """
    Compute approximate polar factor of G via Newton-Schulz iteration.
    
    Args:
        G: Input matrix (m x n), should be pre-normalized
        steps: Number of NS iterations (CUM default: 3)
        eps: Numerical stability
    
    Returns:
        Approximate polar factor (m x n) with singular values ≈ 1
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    if transpose:
        X = X.T
    
    return X
```

**Critical:** The NS function must work on the smaller dimension. Always transpose so that the "rows" dimension is ≤ "cols" dimension. This turns O(m²n) into O(min(m,n)² · max(m,n)).

#### 4.2.3 Factored Preconditioning

```python
def apply_factored_precond(
    u: Tensor,          # momentum (m x n)
    g: Tensor,          # current gradient (m x n)
    row_var: Tensor,    # running row variance (m,)
    col_var: Tensor,    # running column variance (n,)
    beta2: float,
    step: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Update factored second moments and apply preconditioning.
    
    Returns:
        (preconditioned_u, updated_row_var, updated_col_var)
    """
    # Update EMAs (in-place for memory)
    g_sq = g * g                          # elementwise square
    row_var.mul_(beta2).add_(g_sq.sum(dim=1), alpha=1 - beta2)
    col_var.mul_(beta2).add_(g_sq.sum(dim=0), alpha=1 - beta2)
    
    # Bias correction
    bc = 1.0 - beta2 ** step
    row_scale = 1.0 / (row_var.div(bc).sqrt() + eps)  # (m,)
    col_scale = 1.0 / (col_var.div(bc).sqrt() + eps)  # (n,)
    
    # Apply as outer product scaling: O(mn)
    preconditioned = u * row_scale[:, None] * col_scale[None, :]
    
    return preconditioned, row_var, col_var
```

**Memory note:** `g_sq` is a temporary of size O(mn). This is unavoidable — we need the squared gradient to compute row/col sums. However, it can be fused into a single kernel (see Section 5).

#### 4.2.4 Spectral Control

```python
def spectral_damping(
    W: Tensor,            # weight matrix (m x n)
    v: Tensor,            # power iteration vector (n,)
    sigma_max: float,
    alpha_damp: float,
) -> Tuple[float, Tensor]:
    """
    One step of power iteration + smooth spectral damping.
    
    Returns:
        (damping_factor, updated_v)
    """
    # One power iteration step: estimate σ_max(W)
    Wv = W.T @ (W @ v)          # (n,) — note: W@v is (m,), W.T@(.) is (n,)
                                 # This computes v' = WᵀWv, whose dominant eigenvector
                                 # gives σ²_max
    v_new = Wv / (Wv.norm() + 1e-7)
    sigma_est = (W @ v_new).norm()  # ≈ σ_max(W)
    
    # Smooth damping
    excess = max(0.0, sigma_est.item() - sigma_max)
    damping = 1.0 / (1.0 + alpha_damp * excess)
    
    return damping, v_new
```

**Why WᵀW instead of WWᵀ?** We use v ∈ ℝⁿ and compute WᵀWv because typically n ≤ m in transformer weight matrices (or they're square). The cost is 2 × O(mn) regardless of which side we iterate on, but this form lets us store v in the smaller dimension.

### 4.3 Hybrid Optimizer Wrapper

```python
class CUMWithAuxAdam(torch.optim.Optimizer):
    """
    Hybrid optimizer: CUM for 2D hidden weights, AdamW for everything else.
    
    Usage:
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2 or p in embed_and_head]
        
        param_groups = [
            {"params": hidden_weights, "use_cum": True, "lr": 0.02, ...},
            {"params": other_params,   "use_cum": False, "lr": 3e-4, "betas": (0.9, 0.95), ...},
        ]
        optimizer = CUMWithAuxAdam(param_groups)
    """
```

This mirrors Muon's `MuonWithAuxAdam` pattern for drop-in replacement.

---

## 5. GPU Optimization & Memory Layout

This section is critical. A mathematically elegant optimizer that's GPU-unfriendly is useless.

### 5.1 Memory Budget Analysis

For a weight matrix W ∈ ℝᵐˣⁿ in fp32:

| Component | Size (bytes) | Relative to W |
|-----------|-------------|---------------|
| W (parameter) | 4mn | 1.0× |
| m (momentum) | 4mn | 1.0× |
| r (row var) | 4m | m/mn ≈ 0 |
| c (col var) | 4n | n/mn ≈ 0 |
| v (power iter) | 4n | n/mn ≈ 0 |
| **Total state** | **8mn + 4(m+2n)** | **≈ 2.0×** |

**Comparison:**

| Optimizer | State per param | For 4096×4096 |
|-----------|----------------|---------------|
| SGD+Momentum | 1 buffer = 4mn | 64 MB |
| AdamW | 2 buffers = 8mn | 128 MB |
| Muon | 1 buffer = 4mn | 64 MB |
| SOAP | 1 buffer + 2 Kronecker = 4mn + 4(m² + n²) | 192 MB |
| **CUM** | **1 buffer + vectors = 4mn + 4(m + 2n)** | **~64.05 MB** |

CUM's memory overhead vs Muon is **<0.1%**. It's essentially free.

### 5.2 CUDA Kernel Fusion Strategy

The dominant cost in CUM (and Muon) is the Newton-Schulz iteration. But outside of NS, there are several element-wise or broadcast operations that can be fused:

#### Fusion Group 1: Momentum + Preconditioning (fuse into one kernel)

The operations:
```
m = β₁·m + (1-β₁)·g          # element-wise
u = g + β₁·m                  # element-wise
row_sq = sum(g², dim=1)        # reduction
col_sq = sum(g², dim=0)        # reduction
r = β₂·r + (1-β₂)·row_sq     # element-wise
c = β₂·c + (1-β₂)·col_sq     # element-wise
ũ = D_r[:,None] * u * D_c[None,:]  # broadcast multiply
```

**Without fusion:** 7 separate kernel launches, 7 reads+writes of the full mn matrix.  
**With fusion:** 1 kernel launch, 1 read of g and m, 1 write of ũ and updated m/r/c.

This is a **7x reduction in memory bandwidth** for the pre-NS phase.

**Implementation approach:** Use `torch.compile` with `fullgraph=True` or write a custom Triton kernel. `torch.compile` should handle this automatically for simple elementwise + reduction patterns, but verify with profiling.

```python
# Triton kernel skeleton for fused momentum + preconditioning
@triton.jit
def fused_momentum_precond_kernel(
    g_ptr, m_ptr, u_out_ptr,   # (m, n) tensors
    r_ptr, c_ptr,               # (m,) and (n,) vectors
    beta1, beta2, step, eps,
    M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each block processes a BLOCK_M × BLOCK_N tile
    # 1. Load g[i,j] and m[i,j]
    # 2. Update m[i,j] in-place
    # 3. Compute u[i,j] = g[i,j] + β₁·m[i,j]
    # 4. Accumulate g²[i,j] into row_sum and col_sum via atomic_add
    # 5. (Second pass or barrier) Apply D_r, D_c scaling to u
    pass
```

**Note:** The row/col reduction + broadcast scaling is tricky to fuse perfectly because the scaling depends on the reduction output. Two approaches:
1. **Two-pass kernel:** First pass computes m, u, and row/col sums. Second pass applies scaling. Still saves 5 kernel launches.
2. **Approximate:** Use r and c from the *previous* step (lagged by 1 step). This avoids the dependency and allows full single-pass fusion. The lag is negligible since r and c are EMAs with β₂ = 0.99 — one step of lag changes them by <1%.

**Recommendation:** Use the lagged approach (option 2) for maximum fusion. This is a pragmatic approximation that costs virtually nothing in convergence quality.

#### Fusion Group 2: Spectral Control + Weight Update

```
Wv = W @ v                    # matmul (cannot fuse)
σ = ‖Wv‖                      # reduction
damping = 1/(1 + α·max(0,σ-σ_max))  # scalar
W = W - η·damping·scale·orth - λ·W  # element-wise
```

The weight update can be fused into a single kernel:
```
W[i,j] = (1 - λ) · W[i,j] - η · damping · scale · orth[i,j]
```

This is a simple fused multiply-add, which `torch.compile` handles trivially.

### 5.3 Mixed Precision Strategy

CUM should support bf16/fp16 training with the following precision rules:

| Operation | Precision | Rationale |
|-----------|-----------|-----------|
| Gradient accumulation | fp32 | Standard practice |
| Momentum buffer | fp32 | EMA accumulation needs precision |
| Row/col variance (r, c) | fp32 | Small vectors, EMA stability |
| Preconditioning multiply | bf16 | Output feeds into NS which normalizes anyway |
| Newton-Schulz iteration | bf16 | Matrix multiplies benefit from tensor cores; NS is self-correcting |
| Power iteration vector | fp32 | Small vector, needs precision for convergence |
| Weight update | fp32 master weights | Standard AMP pattern |

**Critical:** The NS iteration involves matrix multiplies. On A100/H100, bf16 matrix multiplies run at 2x the FLOP rate of fp32. Since NS is the dominant cost, running it in bf16 cuts CUM's overhead nearly in half compared to fp32 NS.

Muon already demonstrates that bf16 NS works fine — the iteration is self-correcting (errors in early steps get washed out by later steps).

### 5.4 Potential GPU Bottlenecks & Mitigations

#### Bottleneck 1: Memory Bandwidth in Preconditioning

The factored preconditioning reads the full gradient matrix to compute row/col sums, then reads the momentum matrix to apply scaling. This is 2 full reads of O(mn) data.

**Mitigation:** Kernel fusion (Section 5.2). With the lagged approach, preconditioning adds zero extra memory reads — it piggybacks on the momentum update kernel.

#### Bottleneck 2: Power Iteration Memory Reads

Computing W @ v requires reading the full weight matrix W. This is O(mn) memory bandwidth for an O(mn) compute operation — memory-bound on GPU.

**Mitigation:** This is unavoidable if we want spectral control, but it's only 1 matmul vs. NS's 3×2 = 6 matmuls. It's <15% of total cost. If profiling shows this is still painful, we can:
- Run power iteration every K steps instead of every step (K=5 is fine, σ_max changes slowly)
- Amortize: combine the W @ v computation with the weight update kernel

#### Bottleneck 3: NS Iteration — Compute Bound

The NS iteration is 3 iterations of: `A = X @ X.T` + `B = b*A + c*A@A` + `X = a*X + B@X`. Each iteration is 3 matrix multiplies. Total: 9 matrix multiplies.

On A100, for 4096×4096 in bf16:
- Each matmul ≈ 137 GFLOP, throughput ≈ 312 TFLOP/s → ~0.44ms
- 9 matmuls → ~4ms per layer

For Muon (5 iterations, 15 matmuls): ~6.6ms per layer.  
For CUM (3 iterations, 9 matmuls): ~4ms per layer.  
**Savings: ~40% of NS compute, ~2.6ms per layer.**

For a model with 24 hidden layers, that's ~63ms saved per training step (Muon: 158ms → CUM: 96ms for NS alone).

#### Bottleneck 4: Kernel Launch Overhead

Each unfused operation = 1 CUDA kernel launch ≈ 5-10μs. With many small operations (bias correction, scalar divides, vector normalizations), this adds up.

**Mitigation:** `torch.compile` with `mode="max-autotune"` should fuse nearly all elementwise/broadcast ops. Profile with `torch.profiler` to verify no excessive launch gaps.

#### Bottleneck 5: Small Tensor Operations

The row/col variance vectors (r ∈ ℝᵐ, c ∈ ℝⁿ) are tiny. Operations on them (sqrt, divide, bias correction) will severely underutilize the GPU.

**Mitigation:** These must be fused into the larger kernels (Fusion Groups 1 and 2). Never launch standalone kernels for O(m) or O(n) operations.

### 5.5 Multi-GPU / Distributed Training Notes

CUM's distributed properties are identical to Muon's:
- **Data Parallel:** All state (momentum, r, c, v) is per-replica and synchronized via gradient all-reduce (same as any optimizer).
- **Tensor Parallel:** If W is sharded column-wise across GPUs, r is local but c must be all-reduced across the tensor-parallel group. This is a tiny communication (O(n) per layer) — negligible compared to gradient all-reduce.
- **Pipeline Parallel:** Each stage runs CUM independently. Power iteration and NS are purely local operations. No cross-stage synchronization needed.

**Key advantage over SOAP:** SOAP's Kronecker factors (m×m and n×n) are expensive to synchronize in tensor-parallel settings. CUM's factored statistics (m + n) are cheap.

---

## 6. Time Complexity Analysis

### 6.1 Per-Step Cost Breakdown

For a single weight matrix W ∈ ℝᵐˣⁿ, let k = min(m, n) and p = max(m, n):

| Phase | Operation | FLOPs | Memory R/W |
|-------|-----------|-------|------------|
| **Momentum** | m ← β₁m + (1-β₁)g | 3mn | 3mn (r: g, m; w: m) |
| | u = g + β₁·m | 2mn | 2mn |
| **Precond** | row/col sums of g² | 2mn | mn (if fused w/ momentum) |
| | bias correction + sqrt + scale | O(m+n) | O(m+n) (negligible) |
| | ũ = D_r · u · D_c | 2mn | 2mn (if fused: 0 extra) |
| **NS (×3)** | A = X·Xᵀ | 3 × 2k²p | 3 × (2kp + k²) |
| | A² | 3 × 2k³ | 3 × 2k² |
| | B·X | 3 × 2k²p | 3 × (k² + kp) |
| **Spectral** | W@v | 2mn | mn + n |
| | Wᵀ@(W@v) | 2mn | mn + m |
| | norm + damping | O(n) | O(n) |
| **Update** | W -= η·damp·scale·orth + λW | 4mn | 2mn |

**Total FLOPs (dominant terms):**

```
CUM:  12k²p + 6k³ + O(mn)
Muon: 20k²p + 10k³ + O(mn)
```

For square matrices (m = n = d):
```
CUM:  18d³ + O(d²)
Muon: 30d³ + O(d²)
```

**CUM is 1.67× cheaper per step than Muon in the dominant term.**

### 6.2 Comparison Table (Square d×d Matrices)

| Optimizer | FLOPs (dominant) | Memory State | Extra memory R/W per step |
|-----------|-----------------|--------------|--------------------------|
| SGD+Momentum | O(d²) | d² | 3d² |
| AdamW | O(d²) | 2d² | 5d² |
| Muon | 30d³ | d² | ~30d² (NS matmuls) |
| MuonClip | 30d³ + O(seq²·heads) | d² | ~30d² + clip check |
| SOAP (freq=10) | 30d³/10 + d³ (eigenupdate) | d² + 2d² | varies |
| **CUM** | **18d³** | **d² + O(d)** | **~18d² + 4d² (spectral)** |

### 6.3 Wall-Clock Projections

Assuming A100-80GB, bf16, 24-layer transformer with d_model = 4096:

Each hidden layer has multiple 2D weights (Q, K, V, O projections + MLP up/down). Typical count: ~8 weight matrices per layer × 24 layers = 192 2D params.

Per-step NS overhead estimate:
- Muon: 192 × 15 matmuls × 0.44ms = **1267ms**
- CUM: 192 × 9 matmuls × 0.44ms = **760ms**
- **Savings: ~507ms per step (~40%)**

Total optimizer step time (including non-NS work):
- Muon: ~1400ms
- CUM: ~950ms (includes preconditioning + spectral control overhead)
- **Net savings: ~32% faster optimizer step**

Note: Optimizer time is typically 15-25% of total step time (forward + backward + optimizer). A 32% optimizer speedup translates to **5-8% total training speedup** from compute alone. Combined with fewer steps to convergence (from curvature information), the total speedup should be larger.

### 6.4 Tier 0 Wall-Clock Projections (M3 CPU)

On Apple M3, CPU, fp32, single-threaded matmul (or 4-thread via Accelerate):

For the Tier 0c micro-transformer (d_model=128, 4 layers, 24 eligible 2D weight matrices):

Largest weight matrix is 512×128 (MLP up-projection). For a 128×128 matrix:

- Each NS matmul (128×128): ~0.008ms on M3 CPU (Accelerate BLAS)
- Muon NS per layer: 5 × 3 matmuls × 0.008ms = **0.12ms**
- CUM NS per layer: 3 × 3 matmuls × 0.008ms = **0.072ms**

For all 24 layers:
- Muon NS total: 24 × 0.12ms = **2.9ms**
- CUM NS total: 24 × 0.072ms = **1.7ms**
- CUM preconditioning: 24 × ~0.05ms = **1.2ms** (row/col sums + scaling)
- CUM spectral control: 24 × ~0.02ms = **0.5ms** (power iteration on small matrices)

**Expected per-step optimizer time:**
- Muon: ~5ms (NS + momentum + update)
- CUM: ~5.5ms (NS + momentum + precond + spectral + update)

At this tiny scale, the NS savings (~1.2ms) roughly cancel out the preconditioning + spectral overhead (~1.7ms). **CUM will not be faster per-step at Tier 0 scale.** That's expected and fine — the compute advantage only materializes for d ≥ 512 where NS matmuls dominate. At Tier 0, we're purely testing **convergence quality** (fewer steps to target), not per-step speed.

**Expected total training time for Tier 0c (5000 steps, micro-transformer):**
- Forward + backward: ~15ms per step (dominated by attention + MLP on 128-dim)
- Optimizer step: ~5ms
- Total per step: ~20ms
- Total 5000 steps: **~100 seconds ≈ 1.7 minutes**
- With LR sweep (5 LRs × 4 optimizers): **~34 minutes**

This is well within the 20-minute-per-run budget. Even with 3 seeds per config it stays under 2 hours total.

---

## 7. Benchmark Suite Design

### 7.1 Models & Datasets

Four evaluation tiers, increasing in scale. **Start at Tier 0. Do not move to higher tiers until Tier 0 shows clear wins.** Tier 0 is designed for a M3 MacBook with 16GB RAM (≤2GB available for training), CPU-only, fp32, with a hard wall-clock cap of 20 minutes per run.

---

#### Tier 0: Broke Student Mode (M3 Mac, CPU, <20 min)

The entire point of this tier is to answer one question: **does CUM converge faster per step than the baselines on tiny models?** If yes, the idea works and we graduate to GPU tiers. If no, we go back to the drawing board before wasting real compute.

**Hardware constraints:**
- Apple M3, 16GB unified RAM, ~2GB available for training
- CPU only (no MPS — MPS has bad support for custom matmuls and torch.compile)
- fp32 (no mixed precision on CPU)
- Max 20 minutes wall-clock per single training run
- Max ~500MB peak RAM for the entire training process

**Tier 0a: Synthetic Quadratic (Optimizer Unit Test)**

This isn't even a neural network — it's a pure optimization problem that tests whether the optimizer mechanics work.

- **Problem:** Minimize f(W) = ‖AW - B‖²_F where A ∈ ℝ⁶⁴ˣ³², W ∈ ℝ³²ˣ³², B ∈ ℝ⁶⁴ˣ³²
- **A and B:** Fixed random matrices (seeded). A has a known condition number (set via SVD construction with σ_max/σ_min = 100)
- **Target:** Loss < 1e-6
- **Metric:** Steps to target
- **Time:** <10 seconds per run
- **Purpose:** Verifies that CUM's preconditioning helps on ill-conditioned problems where Muon wastes steps exploring noise directions. This is the **most controlled** test — no stochasticity from data sampling, no architecture confounds. If CUM doesn't win here, something is fundamentally wrong.
- **What to look for:** CUM should reach target in noticeably fewer steps than Muon. Adam should also be fast (it handles ill-conditioning well). SGD should be slowest. If Muon and CUM are identical, the preconditioning is doing nothing.

**Tier 0b: Tiny MLP on MNIST (Sanity Check)**

- **Model:** 3-layer MLP: 784 → 128 → 64 → 10 (total ~110K params)
  - 2 hidden weight matrices eligible for CUM/Muon: 784×128 and 128×64
  - Biases + output head → AdamW
- **Dataset:** MNIST (ships with torchvision, no download hassle)
- **Batch size:** 256
- **Training:** 2000 steps (~8.5 epochs)
- **Target:** 97.5% test accuracy
- **Metric:** Steps to target, final test accuracy at 2000 steps
- **Time:** ~1-2 minutes per run
- **Purpose:** Sanity check that CUM works on an actual neural network. MNIST is trivial enough that all optimizers should converge — the question is how fast. The 784×128 weight matrix is rectangular enough to stress-test NS on non-square matrices.
- **What to look for:** CUM should match or beat Muon in steps-to-target. If CUM is significantly worse, check NS convergence error — 3 steps might not be enough without sufficient preconditioning signal on this tiny model.

**Tier 0c: Micro-Transformer on TinyShakespeare (Core Benchmark)**

This is the real test for Tier 0. A legitimate (but tiny) transformer trained on actual text.

- **Model:** GPT-style transformer
  - d_model = 128
  - n_heads = 4 (d_head = 32)
  - n_layers = 4
  - d_ff = 512 (MLP hidden dim)
  - context length = 256
  - Total params: ~1.2M
  - 2D hidden weights eligible for CUM: Q/K/V/O projections (128×128 each × 4 layers = 16 matrices) + MLP up/down (128×512, 512×128 × 4 layers = 8 matrices) = 24 matrices
- **Dataset:** TinyShakespeare (1.1MB text file, ~1M characters). Train/val split: 90/10.
- **Batch size:** 32
- **Training:** 5000 steps
- **Target:** val loss < 1.50 (achievable for this size)
- **Metric:** Val loss vs steps, val loss vs wall-clock, steps to target, final val loss
- **Time:** ~10-15 minutes per run
- **LR sweep:** {0.005, 0.01, 0.02, 0.05, 0.1} — 5 runs per optimizer, ~1 hour total per optimizer
- **Purpose:** This is the minimum viable transformer test. It has actual attention heads (so QK stability matters), actual MLP layers, and real text data. If CUM shows faster convergence here, the thesis holds and we scale up.
- **What to look for:**
  - CUM should converge to target loss in fewer steps than Muon
  - CUM's wall-clock per step should be comparable or faster (fewer NS steps should offset preconditioning cost even on CPU, though CPU matmul overhead is different from GPU)
  - Monitor NS convergence error: verify ‖XXᵀ - I‖_F < 0.05 after 3 preconditioned steps
  - Monitor σ_max/σ_min of momentum before/after preconditioning to verify spectrum compression

**Tier 0d: CIFAR-10 Tiny ConvNet (Cross-Architecture Validation)**

- **Model:** Simple ConvNet
  - Conv2d(3, 32, 3) → Conv2d(32, 64, 3) → Conv2d(64, 64, 3) → FC(1024, 10)
  - Total params: ~85K
  - CUM applied to: Conv layers (flatten last 3 dims to make 2D, as Muon recommends) + FC layer
- **Dataset:** CIFAR-10 (torchvision)
- **Batch size:** 128
- **Training:** 3000 steps (~6.5 epochs)
- **Target:** 70% test accuracy (modest for this architecture)
- **Metric:** Steps to target, final test accuracy
- **Time:** ~3-5 minutes per run
- **Purpose:** Verify CUM works beyond transformers. ConvNet weight matrices are often very rectangular (e.g., 64×576 when flattened), which tests NS on different aspect ratios.

**Tier 0 Success Criteria (MUST pass before moving to Tier 1+):**

| Test | Pass Condition | Fail Action |
|------|---------------|-------------|
| 0a: Synthetic | CUM reaches target in ≤80% of Muon's steps | Debug preconditioning — it should dominate here |
| 0b: MNIST MLP | CUM reaches 97.5% in ≤ Muon's steps | Check NS convergence at 3 steps |
| 0c: Micro-Transformer | CUM val loss curve is below Muon's at every checkpoint | **This is the key gate.** If CUM loses here, do not proceed. |
| 0c: Wall-clock | CUM wall-clock per step ≤ 1.1× Muon's | If >1.1×, profile — preconditioning or spectral control is too expensive |
| 0d: CIFAR ConvNet | CUM reaches 70% in ≤ Muon's steps | Check rectangular matrix handling |

**Tier 0 Optimizers (simplified set — don't waste time on SOAP/MuonClip at this scale):**

| Optimizer | Notes |
|-----------|-------|
| AdamW | Universal baseline |
| Muon | Primary comparison target |
| CUM | Ours |
| SGD+Nesterov | Sanity floor (should be worst) |

**Tier 0 Mac-Specific Notes:**

- **No `torch.compile`**: torch.compile has limited/buggy support on macOS CPU backend. Use eager mode. This means no kernel fusion — that's fine, we're not measuring absolute throughput, we're measuring convergence per step.
- **No MPS**: MPS (Metal Performance Shaders) has incomplete op coverage for custom matmuls and will silently fall back to CPU for some ops, giving misleading timing numbers. Stick to CPU.
- **No Triton**: Triton doesn't support macOS. All fused kernel work is deferred to GPU tiers.
- **Use `torch.set_num_threads(4)`**: M3 has efficiency and performance cores. Limiting to 4 threads avoids thrashing and gives more consistent timing.
- **Memory monitoring**: Use `tracemalloc` (Python) or `resource.getrusage` (POSIX) instead of `torch.cuda.memory_allocated`. Track RSS (Resident Set Size) to make sure you're under the ~2GB budget.
- **Timing**: Use `time.perf_counter()` for wall-clock. Log per-step times to detect any memory/GC spikes.

**Tier 0 Minimal Logging:**

Don't bother with wandb at this scale. Log to CSV:

```python
# tier0_metrics.csv columns:
# step, train_loss, val_loss, val_acc (if classification), 
# step_time_ms, optimizer_time_ms, ns_convergence_error,
# sv_spread_before, sv_spread_after, sigma_max_layer0, damping_layer0,
# rss_mb
```

Plot locally with matplotlib. The plotting scripts from Section 8 work, just point them at the CSV.

---

#### Tier 1: CIFAR-10 ConvNet — GPU (Rapid Iteration)

*Requires Tier 0 pass.*

- **Model:** Small ConvNet (e.g., the CIFAR-10 speedrun architecture from modded-nanogpt)
- **Target:** 94% accuracy
- **Metric:** A100-seconds to target
- **Purpose:** Fast turnaround for hyperparameter sweeps, sanity checks
- **Training tokens:** ~50M (CIFAR images treated as sequences)

#### Tier 2: NanoGPT on FineWeb (Primary Benchmark)

*Requires Tier 1 pass.*

- **Model:** GPT-2 124M parameter transformer
- **Target:** val loss 3.28 (standard NanoGPT speedrun target)
- **Metric:** Steps to target, wall-clock time to target, final val loss at fixed step budget
- **Purpose:** Direct comparison with Muon's published results
- **Dataset:** FineWeb (or OpenWebText)

#### Tier 3: LLaMA-350M on C4 (Scale Validation)

*Requires Tier 2 pass.*

- **Model:** LLaMA 350M architecture
- **Target:** Best val perplexity at 5000 steps
- **Metric:** Token efficiency (val loss vs. tokens processed), memory usage, throughput
- **Purpose:** Validate scaling behavior, compare with COSMOS/SOAP published numbers
- **Dataset:** C4

### 7.2 Optimizers to Compare

Each optimizer should be tuned with a consistent protocol. For each optimizer, sweep learning rate at resolution √[4]{10} ≈ {0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3}:

| Optimizer | Implementation | Key Hyperparameters |
|-----------|---------------|---------------------|
| **AdamW** (baseline) | `torch.optim.AdamW` | lr, β₁=0.9, β₂=0.95, wd=0.01 |
| **SGD+Nesterov** (baseline) | `torch.optim.SGD` | lr, momentum=0.9, wd=0.01 |
| **Muon** (primary comparison) | Keller Jordan's official repo | lr, β₁=0.95, wd=0.01, ns_steps=5 |
| **MuonClip** | Kimi K2 implementation | lr, β₁=0.95, wd=0.01, ns_steps=5, τ=100 |
| **CUM** (ours) | Custom implementation | lr, β₁=0.95, β₂=0.99, wd=0.01, ns_steps=3, σ_max=30, α_damp=0.1 |
| **SOAP** (if memory permits) | Official implementation | lr, precond_freq=10 |

**Fairness rules:**
1. Same model architecture, initialization seed, and data order for all optimizers
2. Same LR schedule (linear warmup + cosine decay)
3. Same batch size
4. Each optimizer gets a fair LR sweep (minimum 5 runs with different LR)
5. Report best run AND mean of top-3 runs (to measure robustness)
6. For hybrid optimizers (Muon, CUM), AdamW is used for non-hidden params with its own tuned LR

### 7.3 Metrics to Track

**Per step (logged every N steps):**

```python
metrics = {
    # Loss & accuracy
    "train_loss": float,
    "val_loss": float,
    "val_perplexity": float,  # exp(val_loss) for LM
    
    # Convergence speed
    "steps_to_target": int,   # first step where val_loss < target
    "tokens_processed": int,
    
    # Compute efficiency
    "wall_clock_seconds": float,
    "optimizer_step_ms": float,       # just the optimizer step
    "ns_time_ms": float,              # just Newton-Schulz
    "precond_time_ms": float,         # just preconditioning (CUM only)
    "spectral_control_time_ms": float, # just power iter + damping (CUM only)
    
    # Memory
    "gpu_memory_allocated_gb": float,
    "gpu_memory_reserved_gb": float,
    "peak_memory_gb": float,
    
    # Optimizer diagnostics
    "grad_norm_global": float,
    "update_norm_global": float,
    "effective_lr_per_layer": list,    # η × damping for CUM
    
    # CUM-specific
    "sigma_max_per_layer": list,       # spectral norm estimates
    "damping_per_layer": list,         # damping factors
    "ns_convergence_error": float,     # ‖XXᵀ - I‖ after NS (verify 3 steps suffice)
    "sv_spread_before_precond": float, # σ_max/σ_min of momentum (sample layers)
    "sv_spread_after_precond": float,  # σ_max/σ_min after preconditioning
    
    # Stability (for MuonClip comparison)
    "max_qk_score": float,            # max attention logit across all heads
    "clip_events_this_step": int,      # how many times MuonClip triggered
}
```

### 7.4 Ablation Studies

To validate each CUM component independently:

| Ablation | What it tests | Expected result |
|----------|---------------|-----------------|
| CUM without preconditioning (ns=3) | Does preconditioning enable fewer NS steps? | Should fail — 3 NS steps without preconditioning won't converge |
| CUM without preconditioning (ns=5) | Is preconditioning helping convergence quality, not just speed? | Should perform similar to Muon (same algorithm) |
| CUM with ns=5 (overkill NS) | Is ns=3 actually sufficient with preconditioning? | Should match CUM ns=3 within noise (confirming 3 steps suffice) |
| CUM without spectral control | Is spectral control needed? | Should be fine for small models, but check σ_max growth |
| CUM with hard clip instead of smooth damping | Smooth vs hard | Smooth should give slightly better loss trajectory |
| CUM with lagged vs current precond stats | Does the lag approximation matter? | Should be indistinguishable |
| CUM with different β₂ values | Sensitivity to second-moment EMA rate | 0.99 should be robust; 0.9 and 0.999 should also work |

---

## 8. Graphing & Evaluation Pipeline

### 8.1 Required Plots

#### Plot Set 1: Convergence Comparison (Primary Results)

**Plot 1a: Val Loss vs Steps**
- X-axis: training steps (log scale optional)
- Y-axis: validation loss
- Lines: AdamW, Muon, MuonClip, CUM (+ SOAP if available)
- Shaded regions: ±1 std across 3 seeds
- Horizontal dashed line at target loss (e.g., 3.28 for NanoGPT)
- Purpose: Shows convergence speed in terms of steps

**Plot 1b: Val Loss vs Wall-Clock Time**
- Same as 1a but X-axis is wall-clock seconds
- Purpose: Shows real-world speed advantage (accounts for per-step cost)

**Plot 1c: Val Loss vs Tokens Processed**
- X-axis: total tokens processed
- Purpose: Shows token efficiency (independent of batch size)

**Plot 1d: Steps to Target Bar Chart**
- Bar chart with 95% CI error bars
- One bar per optimizer
- Purpose: Clean summary metric

#### Plot Set 2: Compute Efficiency

**Plot 2a: Optimizer Step Time Breakdown (Stacked Bar)**
- For each optimizer, stacked bars showing:
  - Momentum computation time
  - Preconditioning time (CUM only)
  - Newton-Schulz time
  - Spectral control time (CUM only)
  - Weight update time
  - QK-clip time (MuonClip only)
- Purpose: Shows where time is spent, validates NS savings

**Plot 2b: Throughput (Tokens/Second) vs Training Step**
- X-axis: step number
- Y-axis: tokens/second
- Purpose: Ensure no throughput degradation over time

**Plot 2c: GPU Memory Usage Bar Chart**
- Peak memory per optimizer
- Breakdown: params, gradients, optimizer state, activations
- Purpose: Validate CUM's memory efficiency claim

#### Plot Set 3: Stability Analysis

**Plot 3a: Max QK Attention Score vs Step**
- X-axis: step
- Y-axis: max(QK scores) across all heads/layers
- Lines: Muon, MuonClip, CUM
- Purpose: Show CUM's preventive stability vs MuonClip's reactive clipping

**Plot 3b: Max Spectral Norm (σ_max) vs Step (Per Layer)**
- X-axis: step
- Y-axis: σ_max
- Subplot per layer (or heatmap: layers × steps → σ_max)
- Lines: Muon vs CUM
- Purpose: Show spectral control working

**Plot 3c: CUM Damping Factor vs Step (Per Layer)**
- Shows when/where damping activates
- Purpose: Validate smooth damping behavior (no sudden jumps)

**Plot 3d: MuonClip Clip Events vs Step**
- Histogram or cumulative count
- Purpose: Quantify how often MuonClip needs to intervene

#### Plot Set 4: CUM Diagnostics (Ablation Support)

**Plot 4a: NS Convergence Error vs NS Steps**
- For CUM (preconditioned) and Muon (raw): measure ‖XXᵀ - I‖_F after k steps
- Subplots at step 100, 1000, 5000 of training
- Purpose: Validate that 3 preconditioned NS steps ≈ 5 raw NS steps

**Plot 4b: Singular Value Spread Before/After Preconditioning**
- For sample layers, plot σ_max/σ_min of momentum before and after precond
- Purpose: Validate that preconditioning compresses the spectrum

**Plot 4c: Ablation Results Grid**
- Table-as-heatmap showing final val loss for each ablation variant
- Purpose: Summary of component contributions

#### Plot Set 5: Scaling (Tier 3 Only)

**Plot 5a: Val Perplexity vs Model Size**
- X-axis: parameter count
- Y-axis: best val perplexity at fixed token budget
- Lines: Each optimizer
- Purpose: Scaling behavior

### 8.2 Plotting Implementation

Use matplotlib + seaborn. Log all metrics to either:

- **Weights & Biases** (preferred, use `wandb.log()`)
- **TensorBoard** (alternative, use `SummaryWriter`)
- **CSV fallback** (always log to CSV as backup)

```python
# Recommended logging setup
import wandb

wandb.init(
    project="cum-optimizer-benchmarks",
    config={
        "optimizer": "CUM",
        "model": "nanogpt-124m",
        "dataset": "fineweb",
        "lr": 0.02,
        # ... all hyperparameters
    },
    tags=["benchmark", "tier2"],
)

# Per step
wandb.log({
    "train/loss": train_loss,
    "val/loss": val_loss,
    "optimizer/step_time_ms": step_time,
    "optimizer/ns_time_ms": ns_time,
    "stability/max_qk_score": max_qk,
    "cum/damping_mean": np.mean(damping_factors),
    "cum/sigma_max_mean": np.mean(sigma_maxes),
}, step=global_step)
```

Plotting script structure:

```
benchmarks/
├── run_benchmark.py          # Main training loop with metric logging
├── sweep_lr.py               # LR sweep launcher
├── configs/
│   ├── nanogpt_124m.yaml
│   ├── cifar10_convnet.yaml
│   └── llama_350m.yaml
├── plotting/
│   ├── convergence_plots.py  # Plot set 1
│   ├── compute_plots.py      # Plot set 2
│   ├── stability_plots.py    # Plot set 3
│   ├── diagnostic_plots.py   # Plot set 4
│   ├── scaling_plots.py      # Plot set 5
│   └── style.py              # Consistent matplotlib style
└── results/
    ├── raw/                  # CSV metric dumps
    └── figures/              # Generated PNGs/PDFs
```

### 8.3 Plotting Style Guide

Consistent visual style across all plots:

```python
# style.py
OPTIMIZER_COLORS = {
    "AdamW": "#1f77b4",       # blue
    "SGD+Nesterov": "#7f7f7f", # gray
    "Muon": "#ff7f0e",        # orange
    "MuonClip": "#d62728",    # red
    "CUM": "#2ca02c",         # green (ours — should stand out)
    "SOAP": "#9467bd",        # purple
}

OPTIMIZER_LINESTYLES = {
    "AdamW": "--",
    "SGD+Nesterov": ":",
    "Muon": "-.",
    "MuonClip": "-.",
    "CUM": "-",               # solid (ours)
    "SOAP": "--",
}

# Standard figure setup
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
```

---

## 9. Known Risks & Mitigations

### Risk 1: 3 NS Steps May Not Suffice

**Severity:** High  
**Likelihood:** Medium  
**Symptom:** Training diverges or converges slower despite preconditioning.

**Detection:** Monitor ‖XXᵀ - I‖_F after NS. Should be < 0.01. If > 0.1, insufficient iterations.

**Mitigation:** 
- Verify convergence error during initial runs
- If 3 steps are insufficient, try 4 (still saves 20% vs Muon)
- Investigate whether NS coefficient tuning for preconditioned inputs helps
- As a fallback, the lagged preconditioning might not compress the spectrum enough — check Plot 4b

### Risk 2: Factored Approximation Too Coarse

**Severity:** Medium  
**Likelihood:** Low-Medium  
**Symptom:** CUM converges to worse final loss than Muon despite faster early convergence.

**Explanation:** Row/col norms lose off-diagonal structure of the gradient covariance. For layers where the gradient has strong cross-row/col correlations, the factored preconditioner is a poor approximation.

**Detection:** Compare final val loss (not just convergence speed) between CUM and Muon at high step counts.

**Mitigation:**
- This is a known limitation of all factored methods (Adafactor has the same issue)
- If severe: add a low-rank correction (rank 4-8 SVD of gradient, very cheap) to capture the top cross-correlations
- If mild: the convergence speed advantage may still make CUM Pareto-optimal even with slightly worse final loss

### Risk 3: Spectral Control Interferes with Learning

**Severity:** Medium  
**Likelihood:** Low  
**Symptom:** Layers near the spectral ceiling learn slower than they should.

**Detection:** Plot per-layer loss contribution or gradient norms. Check if damped layers have systematically higher gradient norms (suggesting they "want" to update more but can't).

**Mitigation:**
- Set σ_max conservatively high (30-50) so it rarely activates during normal training
- α_damp = 0.1 means even at σ_est = 2×σ_max, damping only reduces LR by ~75%. This is gentle.
- If problematic: make σ_max per-layer adaptive (EMA of each layer's historical σ_max)

### Risk 4: Power Iteration Doesn't Converge for Non-Square Matrices

**Severity:** Low  
**Likelihood:** Low  
**Symptom:** σ_est oscillates wildly between steps.

**Explanation:** One power iteration step per optimizer step may not converge for highly rectangular matrices (e.g., MLP up-projection where n = 4m).

**Detection:** Plot σ_est vs step. Should be smooth. If jagged, power iteration isn't converging.

**Mitigation:**
- Run 2-3 power iterations instead of 1 (still cheap: O(mn) per extra iteration)
- Or just use σ_est from 2 steps ago as a running EMA: σ̂ = 0.9·σ̂ + 0.1·σ_est

### Risk 5: Interaction Between Preconditioning and Weight Decay

**Severity:** Low  
**Likelihood:** Medium  
**Symptom:** Weight matrices shrink too much or too little compared to Muon.

**Explanation:** Muon's weight decay (W -= λW) interacts with the orthogonal update to implicitly constrain σ_max(W) ≤ 1/λ. CUM's preconditioning changes the effective update direction, which might alter this implicit constraint.

**Detection:** Compare weight matrix norms and σ_max trajectories between Muon and CUM.

**Mitigation:**
- If weights shrink too much: reduce weight decay for CUM (it has explicit spectral control, so it needs less implicit regularization from WD)
- If weights grow too much: spectral control should catch this, but may need to tune σ_max down

---

## 10. Hyperparameter Guide

### 10.1 Default Configuration

```yaml
# Recommended starting point for NanoGPT-124M
cum:
  lr: 0.02                # Same as Muon default
  beta1: 0.95             # Momentum coefficient
  beta2: 0.99             # Second moment EMA (same as Adam)
  weight_decay: 0.01      # May need to be lower than Muon (see Risk 5)
  ns_steps: 3             # Down from Muon's 5
  eps: 1.0e-7             # Numerical stability
  sigma_max: 30.0         # Soft spectral ceiling (only matters at scale)
  alpha_damp: 0.1         # Spectral damping strength
  nesterov: true          # Use Nesterov momentum

aux_adam:
  lr: 3.0e-4              # For embeddings, biases, norms, output head
  betas: [0.9, 0.95]
  weight_decay: 0.01
```

### 10.2 Hyperparameter Sensitivity Expectations

| Param | Sensitivity | Range to Sweep | Notes |
|-------|------------|----------------|-------|
| lr | **High** | [0.005, 0.1] at √[4]{10} resolution | Most important HP. Start at 0.02 (Muon's default). |
| beta1 | Low | {0.9, 0.95} | 0.95 should work. Only sweep if lr is tuned. |
| beta2 | Low | {0.9, 0.99, 0.999} | Controls preconditioning adaptation speed. 0.99 is robust default. |
| weight_decay | Medium | {0.001, 0.01, 0.1} | May interact with spectral control — sweep after lr. |
| ns_steps | **Low** (if precond works) | {3, 4, 5} | Verify 3 works via convergence error; don't tune unless needed. |
| sigma_max | Low (at NanoGPT scale) | {10, 30, 100} | Only matters if you see instability. 30 is conservative. |
| alpha_damp | Low | {0.05, 0.1, 0.2} | 0.1 is gentle. Only tune if damping is activating too aggressively. |

### 10.3 Scaling Heuristics

When moving to larger models:

- **lr:** Muon's lr scales roughly as `0.02 × (d_base/d_model)^0.5` across model widths. CUM should follow the same scaling due to the shared orthogonalization principle.
- **sigma_max:** May need to increase with model depth. Heuristic: `sigma_max = 30 × sqrt(num_layers / 24)`.
- **alpha_damp:** Keep fixed unless damping is causing problems.
- **ns_steps:** Keep at 3. If anything, preconditioning should work *better* at scale because the gradient covariance has more structure for the factored preconditioner to capture.

---

## 11. File Structure & Build Instructions

### 11.1 Repository Layout

```
cum-optimizer/
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
│
├── cum/
│   ├── __init__.py
│   ├── cum.py                    # CUM optimizer class
│   ├── hybrid.py                 # CUMWithAuxAdam wrapper
│   ├── newton_schulz.py          # NS iteration (torch.compile compatible)
│   ├── factored_precond.py       # Factored preconditioning
│   ├── spectral_control.py       # Power iteration + damping
│   ├── utils.py                  # Aspect ratio scaling, gradient stats
│   └── triton_kernels/           # Optional fused kernels
│       ├── __init__.py
│       ├── fused_momentum_precond.py
│       └── fused_update.py
│
├── benchmarks/
│   ├── run_benchmark.py          # Main training entry point
│   ├── sweep_lr.py               # LR sweep via wandb or manual
│   ├── run_ablations.py          # Ablation study launcher
│   ├── tier0/                    # ← START HERE. Mac CPU, <20 min.
│   │   ├── run_all_tier0.py      # Runs 0a-0d sequentially, prints pass/fail
│   │   ├── synthetic_quadratic.py # Tier 0a: ill-conditioned W optimization
│   │   ├── mnist_mlp.py          # Tier 0b: 3-layer MLP on MNIST
│   │   ├── micro_transformer.py  # Tier 0c: 1.2M param GPT on TinyShakespeare
│   │   ├── cifar10_tiny_conv.py  # Tier 0d: small ConvNet on CIFAR-10
│   │   ├── data/
│   │   │   └── download_tinyshakespeare.py  # wget the 1.1MB text file
│   │   └── plot_tier0.py         # Generate all Tier 0 comparison plots
│   ├── configs/
│   │   ├── cifar10_convnet.yaml
│   │   ├── nanogpt_124m.yaml
│   │   ├── nanogpt_350m.yaml
│   │   └── llama_350m_c4.yaml
│   ├── models/
│   │   ├── convnet.py
│   │   ├── nanogpt.py
│   │   └── llama.py
│   ├── data/
│   │   ├── cifar10.py
│   │   ├── fineweb.py
│   │   └── c4.py
│   └── baselines/
│       ├── adamw_baseline.py
│       ├── muon_baseline.py
│       ├── muonclip_baseline.py
│       └── soap_baseline.py
│
├── evaluation/
│   ├── plotting/
│   │   ├── convergence_plots.py
│   │   ├── compute_plots.py
│   │   ├── stability_plots.py
│   │   ├── diagnostic_plots.py
│   │   ├── scaling_plots.py
│   │   └── style.py
│   ├── analysis/
│   │   ├── ns_convergence.py     # Verify NS step reduction
│   │   ├── sv_spread.py          # Measure singular value compression
│   │   └── memory_profile.py     # Peak memory analysis
│   └── results/
│       ├── raw/                  # CSV / wandb exports
│       └── figures/              # Generated plots
│
├── tests/
│   ├── test_tier0_cpu.py         # ← RUN FIRST. CPU smoke + NS convergence + precond + spectral
│   ├── test_cum.py               # Unit tests for optimizer (GPU)
│   ├── test_newton_schulz.py     # NS convergence tests
│   ├── test_factored_precond.py  # Preconditioning correctness
│   ├── test_spectral_control.py  # Power iteration + damping
│   ├── test_hybrid.py            # Hybrid optimizer integration
│   ├── test_numerical.py         # fp32/bf16 consistency
│   └── test_distributed.py       # Multi-GPU correctness
│
└── docs/
    ├── BUILD_GUIDE.md            # This document
    ├── THEORY.md                 # Mathematical derivations
    └── CHANGELOG.md
```

### 11.2 Dependencies

```toml
# pyproject.toml
[project]
name = "cum-optimizer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
benchmarks = [
    "wandb>=0.16",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]
triton = [
    "triton>=2.2.0",
]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
]
```

### 11.3 Build & Run

```bash
# Install (works on Mac M3, no CUDA required)
git clone https://github.com/yourname/cum-optimizer
cd cum-optimizer
pip install -e ".[benchmarks,dev]"

# ═══════════════════════════════════════════
# STEP 1: CPU smoke tests (< 60 seconds)
# Run these BEFORE anything else.
# ═══════════════════════════════════════════
pytest tests/test_tier0_cpu.py -v

# If any test fails, STOP. Fix the optimizer before benchmarking.
# The most important test is:
#   test_3_preconditioned_steps_converges
# If that fails, the core thesis is broken.

# ═══════════════════════════════════════════
# STEP 2: Tier 0 benchmarks (Mac CPU, ~1-2 hours total)
# ═══════════════════════════════════════════

# Download TinyShakespeare (1.1MB)
python benchmarks/tier0/data/download_tinyshakespeare.py

# Run all Tier 0 benchmarks with LR sweeps for all optimizers
# This runs 0a, 0b, 0c, 0d sequentially and prints a pass/fail summary
python benchmarks/tier0/run_all_tier0.py

# Or run individually:
python benchmarks/tier0/synthetic_quadratic.py    # ~30 seconds
python benchmarks/tier0/mnist_mlp.py              # ~10 minutes (all optimizers × LR sweep)
python benchmarks/tier0/micro_transformer.py      # ~1 hour (all optimizers × LR sweep)
python benchmarks/tier0/cifar10_tiny_conv.py      # ~20 minutes

# Generate Tier 0 plots
python benchmarks/tier0/plot_tier0.py --results-dir benchmarks/tier0/results/

# ═══════════════════════════════════════════
# STEP 3: Check Tier 0 results
# ═══════════════════════════════════════════
# Look at:
#   benchmarks/tier0/results/figures/convergence_0c.png  — CUM vs Muon loss curves
#   benchmarks/tier0/results/figures/ns_error_0c.png     — verify 3 steps suffice
#   benchmarks/tier0/results/figures/sv_spread_0c.png    — spectrum compression
#   benchmarks/tier0/results/tier0_summary.txt           — pass/fail table
#
# If Tier 0c PASSES (CUM curve below Muon curve), proceed to GPU tiers.
# If Tier 0c FAILS, diagnose using diagnostic plots before scaling up.

# ═══════════════════════════════════════════
# STEP 4: GPU tiers (requires CUDA GPU)
# Only proceed if Tier 0 passes.
# ═══════════════════════════════════════════

# Run full unit tests (includes GPU-specific tests)
pytest tests/ -v

# Tier 1: CIFAR-10 on GPU
python benchmarks/run_benchmark.py \
    --config benchmarks/configs/cifar10_convnet.yaml \
    --optimizer cum \
    --lr 0.02 \
    --seed 42

# Tier 2: NanoGPT LR sweep
for opt in adamw muon muonclip cum; do
    python benchmarks/sweep_lr.py \
        --config benchmarks/configs/nanogpt_124m.yaml \
        --optimizer $opt \
        --seeds 42 43 44
done

# Tier 3: LLaMA-350M (if you have the compute)
python benchmarks/run_benchmark.py \
    --config benchmarks/configs/llama_350m_c4.yaml \
    --optimizer cum

# Generate GPU-tier plots
python evaluation/plotting/convergence_plots.py --results-dir evaluation/results/raw/
python evaluation/plotting/compute_plots.py --results-dir evaluation/results/raw/
python evaluation/plotting/stability_plots.py --results-dir evaluation/results/raw/

# Run ablations (Tier 2 model, needs GPU)
python benchmarks/run_ablations.py --config benchmarks/configs/nanogpt_124m.yaml
```

### 11.4 Unit Test Specifications

#### Tier 0 Tests (CPU, Mac-compatible, run first)

These tests run in under 60 seconds total on M3 CPU. They validate correctness before any benchmarking.

```python
# tests/test_tier0_cpu.py — must all pass before ANY benchmark runs

class TestCUMCPUSmoke:
    """Ultra-fast smoke tests. Each test < 2 seconds."""
    
    def test_single_step_doesnt_crash(self):
        """Create 32×32 param, run 1 CUM step, verify no errors."""
    
    def test_output_shape_preserved(self):
        """W stays (m, n) after optimizer step."""
    
    def test_loss_decreases_on_quadratic(self):
        """On f(W) = ‖W - W*‖², loss after 10 steps < loss at step 0."""
    
    def test_nan_free(self):
        """No NaN in W, momentum, r, c, v after 50 steps."""
    
    def test_zero_grad_safe(self):
        """Zero gradient doesn't produce NaN (eps protection)."""
    
    def test_state_dict_roundtrip(self):
        """optimizer.state_dict() → load_state_dict() preserves state."""


class TestNSConvergenceOnCPU:
    """Validate the core NS claim: 3 preconditioned steps ≈ 5 raw steps."""
    
    def test_5_raw_steps_converges(self):
        """Standard Muon: 5 NS steps on raw momentum → ‖XXᵀ - I‖ < 0.01."""
    
    def test_3_raw_steps_insufficient(self):
        """3 NS steps on raw momentum → ‖XXᵀ - I‖ > 0.05 (expected to fail)."""
    
    def test_3_preconditioned_steps_converges(self):
        """CUM: 3 NS steps on preconditioned momentum → ‖XXᵀ - I‖ < 0.02.
        THIS IS THE KEY TEST. If this fails, the entire compute savings thesis is dead."""
    
    def test_spectrum_compression(self):
        """σ_max/σ_min of preconditioned momentum < raw momentum.
        Run on 10 random matrices, verify compression in ≥8/10 cases."""
    
    def test_rectangular_32x128(self):
        """NS converges for rectangular (32×128) matrix in 3 preconditioned steps."""
    
    def test_rectangular_128x32(self):
        """NS converges for tall (128×32) matrix in 3 preconditioned steps."""


class TestFactoredPrecondCPU:
    """Validate preconditioning math on CPU."""
    
    def test_row_var_matches_manual(self):
        """After 5 steps, row_var ≈ manual EMA of sum(g², dim=1)."""
    
    def test_col_var_matches_manual(self):
        """After 5 steps, col_var ≈ manual EMA of sum(g², dim=0)."""
    
    def test_bias_correction_step1(self):
        """At step 1, bias-corrected r̂ = g²_rows (no EMA effect yet)."""
    
    def test_high_var_rows_scaled_down(self):
        """Rows with 10× gradient magnitude get scaled ~3× down (1/√10)."""
    
    def test_precond_preserves_sign_structure(self):
        """sign(ũ[i,j]) == sign(u[i,j]) for all i,j. Preconditioning only scales."""


class TestSpectralControlCPU:
    """Validate spectral control on CPU."""
    
    def test_power_iter_converges_over_steps(self):
        """After 50 optimizer steps, σ_est within 5% of true σ_max (via torch.linalg.svdvals)."""
    
    def test_damping_is_1_below_threshold(self):
        """When σ_max(W) = 5 and σ_target = 30, damping = 1.0 exactly."""
    
    def test_damping_decreases_above_threshold(self):
        """When σ_max(W) = 60 and σ_target = 30, damping < 1.0."""
    
    def test_damping_is_smooth(self):
        """damping(σ=29.9) ≈ damping(σ=30.1). No discontinuity."""
    
    def test_damping_never_zero(self):
        """Even at σ_est = 1000, damping > 0."""


class TestTier0Benchmarks:
    """Functional tests that run the actual Tier 0 benchmarks (longer, ~5 min total)."""
    
    def test_synthetic_quadratic_cum_beats_muon(self):
        """Tier 0a: On ill-conditioned quadratic, CUM steps_to_target < Muon steps_to_target."""
    
    def test_mnist_mlp_cum_converges(self):
        """Tier 0b: CUM reaches 97.5% on MNIST within 2000 steps."""
    
    def test_micro_transformer_cum_matches_muon(self):
        """Tier 0c: After 500 steps on TinyShakespeare, CUM val_loss ≤ Muon val_loss.
        (Abbreviated version — full 5000-step run is in benchmarks/, not tests/)"""
    
    def test_memory_under_budget(self):
        """Tier 0c: Peak RSS during micro-transformer training < 1.5 GB."""
    
    def test_step_time_reasonable(self):
        """Tier 0c: Mean optimizer step time on micro-transformer < 50ms on M3 CPU."""
```

#### Full Test Suite (GPU, all tiers)

```python
# tests/test_cum.py — key test cases

class TestCUMOptimizer:
    
    def test_single_step_updates_weights(self):
        """Verify W changes after one optimizer step."""
    
    def test_momentum_accumulation(self):
        """Verify momentum buffer matches manual β₁ EMA computation."""
    
    def test_factored_precond_reduces_sv_spread(self):
        """σ_max/σ_min of preconditioned momentum < raw momentum."""
    
    def test_ns_convergence_3_steps(self):
        """After preconditioning + 3 NS steps, ‖XXᵀ - I‖_F < 0.01."""
    
    def test_ns_convergence_degrades_without_precond(self):
        """Without preconditioning, 3 NS steps give ‖XXᵀ - I‖_F > 0.05."""
    
    def test_spectral_damping_activates(self):
        """When σ_max(W) > σ_target, damping < 1."""
    
    def test_spectral_damping_inactive_below_threshold(self):
        """When σ_max(W) < σ_target, damping = 1.0."""
    
    def test_spectral_damping_smooth(self):
        """Damping is a continuous function of σ_est."""
    
    def test_weight_decay_applied(self):
        """W shrinks by factor (1-λ) per step independent of update."""
    
    def test_nesterov_vs_standard_momentum(self):
        """Nesterov momentum gives different update than standard."""
    
    def test_aspect_ratio_scaling(self):
        """scale = √(max(1, m/n)) for m×n weight."""
    
    def test_memory_overhead(self):
        """CUM state size = mn + m + 2n (+ constants)."""
    
    def test_bf16_consistency(self):
        """bf16 CUM step matches fp32 within tolerance."""
    
    def test_gradient_zero_handling(self):
        """Zero gradient doesn't cause NaN (eps protection)."""
    
    def test_very_rectangular_matrix(self):
        """Works for 4096×16384 (MLP up-projection shape)."""
    
    def test_hybrid_optimizer_separate_params(self):
        """CUM params and AdamW params are updated independently."""

class TestNewtonSchulz:
    
    def test_converges_to_polar_factor(self):
        """NS output matches torch.linalg.svd polar factor."""
    
    def test_output_is_approximately_orthogonal(self):
        """‖XXᵀ - I‖ < 0.01 after ns_steps iterations."""
    
    def test_deterministic(self):
        """Same input → same output across calls."""
    
    def test_compile_compatible(self):
        """Works under torch.compile(fullgraph=True)."""
    
    def test_gradient_flows(self):
        """Gradients propagate through NS (for any future use)."""

class TestFactoredPrecond:
    
    def test_row_var_correct(self):
        """row_var matches manual sum(g², dim=1) EMA."""
    
    def test_col_var_correct(self):
        """col_var matches manual sum(g², dim=0) EMA."""
    
    def test_bias_correction(self):
        """Bias-corrected estimates are unbiased at step 1."""
    
    def test_scaling_preserves_shape(self):
        """Preconditioned output has same shape as input."""
    
    def test_scaling_effect(self):
        """High-variance rows/cols get scaled down."""

class TestSpectralControl:
    
    def test_power_iteration_converges(self):
        """After many steps, σ_est ≈ true σ_max (via torch.linalg.svdvals)."""
    
    def test_damping_formula(self):
        """damping = 1/(1 + α·max(0, σ - σ_target))."""
    
    def test_damping_range(self):
        """0 < damping ≤ 1 for all valid inputs."""
```

---

## Appendix A: Quick Reference Card

```
╔═══════════════════════════════════════════════════════════╗
║                   CUM Quick Reference                     ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  What:     Muon + factored curvature + smooth spectral    ║
║            control                                        ║
║                                                           ║
║  Where:    2D hidden layer weights only                   ║
║            (everything else → AdamW)                      ║
║                                                           ║
║  Memory:   Same as Muon (+0.1%)                           ║
║  Compute:  ~40% less NS overhead than Muon                ║
║  Stability: Smooth per-layer damping (no hard clip)       ║
║                                                           ║
║  Key HPs:  lr=0.02, β₁=0.95, β₂=0.99, ns=3              ║
║            σ_max=30, α_damp=0.1                           ║
║                                                           ║
║  Pipeline:                                                ║
║    grad → momentum → precondition → NS(3) → damp → W     ║
║                                                           ║
║  Dev workflow:                                            ║
║    1. pytest test_tier0_cpu.py     (< 1 min, CPU)         ║
║    2. run_all_tier0.py             (< 2 hrs, Mac CPU)     ║
║    3. If 0c passes → GPU tiers     (needs CUDA)           ║
║    4. If 0c fails  → diagnose, don't scale up             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Appendix B: Theoretical Claims Summary

| Claim | Status | How to Verify |
|-------|--------|---------------|
| 3 NS steps suffice with preconditioning | **Hypothesis** | Test NS convergence error (Plot 4a) |
| Factored precond compresses SV spread | **Theoretical** + needs empirical | Measure σ_max/σ_min before/after (Plot 4b) |
| Smooth damping prevents QK explosion | **Hypothesis** | Compare max QK scores: CUM vs Muon (Plot 3a) |
| CUM converges faster than Muon per step | **Hypothesis** | Val loss vs steps (Plot 1a) |
| CUM is cheaper per step than Muon | **Theoretical** (fewer matmuls) | Wall-clock profiling (Plot 2a) |
| CUM matches/beats MuonClip stability | **Hypothesis** | Long training runs without loss spikes |
| Memory overhead is negligible | **Proven** | O(m+n) vs O(mn), verified in test_memory_overhead |

---

## Appendix C: Benchmark Results & Iteration Log

### Benchmark Setup

- **Hardware:** Apple M3 (CPU only, 4 threads)
- **Model:** MicroGPT (d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256, ~1.2M params)
- **Data:** TinyShakespeare (1.1M chars, vocab=65)
- **Training:** 2000 steps, batch_size=32, warmup=200 steps, cosine LR decay
- **Optimizer split:** Muon/CUM for hidden 2D weights, AdamW (lr=3e-4, wd=0.01) for embeddings/biases
- **Seed:** 42

### MNIST MLP Results (Tier 0b)

| Optimizer | Test Accuracy | Steps to 97% | Time |
|-----------|--------------|---------------|------|
| AdamW (lr=1e-3) | 98.18% | 500 | 8.5s |
| Muon (lr=0.02) | 98.33% | 500 | 16.6s |
| CUM v1 (lr=0.02) | 98.30% | 500 | 25.7s |
| CUMv2 a=0.3 | 98.41% | 500 | 17.5s |
| CUMv2a=0.5 | 98.17% | 500 | 20.0s |
| CUMv2a=0.7 | 98.39% | 500 | 19.5s |

### Micro-Transformer Results (Tier 0c) — Key Gate

| Optimizer | Val Loss | Time | Notes |
|-----------|----------|------|-------|
| AdamW (lr=1e-3) | 1.6046 | 718s | Baseline |
| **Muon (lr=0.02)** | **1.5198** | **668s** | **Target to beat** |
| CUM v1 (lr=0.02) | 1.5187 | 691s | Barely better, within noise |
| CUMv2a=0.5 | 1.5322 | 513s | Worse than Muon |

### Iteration History

#### CUM v1: Pre-NS Factored Preconditioning (FAILED)

**Architecture:** Momentum → Factored Precond → NS(3 steps) → Spectral Damping → Update

**Why it failed (diagnostic results):**
- Factored preconditioning rotates gradient direction 28° away (cosine sim 0.88 with original)
- NS(3) vs NS(5) only differs by cosine sim 0.997 — precond changes direction 10x more than reducing NS steps
- Preconditioning shrinks gradient norm by ~1000x
- Spectral damping is dead code: weight norms (~0.2-0.3) never approach threshold (30.0), damping always = 1.0
- Net effect: damaged gradient direction locked in by NS, magnitude benefit destroyed

#### CUM v2: Post-NS Adaptive Row/Column Scaling (FAILED)

**Architecture:** Momentum → NS(5 steps) → Row/Col Variance Scaling → Update

**Why it failed:**
- Post-NS scaling concept is sound (apply curvature AFTER direction is set)
- But row/col gradient variance at this model scale is too noisy
- Scaling just adds noise without capturing meaningful curvature
- Result: slightly worse than Muon on transformer (1.5322 vs 1.5198)

#### CUM v3: Soft NS + Cautious Masking (IN PROGRESS)

**Architecture:** Momentum → [Cautious Mask] → NS(5 steps) → Soft Blend → Update

**Key insight:** NS orthogonalization is too aggressive — it equalizes ALL singular values,
destroying information about which directions matter more. The gradient's singular value
structure encodes useful info: high SV = large activations AND large error = important direction.

**Soft NS formula:**
```
update = (1 - α) * NS(u) + α * normalize(u)
```
- α=0: pure Muon (full orthogonalization)
- α>0: partially preserve gradient's singular value structure

**Cautious masking:** Zero out momentum entries that disagree with current gradient sign
before NS, giving NS a cleaner input signal. Rescale to preserve norm.

**Status:** Benchmarking in progress...

### Key Bug Fixes

1. **Weight decay kills Muon/CUM:** NS-orthogonalized updates have per-element magnitude ~lr/sqrt(n_params) ≈ 1.6e-5. Weight decay at 0.01 removes ~2e-4 per step (12x larger). Weights shrink to zero. **Fix:** Set wd=0.0 for all Muon/CUM param groups.

2. **SSL certificate errors on macOS:** Python 3.12 SSL verification fails for dataset downloads. **Fix:** certifi monkey-patch + pre-downloaded datasets.

3. **CUM 4D assertion on conv weights:** CUM only supports 2D params. **Fix:** CUMConv wrapper that reshapes 4D→2D before optimizer step. CUMv2/v3 handle this internally.
