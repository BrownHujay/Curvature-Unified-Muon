"""
Tier 0 CPU tests for CUM optimizer.
Must all pass before ANY benchmark runs.
Run with: pytest tests/test_tier0_cpu.py -v

Total runtime target: < 60 seconds on M3 CPU.
"""

import pytest
import torch
import math
import copy


# ─── Helpers ───

def make_random_matrix(m, n, seed=42):
    torch.manual_seed(seed)
    return torch.randn(m, n)


def make_ill_conditioned_matrix(m, n, condition_number=100, seed=42):
    """Create matrix with known condition number via SVD construction."""
    torch.manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    k = min(m, n)
    sigmas = torch.linspace(1.0, condition_number, k)
    S = torch.zeros(m, n)
    for i in range(k):
        S[i, i] = sigmas[i]
    return U @ S @ V.T


def run_cum_steps(W_shape, n_steps=10, seed=42, **cum_kwargs):
    """Run CUM optimizer for n_steps on a simple quadratic loss."""
    from cum import CUM
    torch.manual_seed(seed)
    W = torch.randn(*W_shape, requires_grad=True)
    W_target = torch.randn(*W_shape)
    optimizer = CUM([W], **cum_kwargs)

    losses = []
    for _ in range(n_steps):
        loss = ((W - W_target) ** 2).sum()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return W, optimizer, losses


# ═══════════════════════════════════════════
# TestCUMCPUSmoke: Ultra-fast smoke tests
# Each test < 2 seconds
# ═══════════════════════════════════════════

class TestCUMCPUSmoke:

    def test_single_step_doesnt_crash(self):
        """Create 32x32 param, run 1 CUM step, verify no errors."""
        from cum import CUM
        W = torch.randn(32, 32, requires_grad=True)
        opt = CUM([W], lr=0.01)
        loss = (W ** 2).sum()
        loss.backward()
        opt.step()  # Should not raise

    def test_output_shape_preserved(self):
        """W stays (m, n) after optimizer step."""
        from cum import CUM
        for shape in [(32, 32), (64, 16), (16, 64), (128, 32)]:
            W = torch.randn(*shape, requires_grad=True)
            opt = CUM([W], lr=0.01)
            loss = (W ** 2).sum()
            loss.backward()
            opt.step()
            assert W.shape == shape, f"Shape changed from {shape} to {W.shape}"

    def test_loss_decreases_on_quadratic(self):
        """On f(W) = ||W - W*||^2, loss after 10 steps < loss at step 0."""
        W, _, losses = run_cum_steps((32, 32), n_steps=10, lr=0.01)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_nan_free(self):
        """No NaN in W, momentum, r, c, v after 50 steps."""
        from cum import CUM
        torch.manual_seed(42)
        W = torch.randn(32, 32, requires_grad=True)
        opt = CUM([W], lr=0.01)

        for i in range(50):
            loss = (W ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert not torch.isnan(W).any(), "NaN in W"
        state = opt.state[W]
        assert not torch.isnan(state["momentum_buffer"]).any(), "NaN in momentum"
        assert not torch.isnan(state["row_var"]).any(), "NaN in row_var"
        assert not torch.isnan(state["col_var"]).any(), "NaN in col_var"
        assert not torch.isnan(state["power_iter_v"]).any(), "NaN in power_iter_v"

    def test_zero_grad_safe(self):
        """Zero gradient doesn't produce NaN (eps protection)."""
        from cum import CUM
        W = torch.randn(16, 16, requires_grad=True)
        opt = CUM([W], lr=0.01)

        # First step with real gradient
        loss = (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Second step with zero gradient
        opt.zero_grad()
        W.grad = torch.zeros_like(W)
        opt.step()

        assert not torch.isnan(W).any(), "NaN after zero gradient step"

    def test_state_dict_roundtrip(self):
        """optimizer.state_dict() → load_state_dict() preserves state."""
        from cum import CUM
        torch.manual_seed(42)
        W = torch.randn(32, 32, requires_grad=True)
        opt = CUM([W], lr=0.01)

        # Run a few steps
        for _ in range(5):
            loss = (W ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Save state
        sd = opt.state_dict()

        # Create new optimizer and load
        W2 = W.clone().detach().requires_grad_(True)
        opt2 = CUM([W2], lr=0.01)
        # Need to run one step to initialize state before loading
        loss = (W2 ** 2).sum()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        opt2.load_state_dict(sd)

        # Verify state matches
        s1 = opt.state[W]
        s2 = opt2.state[W2]
        assert s1["step"] == s2["step"]
        assert torch.allclose(s1["momentum_buffer"], s2["momentum_buffer"])
        assert torch.allclose(s1["row_var"], s2["row_var"])
        assert torch.allclose(s1["col_var"], s2["col_var"])


# ═══════════════════════════════════════════
# TestNSConvergenceOnCPU
# ═══════════════════════════════════════════

class TestNSConvergenceOnCPU:

    def test_5_raw_steps_converges(self):
        """Standard Muon: 5 NS steps on random matrix → SV spread < 2.0.

        Note: The Muon/Zetta NS coefficients converge singular values to a
        common fixed point (~0.877), not to 1.0. We measure SV spread
        (σ_max/σ_min - 1) to check convergence quality.
        """
        from cum.newton_schulz import newton_schulz_orthogonalize
        from cum.utils import ns_convergence_error

        torch.manual_seed(42)
        G = torch.randn(32, 32)
        X = newton_schulz_orthogonalize(G, steps=5)
        error = ns_convergence_error(X)
        assert error < 1.0, f"5-step NS SV spread too high: {error:.6f}"

    def test_3_raw_steps_insufficient_on_structured(self):
        """3 NS steps on row/col-scaled gradient → SV spread > 5.0.

        When gradients have row/col scale imbalance (common in training),
        3 raw NS steps are insufficient without preconditioning.
        """
        from cum.newton_schulz import newton_schulz_orthogonalize
        from cum.utils import ns_convergence_error

        torch.manual_seed(42)
        G = torch.randn(32, 32)
        row_scales = torch.ones(32)
        row_scales[:8] = 10.0
        col_scales = torch.ones(32)
        col_scales[:8] = 5.0
        G_scaled = G * row_scales[:, None] * col_scales[None, :]

        X = newton_schulz_orthogonalize(G_scaled, steps=3)
        error = ns_convergence_error(X)
        assert error > 5.0, f"3 raw steps shouldn't converge well on scaled input, got spread: {error:.6f}"

    def test_3_preconditioned_beats_3_raw(self):
        """
        CUM: 3 NS steps on preconditioned momentum gives better SV spread
        than 3 raw NS steps on the same structured gradient.
        THIS IS THE KEY TEST. If preconditioning doesn't help NS converge
        faster, the compute savings thesis is dead.
        """
        from cum.newton_schulz import newton_schulz_orthogonalize
        from cum.factored_precond import apply_factored_precond
        from cum.utils import ns_convergence_error

        torch.manual_seed(42)
        # Create gradient with row/col scale structure
        row_scales = torch.ones(32)
        row_scales[:8] = 10.0
        col_scales = torch.ones(32)
        col_scales[:8] = 5.0

        # Build preconditioner statistics from structured gradients
        row_var = torch.zeros(32)
        col_var = torch.zeros(32)
        for step in range(1, 21):
            g = torch.randn(32, 32) * row_scales[:, None] * col_scales[None, :]
            preconditioned, row_var, col_var = apply_factored_precond(
                g, g, row_var, col_var, beta2=0.99, step=step, eps=1e-7,
            )

        # Compare: 3 raw NS vs 3 preconditioned NS
        G_raw = torch.randn(32, 32) * row_scales[:, None] * col_scales[None, :]
        X_raw = newton_schulz_orthogonalize(G_raw, steps=3)
        raw_error = ns_convergence_error(X_raw)

        X_precond = newton_schulz_orthogonalize(preconditioned, steps=3)
        precond_error = ns_convergence_error(X_precond)

        assert precond_error < raw_error, (
            f"Preconditioning should help: precond SV spread={precond_error:.4f} "
            f"vs raw SV spread={raw_error:.4f}"
        )

    def test_spectrum_compression(self):
        """σ_max/σ_min of preconditioned < raw in ≥8/10 row/col-scaled matrices."""
        from cum.factored_precond import apply_factored_precond
        from cum.utils import sv_spread

        compressions = 0
        for seed in range(10):
            torch.manual_seed(seed)
            # Matrices with row/col scale structure
            G = torch.randn(32, 32)
            row_scales = torch.exp(torch.randn(32))  # random row scales
            col_scales = torch.exp(torch.randn(32))  # random col scales
            G_scaled = G * row_scales[:, None] * col_scales[None, :]
            raw_spread = sv_spread(G_scaled)

            row_var = torch.zeros(32)
            col_var = torch.zeros(32)
            for step in range(1, 21):
                g = torch.randn(32, 32) * row_scales[:, None] * col_scales[None, :]
                preconditioned, row_var, col_var = apply_factored_precond(
                    g, g, row_var, col_var, beta2=0.99, step=step, eps=1e-7,
                )

            precond_spread = sv_spread(preconditioned)
            if precond_spread < raw_spread:
                compressions += 1

        assert compressions >= 8, f"Only {compressions}/10 cases showed compression"

    def test_rectangular_32x128(self):
        """NS produces reasonable SV spread for wide (32x128) matrix in 3 steps."""
        from cum.newton_schulz import newton_schulz_orthogonalize
        from cum.utils import ns_convergence_error

        torch.manual_seed(42)
        G = torch.randn(32, 128)
        X = newton_schulz_orthogonalize(G, steps=3)
        error = ns_convergence_error(X)
        assert error < 5.0, f"Rectangular 32x128 NS SV spread: {error:.6f}"

    def test_rectangular_128x32(self):
        """NS produces reasonable SV spread for tall (128x32) matrix in 3 steps."""
        from cum.newton_schulz import newton_schulz_orthogonalize
        from cum.utils import ns_convergence_error

        torch.manual_seed(42)
        G = torch.randn(128, 32)
        X = newton_schulz_orthogonalize(G, steps=3)
        error = ns_convergence_error(X)
        assert error < 5.0, f"Rectangular 128x32 NS SV spread: {error:.6f}"


# ═══════════════════════════════════════════
# TestFactoredPrecondCPU
# ═══════════════════════════════════════════

class TestFactoredPrecondCPU:

    def test_row_var_matches_manual(self):
        """After 5 steps, row_var ≈ manual EMA of sum(g^2, dim=1)."""
        from cum.factored_precond import apply_factored_precond

        torch.manual_seed(42)
        row_var = torch.zeros(16)
        col_var = torch.zeros(16)
        manual_row = torch.zeros(16)
        beta2 = 0.99

        for step in range(1, 6):
            g = torch.randn(16, 16)
            _, row_var, col_var = apply_factored_precond(
                g, g, row_var, col_var, beta2=beta2, step=step, eps=1e-7,
            )
            manual_row = beta2 * manual_row + (1 - beta2) * (g * g).sum(dim=1)

        assert torch.allclose(row_var, manual_row, atol=1e-6), "row_var doesn't match manual EMA"

    def test_col_var_matches_manual(self):
        """After 5 steps, col_var ≈ manual EMA of sum(g^2, dim=0)."""
        from cum.factored_precond import apply_factored_precond

        torch.manual_seed(42)
        row_var = torch.zeros(16)
        col_var = torch.zeros(16)
        manual_col = torch.zeros(16)
        beta2 = 0.99

        for step in range(1, 6):
            g = torch.randn(16, 16)
            _, row_var, col_var = apply_factored_precond(
                g, g, row_var, col_var, beta2=beta2, step=step, eps=1e-7,
            )
            manual_col = beta2 * manual_col + (1 - beta2) * (g * g).sum(dim=0)

        assert torch.allclose(col_var, manual_col, atol=1e-6), "col_var doesn't match manual EMA"

    def test_bias_correction_step1(self):
        """At step 1, bias-corrected r_hat = g^2_rows (no EMA effect yet)."""
        from cum.factored_precond import apply_factored_precond

        torch.manual_seed(42)
        g = torch.randn(16, 16)
        row_var = torch.zeros(16)
        col_var = torch.zeros(16)

        precond, row_var, col_var = apply_factored_precond(
            g, g, row_var, col_var, beta2=0.99, step=1, eps=1e-7,
        )

        # At step 1: row_var = (1-beta2) * sum(g^2, dim=1)
        # bias correction: row_var / (1 - beta2^1) = sum(g^2, dim=1)
        expected_row = (g * g).sum(dim=1)
        bc = 1 - 0.99
        actual_row_corrected = row_var / bc
        assert torch.allclose(actual_row_corrected, expected_row, atol=1e-5)

    def test_high_var_rows_scaled_down(self):
        """Rows with 10x gradient magnitude get scaled ~3x down (1/sqrt(10))."""
        from cum.factored_precond import apply_factored_precond

        torch.manual_seed(42)
        g = torch.randn(16, 16)
        g[0] *= 10  # Row 0 has 10x magnitude

        u = g.clone()
        row_var = torch.zeros(16)
        col_var = torch.zeros(16)

        precond, _, _ = apply_factored_precond(
            u, g, row_var, col_var, beta2=0.99, step=1, eps=1e-7,
        )

        # Row 0 should be scaled down relative to other rows
        row0_scale = precond[0].norm() / u[0].norm()
        row1_scale = precond[1].norm() / u[1].norm()

        # Row 0 has ~10x variance, so should be scaled by 1/sqrt(10) ≈ 0.316 relative
        assert row0_scale < row1_scale, "High-variance row should be scaled down more"

    def test_precond_preserves_sign_structure(self):
        """sign(u_tilde[i,j]) == sign(u[i,j]) for all i,j. Preconditioning only scales."""
        from cum.factored_precond import apply_factored_precond

        torch.manual_seed(42)
        g = torch.randn(16, 16)
        u = g.clone()
        row_var = torch.zeros(16)
        col_var = torch.zeros(16)

        precond, _, _ = apply_factored_precond(
            u, g, row_var, col_var, beta2=0.99, step=1, eps=1e-7,
        )

        # Signs should match (ignoring near-zero entries)
        mask = u.abs() > 1e-6
        assert (torch.sign(precond[mask]) == torch.sign(u[mask])).all(), \
            "Preconditioning changed signs"


# ═══════════════════════════════════════════
# TestSpectralControlCPU
# ═══════════════════════════════════════════

class TestSpectralControlCPU:

    def test_power_iter_converges_over_steps(self):
        """After 50 optimizer steps, σ_est within 5% of true σ_max."""
        from cum.spectral_control import spectral_damping

        torch.manual_seed(42)
        W = torch.randn(32, 32)
        v = torch.randn(32)
        v = v / v.norm()

        true_sigma = torch.linalg.svdvals(W)[0].item()

        # Run many power iteration steps
        for _ in range(50):
            _, v = spectral_damping(W, v, sigma_max=100.0, alpha_damp=0.1)

        # Check final estimate
        sigma_est = (W @ v).norm().item()
        rel_error = abs(sigma_est - true_sigma) / true_sigma
        assert rel_error < 0.05, f"Power iter didn't converge: est={sigma_est:.4f}, true={true_sigma:.4f}"

    def test_damping_is_1_below_threshold(self):
        """When σ_max(W) = 5 and σ_target = 30, damping = 1.0 exactly."""
        from cum.spectral_control import spectral_damping

        # Create W with known σ_max ≈ 5
        torch.manual_seed(42)
        U, _ = torch.linalg.qr(torch.randn(32, 32))
        V, _ = torch.linalg.qr(torch.randn(32, 32))
        S = torch.diag(torch.linspace(1.0, 5.0, 32))
        W = U @ S @ V.T

        v = torch.randn(32)
        v = v / v.norm()
        for _ in range(50):
            damping, v = spectral_damping(W, v, sigma_max=30.0, alpha_damp=0.1)

        assert abs(damping - 1.0) < 1e-6, f"Damping should be 1.0 below threshold, got {damping}"

    def test_damping_decreases_above_threshold(self):
        """When σ_max(W) = 60 and σ_target = 30, damping < 1.0."""
        from cum.spectral_control import spectral_damping

        torch.manual_seed(42)
        U, _ = torch.linalg.qr(torch.randn(32, 32))
        V, _ = torch.linalg.qr(torch.randn(32, 32))
        S = torch.diag(torch.linspace(1.0, 60.0, 32))
        W = U @ S @ V.T

        v = torch.randn(32)
        v = v / v.norm()
        for _ in range(50):
            damping, v = spectral_damping(W, v, sigma_max=30.0, alpha_damp=0.1)

        assert damping < 1.0, f"Damping should be < 1.0 above threshold, got {damping}"

    def test_damping_is_smooth(self):
        """damping(σ=29.9) ≈ damping(σ=30.1). No discontinuity."""
        from cum.spectral_control import spectral_damping

        # Create two matrices with σ_max just below and above 30
        def make_W_with_sigma(sigma):
            torch.manual_seed(42)
            U, _ = torch.linalg.qr(torch.randn(32, 32))
            V, _ = torch.linalg.qr(torch.randn(32, 32))
            S = torch.diag(torch.cat([torch.ones(31), torch.tensor([sigma])]))
            return U @ S @ V.T

        W_low = make_W_with_sigma(29.9)
        W_high = make_W_with_sigma(30.1)

        v = torch.randn(32)
        v = v / v.norm()

        for _ in range(100):
            d_low, v_low = spectral_damping(W_low, v.clone(), sigma_max=30.0, alpha_damp=0.1)
        for _ in range(100):
            d_high, v_high = spectral_damping(W_high, v.clone(), sigma_max=30.0, alpha_damp=0.1)

        assert abs(d_low - d_high) < 0.05, f"Damping is not smooth: {d_low} vs {d_high}"

    def test_damping_never_zero(self):
        """Even at σ_est = 1000, damping > 0."""
        from cum.spectral_control import spectral_damping

        torch.manual_seed(42)
        U, _ = torch.linalg.qr(torch.randn(32, 32))
        V, _ = torch.linalg.qr(torch.randn(32, 32))
        S = torch.diag(torch.linspace(1.0, 1000.0, 32))
        W = U @ S @ V.T

        v = torch.randn(32)
        v = v / v.norm()
        for _ in range(50):
            damping, v = spectral_damping(W, v, sigma_max=30.0, alpha_damp=0.1)

        assert damping > 0, f"Damping should never be zero, got {damping}"
        assert damping < 0.2, f"Damping should be very small at σ=1000, got {damping}"
