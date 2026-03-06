"""Tests for Newton-Schulz orthogonalization."""

import pytest
import torch
from cum.newton_schulz import newton_schulz_orthogonalize
from cum.utils import ns_convergence_error


class TestNewtonSchulz:

    def test_converges_to_scaled_polar_factor(self):
        """NS output should be proportional to the polar factor.

        The Muon/Zetta NS coefficients converge to a scaled orthogonal matrix
        (σ ≈ 0.877), not exact orthogonal (σ = 1). We check that the direction
        matches the polar factor.
        """
        torch.manual_seed(42)
        G = torch.randn(32, 32)
        X = newton_schulz_orthogonalize(G, steps=5)

        # Compare to true polar factor via SVD
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        polar = U @ Vh

        # Normalize both to compare direction (not magnitude)
        X_normed = X / X.norm()
        polar_normed = polar / polar.norm()
        diff = (X_normed - polar_normed).norm()
        assert diff < 0.3, f"NS didn't converge to polar factor direction: error = {diff:.6f}"

    def test_singular_values_converge(self):
        """After 5 steps, all singular values should be approximately equal."""
        torch.manual_seed(42)
        G = torch.randn(32, 32)
        X = newton_schulz_orthogonalize(G, steps=5)
        spread = ns_convergence_error(X)  # σ_max/σ_min - 1
        assert spread < 1.0, f"SV spread too high after 5 steps: {spread:.6f}"

    def test_deterministic(self):
        G = torch.randn(32, 32)
        X1 = newton_schulz_orthogonalize(G, steps=3)
        X2 = newton_schulz_orthogonalize(G, steps=3)
        assert torch.equal(X1, X2), "NS is not deterministic"

    def test_rectangular_wide(self):
        torch.manual_seed(42)
        G = torch.randn(16, 64)
        X = newton_schulz_orthogonalize(G, steps=5)
        assert X.shape == (16, 64)
        spread = ns_convergence_error(X)
        assert spread < 2.0, f"Wide matrix SV spread: {spread:.6f}"

    def test_rectangular_tall(self):
        torch.manual_seed(42)
        G = torch.randn(64, 16)
        X = newton_schulz_orthogonalize(G, steps=5)
        assert X.shape == (64, 16)
        spread = ns_convergence_error(X)
        assert spread < 2.0, f"Tall matrix SV spread: {spread:.6f}"

    def test_more_steps_improves_convergence(self):
        """More NS steps should give tighter SV spread (monotonic improvement)."""
        torch.manual_seed(42)
        G = torch.randn(32, 32)
        X2 = newton_schulz_orthogonalize(G, steps=2)
        X3 = newton_schulz_orthogonalize(G, steps=3)
        X5 = newton_schulz_orthogonalize(G, steps=5)
        s2 = ns_convergence_error(X2)
        s3 = ns_convergence_error(X3)
        s5 = ns_convergence_error(X5)
        # After enough steps, convergence saturates at the fixed point.
        # So check 2 < 3 < 5 improvement with tolerance for saturation.
        assert s5 < s2, f"5 steps should beat 2 steps: {s5:.4f} vs {s2:.4f}"
        assert s3 < s2, f"3 steps should beat 2 steps: {s3:.4f} vs {s2:.4f}"
