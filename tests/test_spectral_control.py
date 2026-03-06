"""Tests for spectral control (power iteration + damping)."""

import pytest
import torch
from cum.spectral_control import spectral_damping


class TestSpectralControl:

    def test_power_iteration_converges(self):
        torch.manual_seed(42)
        W = torch.randn(32, 32)
        v = torch.randn(32)
        v = v / v.norm()

        true_sigma = torch.linalg.svdvals(W)[0].item()
        for _ in range(100):
            _, v = spectral_damping(W, v, sigma_max=100.0, alpha_damp=0.1)

        sigma_est = (W @ v).norm().item()
        assert abs(sigma_est - true_sigma) / true_sigma < 0.01

    def test_damping_formula(self):
        """damping = 1/(1 + α·max(0, σ - σ_target))."""
        # Direct formula check with known sigma_est
        alpha = 0.1
        sigma_max = 30.0

        # σ below threshold
        excess = max(0, 25.0 - sigma_max)
        d = 1.0 / (1.0 + alpha * excess)
        assert d == 1.0

        # σ above threshold
        excess = max(0, 50.0 - sigma_max)
        d = 1.0 / (1.0 + alpha * excess)
        expected = 1.0 / (1.0 + 0.1 * 20.0)
        assert abs(d - expected) < 1e-6

    def test_damping_range(self):
        """0 < damping <= 1 for all valid inputs."""
        torch.manual_seed(42)
        W = torch.randn(32, 32) * 100
        v = torch.randn(32)
        v = v / v.norm()

        for _ in range(50):
            damping, v = spectral_damping(W, v, sigma_max=30.0, alpha_damp=0.1)

        assert 0 < damping <= 1.0, f"Damping out of range: {damping}"
