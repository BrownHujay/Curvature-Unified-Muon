"""Tests for factored preconditioning."""

import pytest
import torch
from cum.factored_precond import apply_factored_precond


class TestFactoredPrecond:

    def test_scaling_preserves_shape(self):
        g = torch.randn(32, 64)
        u = g.clone()
        row_var = torch.zeros(32)
        col_var = torch.zeros(64)

        precond, _, _ = apply_factored_precond(u, g, row_var, col_var, 0.99, 1, 1e-7)
        assert precond.shape == (32, 64)

    def test_scaling_effect(self):
        """High-variance rows/cols get scaled down."""
        torch.manual_seed(42)
        g = torch.randn(16, 16)
        g[0] *= 5  # Row 0 has high variance

        row_var = torch.zeros(16)
        col_var = torch.zeros(16)
        precond, _, _ = apply_factored_precond(g, g, row_var, col_var, 0.99, 1, 1e-7)

        # Row 0 should be more scaled down than row 1
        ratio_0 = precond[0].norm() / g[0].norm()
        ratio_1 = precond[1].norm() / g[1].norm()
        assert ratio_0 < ratio_1, "High-variance row not scaled down"
