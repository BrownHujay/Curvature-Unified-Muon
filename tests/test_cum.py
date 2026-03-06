"""Unit tests for the CUM optimizer (GPU-capable)."""

import pytest
import torch
from cum import CUM
from cum.utils import aspect_ratio_scale


class TestCUMOptimizer:

    def test_single_step_updates_weights(self):
        W = torch.randn(32, 32, requires_grad=True)
        W_orig = W.clone()
        opt = CUM([W], lr=0.01)
        loss = (W ** 2).sum()
        loss.backward()
        opt.step()
        assert not torch.equal(W, W_orig), "Weights didn't change"

    def test_momentum_accumulation(self):
        W = torch.randn(32, 32, requires_grad=True)
        opt = CUM([W], lr=0.01, beta1=0.9)

        # Step 1
        loss = (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        g1 = W.grad.clone()
        opt.step()

        # Check momentum = (1-beta1) * g1
        m = opt.state[W]["momentum_buffer"]
        expected = 0.1 * g1
        assert torch.allclose(m, expected, atol=1e-5), "Momentum accumulation wrong"

    def test_weight_decay_applied(self):
        W = torch.randn(32, 32, requires_grad=True)
        W_before = W.data.clone()
        opt = CUM([W], lr=0.0, weight_decay=0.1)  # lr=0 so only WD acts

        loss = (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # W should shrink by factor (1 - wd) = 0.9
        # But the orthogonal update with lr=0 means only WD acts
        # The update also adds -lr*damping*scale*orth, but lr=0 so that's 0
        # However the NS step still produces an orth matrix
        # So W_new = (1 - 0.1) * W_before - 0.0 * orth = 0.9 * W_before
        expected = 0.9 * W_before
        # Allow some tolerance since the orthogonal update still happens
        # Actually with lr=0, the add_ with alpha=0 does nothing
        assert torch.allclose(W.data, expected, atol=1e-5), "Weight decay not applied correctly"

    def test_aspect_ratio_scaling(self):
        assert aspect_ratio_scale(64, 64) == 1.0
        assert abs(aspect_ratio_scale(128, 64) - (128/64)**0.5) < 1e-6
        assert aspect_ratio_scale(32, 64) == 1.0  # m < n → max(1, m/n) = 1

    def test_gradient_zero_handling(self):
        W = torch.randn(16, 16, requires_grad=True)
        opt = CUM([W], lr=0.01)

        opt.zero_grad()
        W.grad = torch.zeros_like(W)
        opt.step()  # Should not produce NaN
        assert not torch.isnan(W).any()

    def test_very_rectangular_matrix(self):
        """Works for highly rectangular matrices."""
        for shape in [(16, 256), (256, 16), (64, 512)]:
            W = torch.randn(*shape, requires_grad=True)
            opt = CUM([W], lr=0.01)
            loss = (W ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert W.shape == shape
            assert not torch.isnan(W).any()
