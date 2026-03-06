"""Tests for hybrid CUM + AdamW optimizer."""

import pytest
import torch
import torch.nn as nn
from cum.hybrid import CUMWithAuxAdam


class TestHybridOptimizer:

    def test_creates_without_error(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        hidden_2d = [p for p in model.parameters() if p.ndim == 2]
        other = [p for p in model.parameters() if p.ndim < 2]

        opt = CUMWithAuxAdam([
            {"params": hidden_2d, "use_cum": True, "lr": 0.02},
            {"params": other, "use_cum": False, "lr": 3e-4, "betas": (0.9, 0.95)},
        ])

        # Run a step
        x = torch.randn(4, 32)
        loss = model(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_separate_params_updated(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        hidden_2d = [p for p in model.parameters() if p.ndim == 2]
        other = [p for p in model.parameters() if p.ndim < 2]

        w_before = [p.clone() for p in model.parameters()]

        opt = CUMWithAuxAdam([
            {"params": hidden_2d, "use_cum": True, "lr": 0.02},
            {"params": other, "use_cum": False, "lr": 3e-4, "betas": (0.9, 0.95)},
        ])

        x = torch.randn(4, 32)
        loss = model(x).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # All params should have changed
        for p_before, p_after in zip(w_before, model.parameters()):
            assert not torch.equal(p_before, p_after.data), "Some params didn't update"
