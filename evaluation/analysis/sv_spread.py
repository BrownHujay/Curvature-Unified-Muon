"""
Analysis: Singular value spread measurement.
Validates that factored preconditioning compresses the spectrum.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from cum.factored_precond import apply_factored_precond
from cum.utils import sv_spread


def analyze_sv_spread(
    matrix_sizes=[(32, 32), (64, 64), (128, 128), (32, 128), (128, 32)],
    n_seeds=10,
    beta2_values=[0.9, 0.99, 0.999],
    precond_steps=20,
):
    """Measure SV spread compression across different settings."""

    print("Singular Value Spread Analysis")
    print("=" * 70)

    for beta2 in beta2_values:
        print(f"\nbeta2 = {beta2}")
        print(f"{'Size':<12} {'Raw Spread':<14} {'Precond Spread':<16} {'Compression':<14} {'Success':<8}")
        print("-" * 64)

        for m, n in matrix_sizes:
            raw_spreads = []
            precond_spreads = []

            for seed in range(n_seeds):
                torch.manual_seed(seed * 100 + 7)

                # Random matrix with natural spread
                G = torch.randn(m, n)
                raw_spreads.append(sv_spread(G))

                row_var = torch.zeros(m)
                col_var = torch.zeros(n)
                for step in range(1, precond_steps + 1):
                    g = G + torch.randn(m, n) * 0.1
                    G_precond, row_var, col_var = apply_factored_precond(
                        g, g, row_var, col_var, beta2=beta2, step=step, eps=1e-7,
                    )

                precond_spreads.append(sv_spread(G_precond))

            avg_raw = sum(raw_spreads) / len(raw_spreads)
            avg_precond = sum(precond_spreads) / len(precond_spreads)
            compression = avg_raw / max(avg_precond, 1e-10)
            success = sum(1 for r, p in zip(raw_spreads, precond_spreads) if p < r)

            print(f"{m}x{n:<8} {avg_raw:<14.2f} {avg_precond:<16.2f} {compression:<14.1f}x {success}/{n_seeds}")

    print("=" * 70)


if __name__ == "__main__":
    analyze_sv_spread()
