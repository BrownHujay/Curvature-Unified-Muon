"""
Analysis: Newton-Schulz convergence verification.
Validates that 3 preconditioned NS steps match 5 raw steps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from cum.newton_schulz import newton_schulz_orthogonalize
from cum.factored_precond import apply_factored_precond
from cum.utils import ns_convergence_error, sv_spread


def analyze_ns_convergence(
    matrix_sizes=[(32, 32), (64, 64), (128, 128), (32, 128), (128, 32)],
    condition_numbers=[10, 50, 100, 500],
    n_seeds=5,
):
    """Comprehensive NS convergence analysis."""

    print("NS Convergence Analysis")
    print("=" * 80)
    print(f"{'Size':<12} {'Cond#':<8} {'Raw(3)':<12} {'Raw(5)':<12} {'Precond(3)':<12} {'SV Compress':<12}")
    print("-" * 80)

    for m, n in matrix_sizes:
        for cond in condition_numbers:
            raw3_errors = []
            raw5_errors = []
            precond3_errors = []
            compressions = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)

                # Create ill-conditioned matrix
                U, _ = torch.linalg.qr(torch.randn(m, m))
                V, _ = torch.linalg.qr(torch.randn(n, n))
                k = min(m, n)
                sigmas = torch.linspace(1.0, cond, k)
                S = torch.zeros(m, n)
                for i in range(k):
                    S[i, i] = sigmas[i]
                G = U @ S @ V.T

                # Raw NS
                X3 = newton_schulz_orthogonalize(G, steps=3)
                X5 = newton_schulz_orthogonalize(G, steps=5)
                raw3_errors.append(ns_convergence_error(X3))
                raw5_errors.append(ns_convergence_error(X5))

                # Preconditioned NS
                raw_spread = sv_spread(G)
                row_var = torch.zeros(m)
                col_var = torch.zeros(n)
                for step in range(1, 11):
                    g = G + torch.randn(m, n) * 0.1
                    G_precond, row_var, col_var = apply_factored_precond(
                        g, g, row_var, col_var, beta2=0.99, step=step, eps=1e-7,
                    )

                precond_spread = sv_spread(G_precond)
                X_precond = newton_schulz_orthogonalize(G_precond, steps=3)
                precond3_errors.append(ns_convergence_error(X_precond))
                compressions.append(raw_spread / max(precond_spread, 1e-10))

            print(f"{m}x{n:<8} {cond:<8} "
                  f"{sum(raw3_errors)/len(raw3_errors):<12.6f} "
                  f"{sum(raw5_errors)/len(raw5_errors):<12.6f} "
                  f"{sum(precond3_errors)/len(precond3_errors):<12.6f} "
                  f"{sum(compressions)/len(compressions):<12.1f}x")

    print("=" * 80)
    print("Key: Raw(k) = ||XX^T-I|| after k raw NS steps")
    print("     Precond(3) = ||XX^T-I|| after 3 preconditioned NS steps")
    print("     SV Compress = ratio of raw to preconditioned SV spread")


if __name__ == "__main__":
    analyze_ns_convergence()
