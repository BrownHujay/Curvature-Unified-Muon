"""
Plot Set 4: CUM Diagnostics
- 4a: NS Convergence Error vs NS Steps
- 4b: Singular Value Spread Before/After Preconditioning
- 4c: Ablation Results Grid
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.plotting.style import apply_style, get_color

apply_style()


def plot_ns_convergence(output_path, max_steps=7):
    """Plot 4a: NS convergence error vs number of NS steps.

    Compares raw (Muon-style) vs preconditioned (CUM-style) NS convergence.
    """
    import torch
    from cum.newton_schulz import newton_schulz_orthogonalize
    from cum.factored_precond import apply_factored_precond
    from cum.utils import ns_convergence_error

    torch.manual_seed(42)

    # Create ill-conditioned matrix
    m, n = 64, 64
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    sigmas = torch.linspace(1.0, 100.0, min(m, n))
    S = torch.zeros(m, n)
    for i in range(min(m, n)):
        S[i, i] = sigmas[i]
    G = U @ S @ V.T

    # Build preconditioned version
    row_var = torch.zeros(m)
    col_var = torch.zeros(n)
    for step in range(1, 21):
        g = G + torch.randn(m, n) * 0.1
        G_precond, row_var, col_var = apply_factored_precond(
            g, g, row_var, col_var, beta2=0.99, step=step, eps=1e-7,
        )

    # Measure convergence at each step count
    ns_range = range(1, max_steps + 1)
    raw_errors = []
    precond_errors = []

    for ns in ns_range:
        X_raw = newton_schulz_orthogonalize(G, steps=ns)
        X_precond = newton_schulz_orthogonalize(G_precond, steps=ns)
        raw_errors.append(ns_convergence_error(X_raw))
        precond_errors.append(ns_convergence_error(X_precond))

    fig, ax = plt.subplots()
    ax.semilogy(list(ns_range), raw_errors, 'o-', color=get_color("Muon"),
                label="Raw (Muon-style)", linewidth=2, markersize=8)
    ax.semilogy(list(ns_range), precond_errors, 's-', color=get_color("CUM"),
                label="Preconditioned (CUM-style)", linewidth=2, markersize=8)

    ax.axhline(y=0.01, color='k', linestyle='--', alpha=0.3, label="Target (0.01)")
    ax.axvline(x=3, color=get_color("CUM"), linestyle=':', alpha=0.3, label="CUM default (3)")
    ax.axvline(x=5, color=get_color("Muon"), linestyle=':', alpha=0.3, label="Muon default (5)")

    ax.set_xlabel("Number of NS Steps")
    ax.set_ylabel("||XX^T - I||_F (log scale)")
    ax.set_title("Newton-Schulz Convergence: Raw vs Preconditioned")
    ax.legend()
    ax.set_xticks(list(ns_range))

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_sv_spread(output_path, n_samples=10):
    """Plot 4b: Singular value spread before/after preconditioning."""
    import torch
    from cum.factored_precond import apply_factored_precond
    from cum.utils import sv_spread

    raw_spreads = []
    precond_spreads = []

    for seed in range(n_samples):
        torch.manual_seed(seed)
        m, n = 64, 64
        U, _ = torch.linalg.qr(torch.randn(m, m))
        V, _ = torch.linalg.qr(torch.randn(n, n))
        sigmas = torch.linspace(1.0, 50.0, min(m, n))
        S = torch.zeros(m, n)
        for i in range(min(m, n)):
            S[i, i] = sigmas[i]
        G = U @ S @ V.T

        raw_spreads.append(sv_spread(G))

        row_var = torch.zeros(m)
        col_var = torch.zeros(n)
        for step in range(1, 11):
            g = G + torch.randn(m, n) * 0.1
            G_precond, row_var, col_var = apply_factored_precond(
                g, g, row_var, col_var, beta2=0.99, step=step, eps=1e-7,
            )

        precond_spreads.append(sv_spread(G_precond))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_samples)
    width = 0.35

    ax.bar(x - width/2, raw_spreads, width, label="Before Precond (Raw)",
           color=get_color("Muon"), alpha=0.8)
    ax.bar(x + width/2, precond_spreads, width, label="After Precond (CUM)",
           color=get_color("CUM"), alpha=0.8)

    ax.set_xlabel("Sample Matrix")
    ax.set_ylabel("SV Spread (sigma_max / sigma_min)")
    ax.set_title("Singular Value Spread: Before vs After Preconditioning")
    ax.set_xticks(x)
    ax.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_ablation_grid(data, output_path):
    """Plot 4c: Ablation results as heatmap."""
    if not data:
        print("No ablation data provided, skipping.")
        return

    variants = list(data.keys())
    metrics = list(data[variants[0]].keys()) if variants else []

    values = np.array([[data[v][m] for m in metrics] for v in variants])

    fig, ax = plt.subplots(figsize=(10, len(variants) * 0.6 + 2))
    im = ax.imshow(values, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)

    for i in range(len(variants)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{values[i, j]:.4f}', ha='center', va='center', fontsize=9)

    ax.set_title("Ablation Study Results")
    fig.colorbar(im, ax=ax, label="Val Loss")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots")
    parser.add_argument("--output-dir", default="evaluation/results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # These can be generated without benchmark results
    plot_ns_convergence(os.path.join(args.output_dir, "ns_convergence.png"))
    plot_sv_spread(os.path.join(args.output_dir, "sv_spread.png"))

    print(f"\nDiagnostic plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
