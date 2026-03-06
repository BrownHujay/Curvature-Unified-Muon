"""
Plot Set 3: Stability Analysis Plots
- 3a: Max QK Attention Score vs Step
- 3b: Max Spectral Norm vs Step (Per Layer)
- 3c: CUM Damping Factor vs Step (Per Layer)
- 3d: MuonClip Clip Events vs Step
"""

import os
import sys
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.plotting.style import apply_style, get_color, get_linestyle

apply_style()


def plot_qk_scores(data, output_path):
    """Plot 3a: Max QK attention score vs step for each optimizer."""
    fig, ax = plt.subplots()

    for opt_name, metrics in data.items():
        steps = [m["step"] for m in metrics]
        qk_scores = [m.get("max_qk_score", 0) for m in metrics]

        if any(q > 0 for q in qk_scores):
            ax.plot(steps, qk_scores, color=get_color(opt_name),
                    linestyle=get_linestyle(opt_name),
                    label=opt_name, linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Max QK Attention Score")
    ax.set_title("Attention Stability: Max QK Score")
    ax.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_spectral_norms(data, output_path, n_layers=4):
    """Plot 3b: Spectral norms per layer vs training step."""
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for opt_name, metrics in data.items():
        steps = [m["step"] for m in metrics]

        for layer_idx in range(n_layers):
            key = f"sigma_max_layer{layer_idx}"
            values = [m.get(key, 0) for m in metrics]

            if any(v > 0 for v in values):
                axes[layer_idx].plot(steps, values, color=get_color(opt_name),
                                     linestyle=get_linestyle(opt_name),
                                     label=opt_name, linewidth=1.5)
                axes[layer_idx].set_title(f"Layer {layer_idx}")
                axes[layer_idx].set_xlabel("Step")

    axes[0].set_ylabel("Spectral Norm")
    axes[0].legend(fontsize=9)
    fig.suptitle("Spectral Norm per Layer", y=1.02)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_damping_factors(data, output_path):
    """Plot 3c: CUM damping factor vs step."""
    fig, ax = plt.subplots()

    if "CUM" in data:
        metrics = data["CUM"]
        steps = [m["step"] for m in metrics]
        damping = [m.get("damping_mean", 1.0) for m in metrics]

        ax.plot(steps, damping, color=get_color("CUM"), linewidth=2, label="CUM damping (mean)")
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label="No damping (1.0)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Damping Factor")
    ax.set_title("CUM Spectral Damping Over Training")
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate stability plots")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.results_dir), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Stability plots require structured metric data with QK scores and spectral norms.")
    print("Run benchmarks first with --log-stability flag.")


if __name__ == "__main__":
    main()
