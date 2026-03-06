"""
Generate all Tier 0 comparison plots.
Usage: python benchmarks/tier0/plot_tier0.py --results-dir benchmarks/tier0/results/
"""

import os
import sys
import argparse
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

COLORS = {
    "AdamW": "#1f77b4",
    "SGD+Nesterov": "#7f7f7f",
    "Muon": "#ff7f0e",
    "CUM": "#2ca02c",
}

LINESTYLES = {
    "AdamW": "--",
    "SGD+Nesterov": ":",
    "Muon": "-.",
    "CUM": "-",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_tier0c_convergence(results_dir, output_dir):
    """Plot val loss vs steps for all optimizers (Tier 0c)."""
    fig, ax = plt.subplots()

    for csv_file in sorted(glob.glob(os.path.join(results_dir, "tier0c_*.csv"))):
        if "summary" in csv_file:
            continue

        basename = os.path.basename(csv_file).replace("tier0c_", "").replace(".csv", "")
        parts = basename.split("_lr")
        opt_name = parts[0]
        lr = parts[1] if len(parts) > 1 else ""

        steps, val_losses = [], []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                val_losses.append(float(row["val_loss"]))

        if opt_name in COLORS:
            ax.plot(steps, val_losses, color=COLORS[opt_name],
                    linestyle=LINESTYLES.get(opt_name, "-"),
                    label=f"{opt_name} (lr={lr})", alpha=0.7)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Tier 0c: Micro-Transformer Convergence")
    ax.axhline(y=1.50, color='k', linestyle='--', alpha=0.3, label='Target (1.50)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "convergence_0c.png"))
    plt.close()
    print(f"Saved convergence_0c.png")


def plot_steps_to_target_bar(results_dir, output_dir):
    """Bar chart of best steps-to-target across optimizers."""
    summary_files = {
        "0a": "tier0a_results.csv",
        "0b": "tier0b_results.csv",
        "0c": "tier0c_summary.csv",
        "0d": "tier0d_results.csv",
    }

    for tier, filename in summary_files.items():
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            continue

        best_per_opt = {}
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                opt = row["optimizer"]
                steps = int(row["steps_to_target"])
                if opt not in best_per_opt or steps < best_per_opt[opt]:
                    best_per_opt[opt] = steps

        if not best_per_opt:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        opts = list(best_per_opt.keys())
        steps = [best_per_opt[o] for o in opts]
        colors = [COLORS.get(o, "#333333") for o in opts]

        ax.bar(opts, steps, color=colors)
        ax.set_ylabel("Steps to Target")
        ax.set_title(f"Tier {tier}: Steps to Target (Best LR)")
        ax.grid(True, alpha=0.3, axis='y')

        plt.savefig(os.path.join(output_dir, f"steps_to_target_{tier}.png"))
        plt.close()
        print(f"Saved steps_to_target_{tier}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    plot_tier0c_convergence(args.results_dir, output_dir)
    plot_steps_to_target_bar(args.results_dir, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
