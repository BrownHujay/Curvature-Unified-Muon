"""
Plot Set 1: Convergence Comparison Plots
- 1a: Val Loss vs Steps
- 1b: Val Loss vs Wall-Clock Time
- 1c: Val Loss vs Tokens Processed
- 1d: Steps to Target Bar Chart
"""

import os
import sys
import argparse
import csv
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.plotting.style import apply_style, get_color, get_linestyle

apply_style()


def load_metrics(results_dir, pattern="*.csv"):
    """Load all CSV metric files from results directory."""
    data = defaultdict(list)
    for csv_file in sorted(glob.glob(os.path.join(results_dir, pattern))):
        basename = os.path.basename(csv_file).replace(".csv", "")
        rows = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) if k != "optimizer" else v for k, v in row.items()})
        if rows:
            data[basename] = rows
    return data


def plot_val_loss_vs_steps(data, output_path, title="Val Loss vs Steps", target_loss=None):
    """Plot 1a: Validation loss vs training steps."""
    fig, ax = plt.subplots()

    for name, rows in data.items():
        # Extract optimizer name from filename
        opt_name = name.split("_")[0] if "_" in name else name
        steps = [r.get("step", i) for i, r in enumerate(rows)]
        val_losses = [r["val_loss"] for r in rows if "val_loss" in r]

        if val_losses:
            ax.plot(steps[:len(val_losses)], val_losses,
                    color=get_color(opt_name),
                    linestyle=get_linestyle(opt_name),
                    label=opt_name, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Loss")
    ax.set_title(title)

    if target_loss is not None:
        ax.axhline(y=target_loss, color='k', linestyle='--', alpha=0.3, label=f'Target ({target_loss})')

    ax.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_val_loss_vs_wallclock(data, output_path, title="Val Loss vs Wall-Clock Time"):
    """Plot 1b: Validation loss vs cumulative wall-clock time."""
    fig, ax = plt.subplots()

    for name, rows in data.items():
        opt_name = name.split("_")[0] if "_" in name else name

        # Accumulate step times
        cumulative_time = []
        val_losses = []
        total_time = 0
        for r in rows:
            if "step_time_ms" in r:
                total_time += r["step_time_ms"] / 1000.0  # Convert to seconds
            if "val_loss" in r:
                cumulative_time.append(total_time)
                val_losses.append(r["val_loss"])

        if val_losses:
            ax.plot(cumulative_time, val_losses,
                    color=get_color(opt_name),
                    linestyle=get_linestyle(opt_name),
                    label=opt_name, linewidth=2)

    ax.set_xlabel("Wall-Clock Time (seconds)")
    ax.set_ylabel("Validation Loss")
    ax.set_title(title)
    ax.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_steps_to_target(summary_data, output_path, title="Steps to Target"):
    """Plot 1d: Bar chart of steps to target."""
    fig, ax = plt.subplots(figsize=(6, 4))

    opts = list(summary_data.keys())
    steps = list(summary_data.values())
    colors = [get_color(o) for o in opts]

    bars = ax.bar(opts, steps, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, steps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                str(val), ha='center', va='bottom', fontsize=11)

    ax.set_ylabel("Steps to Target")
    ax.set_title(title)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate convergence plots")
    parser.add_argument("--results-dir", required=True, help="Directory with CSV results")
    parser.add_argument("--output-dir", default=None, help="Output directory for figures")
    parser.add_argument("--target-loss", type=float, default=None, help="Target loss threshold")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.results_dir), "figures")
    os.makedirs(output_dir, exist_ok=True)

    data = load_metrics(args.results_dir)

    if data:
        plot_val_loss_vs_steps(data, os.path.join(output_dir, "val_loss_vs_steps.png"),
                               target_loss=args.target_loss)
        plot_val_loss_vs_wallclock(data, os.path.join(output_dir, "val_loss_vs_wallclock.png"))
    else:
        print("No CSV data found in", args.results_dir)


if __name__ == "__main__":
    main()
