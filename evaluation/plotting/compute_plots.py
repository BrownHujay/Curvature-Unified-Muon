"""
Plot Set 2: Compute Efficiency Plots
- 2a: Optimizer Step Time Breakdown (Stacked Bar)
- 2b: Throughput vs Training Step
- 2c: GPU Memory Usage Bar Chart
"""

import os
import sys
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.plotting.style import apply_style, get_color

apply_style()


def plot_step_time_breakdown(data, output_path):
    """Plot 2a: Stacked bar chart of optimizer step time components."""
    fig, ax = plt.subplots(figsize=(8, 5))

    optimizers = list(data.keys())
    components = ["momentum_ms", "precond_ms", "ns_ms", "spectral_ms", "update_ms"]
    component_labels = ["Momentum", "Preconditioning", "Newton-Schulz", "Spectral Control", "Weight Update"]
    component_colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"]

    x = np.arange(len(optimizers))
    width = 0.6

    bottom = np.zeros(len(optimizers))
    for comp, label, color in zip(components, component_labels, component_colors):
        values = [data[opt].get(comp, 0) for opt in optimizers]
        ax.bar(x, values, width, bottom=bottom, label=label, color=color)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Optimizer Step Time Breakdown")
    ax.legend(loc="upper right")

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_throughput_vs_step(data, output_path):
    """Plot 2b: Tokens/second vs training step."""
    fig, ax = plt.subplots()

    for opt_name, metrics in data.items():
        steps = [m["step"] for m in metrics]
        throughput = [m.get("tokens_per_sec", 0) for m in metrics]

        if any(t > 0 for t in throughput):
            ax.plot(steps, throughput, color=get_color(opt_name), label=opt_name, linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Tokens/Second")
    ax.set_title("Training Throughput")
    ax.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def plot_memory_usage(data, output_path):
    """Plot 2c: Peak GPU memory per optimizer."""
    fig, ax = plt.subplots(figsize=(6, 4))

    optimizers = list(data.keys())
    memory_gb = [data[opt].get("peak_memory_gb", 0) for opt in optimizers]
    colors = [get_color(opt) for opt in optimizers]

    bars = ax.bar(optimizers, memory_gb, color=colors, edgecolor='white')

    for bar, val in zip(bars, memory_gb):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("GPU Memory Usage")

    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate compute efficiency plots")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.results_dir), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Compute plots require structured metric data.")
    print("Run benchmarks first, then point --results-dir to the output.")


if __name__ == "__main__":
    main()
