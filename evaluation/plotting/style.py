"""
Consistent plotting style for CUM optimizer benchmarks.
"""

import matplotlib.pyplot as plt

OPTIMIZER_COLORS = {
    "AdamW": "#1f77b4",       # blue
    "SGD+Nesterov": "#7f7f7f", # gray
    "Muon": "#ff7f0e",        # orange
    "MuonClip": "#d62728",    # red
    "CUM": "#2ca02c",         # green (ours — should stand out)
    "SOAP": "#9467bd",        # purple
}

OPTIMIZER_LINESTYLES = {
    "AdamW": "--",
    "SGD+Nesterov": ":",
    "Muon": "-.",
    "MuonClip": "-.",
    "CUM": "-",               # solid (ours)
    "SOAP": "--",
}

OPTIMIZER_MARKERS = {
    "AdamW": "s",
    "SGD+Nesterov": "^",
    "Muon": "o",
    "MuonClip": "D",
    "CUM": "*",
    "SOAP": "P",
}


def apply_style():
    """Apply consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_color(optimizer_name):
    return OPTIMIZER_COLORS.get(optimizer_name, "#333333")


def get_linestyle(optimizer_name):
    return OPTIMIZER_LINESTYLES.get(optimizer_name, "-")


def get_marker(optimizer_name):
    return OPTIMIZER_MARKERS.get(optimizer_name, "o")
