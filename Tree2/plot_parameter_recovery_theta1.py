#!/usr/bin/env python3
"""
Plot θ1 parameter recovery (ground truth vs fitted) for analytical, PDA, RSS, IBS.
Reads latest CSV from results/parameter_recovery/ for each method; 2×2 layout.
"""

import glob
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Default: run from repo root or Tree2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "parameter_recovery")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "figures", "Fig3.png")

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.linewidth': 0.75,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.loc': 'upper right'
})

# Parameter name: CSV may have "θ1" (Unicode) or "theta1"
TRUE_COL = "true_θ1"
FITTED_COL = "fitted_θ1"
# Fallback if CSV uses ASCII
TRUE_COL_ALT = "true_theta1"
FITTED_COL_ALT = "fitted_theta1"


def find_latest_csv(pattern: str) -> Optional[str]:
    """Return path to most recently modified CSV matching pattern."""
    full_pattern = os.path.join(RESULTS_DIR, pattern)
    files = glob.glob(full_pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_theta1_columns(df: pd.DataFrame):
    """Return (true_col, fitted_col) for θ1, with fallback for naming."""
    if TRUE_COL in df.columns and FITTED_COL in df.columns:
        return TRUE_COL, FITTED_COL
    if TRUE_COL_ALT in df.columns and FITTED_COL_ALT in df.columns:
        return TRUE_COL_ALT, FITTED_COL_ALT
    # Try any column containing theta1 or θ1
    for c in df.columns:
        if "theta1" in c.lower() or "θ1" in c:
            if c.startswith("true"):
                other = "fitted" + c[4:]
                if other in df.columns:
                    return c, other
    raise KeyError(f"No θ1 columns found. Columns: {list(df.columns)}")


def load_recovery_data(method: str) -> Optional[pd.DataFrame]:
    """Load latest parameter recovery CSV for method (analytical, pda, rss, ibs)."""
    pattern = f"parameter_recovery_model6_{method}_*.csv"
    path = find_latest_csv(pattern)
    if path is None:
        return None
    return pd.read_csv(path)


def plot_theta1_recovery(ax, df: pd.DataFrame, title: str):
    """Scatter true θ1 vs fitted θ1 with identity line and correlation."""
    true_col, fitted_col = get_theta1_columns(df)
    x = df[true_col].values
    y = df[fitted_col].values

    ax.scatter(x, y, alpha=1.0, s=60, facecolors="none", edgecolors="black")

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Identity", color="red")

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
    ax.set_xlabel(r"Ground truth $\theta_1$", fontsize=11)
    ax.set_ylabel(r"Fitted $\theta_1$", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    methods = [
        ("analytical", "Analytical"),
        ("pda", "PDA"),
        ("rss", "RSS"),
        ("ibs", "IBS"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5.25, 5.25))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(methods):
        df = load_recovery_data(key)
        if df is None:
            axes[idx].set_title(f"{title} (no data)")
            axes[idx].text(0.5, 0.5, "No CSV found", ha="center", va="center", transform=axes[idx].transAxes)
            continue
        plot_theta1_recovery(axes[idx], df, title)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=500)
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
