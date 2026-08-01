#!/usr/bin/env python3
"""
Plot overall rt1 distribution: Human histogram + KDE curves for
Human, TS_RSS, TS_PDA, FG_RSS, FG_PDA (TS = two-stage model1, FG = forward greedy model6).

Usage (from project root):
    python Tree2/plot_rt1_kde_overall.py
Or from Tree2/:
    python plot_rt1_kde_overall.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Paths: support both project root and Tree2/ as cwd
if os.path.isfile("Tree2/data/Tree2.json"):
    PREFIX = "Tree2/data"
else:
    PREFIX = "data"

# (source key for color, label, jsonl path)
SOURCES = [
    ("Human", "Human", f"{PREFIX}/Tree2.json"),
    ("TS_RSS", "TS_RSS", f"{PREFIX}/rss/model1.json"),
    ("TS_PDA", "TS_PDA", f"{PREFIX}/pda/model1_cleaned.json"),
    ("FG_RSS", "FG_RSS", f"{PREFIX}/rss/model6_RSS.json"),
    ("FG_PDA", "FG_PDA", f"{PREFIX}/pda/model6.json"),
]

# SOURCE_COLORS = {
#     "Human": "black",
#     "TS_RSS": "#335372",
#     "TS_PDA": "#4F7BA8",
#     "FG_RSS": "#E25659",
#     "FG_PDA": "#EB8386",
# }

SOURCE_COLORS = {
    'Human': '#1072BD',
    'TS_RSS': '#77AE43',
    'TS_PDA': '#EDB021',
    'FG_RSS': '#D7592C',
    'FG_PDA': '#7F318D',
}


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_df_rt1(records):
    return pd.DataFrame([
        {"rt1": float(r["rt1"])}
        for r in records
        if "rt1" in r and r.get("rt1", None) is not None
    ])


def plot_kde_or_mean(ax, x_grid, values, color, label, linestyle="-"):
    if len(values) >= 2:
        try:
            kde = gaussian_kde(values)
            dens = kde(x_grid)
            ax.plot(x_grid, dens, color=color, linewidth=1.8, label=label, linestyle=linestyle)
            ax.fill_between(x_grid, dens, alpha=0.15, color=color)
        except np.linalg.LinAlgError:
            ax.axvline(np.mean(values), color=color, linestyle="--", linewidth=1.5, label=label)
    elif len(values) == 1:
        ax.axvline(values[0], color=color, linestyle="--", linewidth=1.5, label=label)


def main():
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.75,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'legend.loc': 'upper right'
    })

    rt1_min, rt1_max = 0.0, 10000.0
    n_grid = 200
    x_grid = np.linspace(rt1_min, rt1_max, n_grid)
    num_bins = 50

    # Load all sources
    data = {}
    for key, label, path in SOURCES:
        records = load_jsonl(path)
        df = to_df_rt1(records)
        data[key] = df["rt1"].dropna().values if len(df) > 0 else np.array([])

    fig, ax = plt.subplots(figsize=(6.7, 2.65))

    # Human: histogram + KDE curve
    h_all = data["Human"]
    color_h = SOURCE_COLORS["Human"]
    if len(h_all) > 0:
        ax.hist(
            h_all,
            bins=num_bins,
            range=(rt1_min, rt1_max),
            density=True,
            color=color_h,
            alpha=0.15,
            edgecolor=color_h,
            linewidth=0.5,
            label="Human (hist)",
        )
    plot_kde_or_mean(ax, x_grid, h_all, color_h, "Human (Gaussian KDE)", "-")

    # Model KDEs: TS_RSS, TS_PDA, FG_RSS, FG_PDA
    linestyles = {"TS_RSS": "-", "TS_PDA": "--", "FG_RSS": "-", "FG_PDA": "--"}
    for key in ["TS_RSS", "TS_PDA", "FG_RSS", "FG_PDA"]:
        vals = data[key]
        if len(vals) == 0:
            continue
        plot_kde_or_mean(ax, x_grid, vals, SOURCE_COLORS[key], key, linestyles[key])

    ax.set_xlim(rt1_min, rt1_max)
    ax.set_ylim(0, None)
    ax.set_xlabel("First stage RT (ms)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend(fontsize=10, ncol=1)
    # ax.grid(False, alpha=0.3)

    plt.tight_layout()
    out_path = "Tree2/figures/Fig5A.svg" if os.path.isdir("Tree2/figures") else "figures/Fig5A.svg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
