#!/usr/bin/env python3
"""
Plot rt2 distribution by diff2 level: Gaussian KDE for human vs RSS vs PDA simulated data.
4 rows x 2 columns; human, RSS, and PDA in three colors per panel.

Parameters (edit in script):
  PLOT_RSS_HIST: if True, draw RSS histogram (default False).
  PLOT_PDA_HIST: if True, draw PDA histogram (default True).
  PLOT_RSS_KDE: if True, draw RSS KDE (default True).
Human histogram/KDE and PDA KDE are always plotted.

Usage (from project root):
    python Tree2/plot_rt2_kde_by_diff2.py
Or from Tree2/:
    python plot_rt2_kde_by_diff2.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Paths: support both project root and Tree2/ as cwd
if os.path.isfile("Tree2/data/Tree2.json"):
    HUMAN_FILE = "Tree2/data/Tree2.json"
    RSS_FILE = "Tree2/data/rss/model1.json"
    PDA_FILE = "Tree2/data/pda/model1_cleaned.json"
else:
    HUMAN_FILE = "data/Tree2.json"
    RSS_FILE = "data/rss/model1.json"
    PDA_FILE = "data/pda/model1_cleaned.json"

# Whether to plot histogram / KDE for RSS and PDA
PLOT_RSS_HIST = True
PLOT_PDA_HIST = True
PLOT_RSS_KDE = False


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_df(records):
    return pd.DataFrame([
        {"diff2": r["diff2"], "rt2": float(r["rt2"])}
        for r in records
        if "diff2" in r and "rt2" in r and r.get("rt2", None) is not None
    ])


def main():
    human_records = load_jsonl(HUMAN_FILE)
    rss_records = load_jsonl(RSS_FILE)
    pda_records = load_jsonl(PDA_FILE)
    df_h = to_df(human_records)
    df_rss = to_df(rss_records)
    df_pda = to_df(pda_records)

    # Common diff2 levels (union, sorted); we'll plot up to 8
    all_diff2 = sorted(
        set(df_h["diff2"].dropna().unique().tolist())
        | set(df_rss["diff2"].dropna().unique().tolist())
        | set(df_pda["diff2"].dropna().unique().tolist())
    )
    unique_diff2 = (all_diff2 or [0.0])[:8]
    nrows, ncols = 4, 2

    # Shared x grid for KDE (rt2 in ms)
    rt2_min, rt2_max = 0.0, 10000.0
    n_grid = 200
    x_grid = np.linspace(rt2_min, rt2_max, n_grid)

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 8), sharex=True, sharey=False)
    axs = axs.flatten()

    color_human = "C0"   # blue
    color_rss = "C1"     # orange
    color_pda = "C2"     # green
    num_bins = 100

    for i, diff2_val in enumerate(unique_diff2):
        ax = axs[i]
        h_sub = df_h[df_h["diff2"] == diff2_val]["rt2"].values
        rss_sub = df_rss[df_rss["diff2"] == diff2_val]["rt2"].values
        pda_sub = df_pda[df_pda["diff2"] == diff2_val]["rt2"].values

        # Human histogram (density scale, drawn first so KDE is on top)
        if len(h_sub) > 0:
            ax.hist(h_sub, bins=num_bins, range=(rt2_min, rt2_max), density=True,
                   color=color_human, alpha=0.35, edgecolor=color_human, linewidth=0.5,
                   label="Human (hist)")

        # Gaussian KDE for human
        if len(h_sub) >= 2:
            kde_h = gaussian_kde(h_sub)
            try:
                dens_h = kde_h(x_grid)
                ax.fill_between(x_grid, dens_h, alpha=0.3, color=color_human)
                ax.plot(x_grid, dens_h, color=color_human, linewidth=2, label="Human (KDE)")
            except np.linalg.LinAlgError:
                ax.axvline(np.mean(h_sub), color=color_human, linestyle="--", linewidth=2, label="Human (mean)")
        elif len(h_sub) == 1:
            ax.axvline(h_sub[0], color=color_human, linestyle="--", linewidth=2, label="Human (n=1)")

        # RSS histogram (density scale, drawn before KDE so KDE is on top)
        if PLOT_RSS_HIST and len(rss_sub) > 0:
            ax.hist(rss_sub, bins=num_bins, range=(rt2_min, rt2_max), density=True,
                   color=color_rss, alpha=0.35, edgecolor=color_rss, linewidth=0.5,
                   label="RSS (hist)")

        # RSS KDE
        if PLOT_RSS_KDE and len(rss_sub) >= 2:
            kde_rss = gaussian_kde(rss_sub)
            try:
                dens_rss = kde_rss(x_grid)
                ax.fill_between(x_grid, dens_rss, alpha=0.3, color=color_rss)
                ax.plot(x_grid, dens_rss, color=color_rss, linewidth=2, label="RSS (KDE)")
            except np.linalg.LinAlgError:
                ax.axvline(np.mean(rss_sub), color=color_rss, linestyle=":", linewidth=2, label="RSS (mean)")
        elif PLOT_RSS_KDE and len(rss_sub) == 1:
            ax.axvline(rss_sub[0], color=color_rss, linestyle=":", linewidth=2, label="RSS (n=1)")

        # PDA histogram (density scale, drawn before KDE so KDE is on top)
        if PLOT_PDA_HIST and len(pda_sub) > 0:
            ax.hist(pda_sub, bins=num_bins, range=(rt2_min, rt2_max), density=True,
                   color=color_pda, alpha=0.35, edgecolor=color_pda, linewidth=0.5,
                   label="PDA (hist)")

        # PDA KDE
        if len(pda_sub) >= 2:
            kde_pda = gaussian_kde(pda_sub)
            try:
                dens_pda = kde_pda(x_grid)
                ax.fill_between(x_grid, dens_pda, alpha=0.3, color=color_pda)
                ax.plot(x_grid, dens_pda, color=color_pda, linewidth=2, label="PDA (KDE)")
            except np.linalg.LinAlgError:
                ax.axvline(np.mean(pda_sub), color=color_pda, linestyle="-.", linewidth=2, label="PDA (mean)")
        elif len(pda_sub) == 1:
            ax.axvline(pda_sub[0], color=color_pda, linestyle="-.", linewidth=2, label="PDA (n=1)")

        ax.set_xlim(rt2_min, rt2_max)
        ax.set_ylabel("Density")
        ax.set_title(f"diff2 = {diff2_val}")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(unique_diff2), nrows * ncols):
        fig.delaxes(axs[j])

    # x-label on bottom row of used panels
    start_bottom = (len(unique_diff2) - 1) // ncols * ncols
    for idx in range(start_bottom, len(unique_diff2)):
        axs[idx].set_xlabel("rt2 (ms)")

    plt.tight_layout()
    out_path = "Tree2/figures/rt2_kde_by_diff2_human_rss_pda.png" if os.path.isdir("Tree2/figures") else "figures/rt2_kde_by_diff2_human_rss_pda.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
