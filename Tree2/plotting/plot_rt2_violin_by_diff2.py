#!/usr/bin/env python3
"""
Half violin plot: x-axis = diff2, y-axis = reaction time (rt2).
At each diff2 level, Human / RSS / PDA three distributions overlaid (same position) with transparency.

Usage (from project root):
    python Tree2/plot_rt2_violin_by_diff2.py
Or from Tree2/:
    python plot_rt2_violin_by_diff2.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths: support both project root and Tree2/ as cwd
if os.path.isfile("Tree2/data/Tree2.json"):
    HUMAN_FILE = "Tree2/data/Tree2.json"
    RSS_FILE = "Tree2/data/rss/model1.json"
    PDA_FILE = "Tree2/data/pda/model1_cleaned.json"
else:
    HUMAN_FILE = "data/Tree2.json"
    RSS_FILE = "data/rss/model1.json"
    PDA_FILE = "data/pda/model1_cleaned.json"


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_df(records, source):
    df = pd.DataFrame([
        {"diff2": r["diff2"], "rt2": float(r["rt2"])}
        for r in records
        if "diff2" in r and "rt2" in r and r.get("rt2", None) is not None
    ])
    df["source"] = source
    return df


def _clip_violin_to_half(bodies, positions, half="left"):
    """Clip each violin body to left or right half. half in ('left','right')."""
    for body, pos in zip(bodies, positions):
        path = body.get_paths()[0]
        verts = path.vertices.copy()
        x, y = verts[:, 0], verts[:, 1]
        if half == "left":
            # keep x <= pos, clip right side to line x=pos
            x_new = np.minimum(x, pos)
        else:
            x_new = np.maximum(x, pos)
        body.get_paths()[0].vertices[:, 0] = x_new


def main():
    human_records = load_jsonl(HUMAN_FILE)
    rss_records = load_jsonl(RSS_FILE)
    pda_records = load_jsonl(PDA_FILE)

    df_h = to_df(human_records, "Human")
    df_rss = to_df(rss_records, "RSS")
    df_pda = to_df(pda_records, "PDA")

    # Limit diff2 to top N unique values
    all_diff2 = sorted(
        set(df_h["diff2"].dropna().unique())
        | set(df_rss["diff2"].dropna().unique())
        | set(df_pda["diff2"].dropna().unique())
    )
    unique_diff2 = (all_diff2 or [0.0])[:10]

    # At each diff2, one x position; three violins (Human, RSS, PDA) at the same position → 重合
    sources = ["Human", "RSS", "PDA"]
    colors = {"Human": "C0", "RSS": "C1", "PDA": "C2"}
    n_diff = len(unique_diff2)
    positions = []
    data_arrays = []
    color_list = []

    for i, d in enumerate(unique_diff2):
        for src in sources:
            if src == "Human":
                df_src = df_h
            elif src == "RSS":
                df_src = df_rss
            else:
                df_src = df_pda
            sub = df_src[df_src["diff2"] == d]["rt2"].values
            sub = sub[np.isfinite(sub)]
            if len(sub) == 0:
                sub = np.array([np.nan])
            positions.append(i)  # same position for all three at this diff2
            data_arrays.append(sub)
            color_list.append(colors[src])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Human histogram at each diff2 level (draw first so behind violins)
    rt2_min, rt2_max = 0.0, 10000.0
    n_bins = 50
    hist_scale = 0.25  # max horizontal extent of histogram bar from position (in x units)
    color_human_hist = "C0"
    for i, d in enumerate(unique_diff2):
        h_rt2 = df_h[df_h["diff2"] == d]["rt2"].values
        h_rt2 = h_rt2[np.isfinite(h_rt2)]
        if len(h_rt2) < 2:
            continue
        dens, bin_edges = np.histogram(h_rt2, bins=n_bins, range=(rt2_min, rt2_max), density=True)
        bin_h = bin_edges[1] - bin_edges[0]
        # horizontal bars: from x=i extending right by density (scaled)
        max_d = dens.max() if dens.max() > 0 else 1.0
        widths = dens * (hist_scale / max_d)
        ax.barh(
            bin_edges[:-1],
            widths,
            height=bin_h,
            left=i,
            color=color_human_hist,
            alpha=0.4,
            edgecolor=color_human_hist,
            linewidth=0.3,
        )

    w = 0.65  # width so overlapping violins are visible
    parts = ax.violinplot(
        data_arrays,
        positions=positions,
        widths=w,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    alpha = 0.5
    for pc in parts["bodies"]:
        pc.set_alpha(alpha)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)

    for i, (pc, pos) in enumerate(zip(parts["bodies"], positions)):
        pc.set_facecolor(color_list[i])
        _clip_violin_to_half([pc], [pos], half="left")

    ax.set_xticks(range(n_diff))
    ax.set_xticklabels([str(float(d)) for d in unique_diff2], rotation=45)
    ax.set_xlim(-0.6, n_diff - 0.4)
    ax.set_xlabel("diff2")
    ax.set_ylabel("Reaction time rt2 (ms)")
    ax.set_title("rt2 by diff2: Human vs RSS vs PDA (overlaid half violins)")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[s], alpha=alpha, edgecolor="black", label=s)
        for s in sources
    ]
    legend_elements.append(
        Patch(facecolor=color_human_hist, alpha=0.4, edgecolor=color_human_hist, label="Human (hist)")
    )
    ax.legend(handles=legend_elements, title="Source")
    plt.tight_layout()

    out_dir = "Tree2/figures" if os.path.isdir("Tree2/figures") else "figures"
    out_path = os.path.join(out_dir, "rt2_violin_by_diff2.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
