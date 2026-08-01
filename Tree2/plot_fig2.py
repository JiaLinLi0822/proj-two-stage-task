#!/usr/bin/env python3
"""
Plot Fig2 from data saved by fig2.jl (PDA vs Analytical likelihood agreement).
Reads CSVs from results/fig2/ and saves Fig2.png to figures/.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "results" / "fig2"
DEFAULT_OUT_PATH = SCRIPT_DIR / "figures" / "Fig2.svg"

FONT_GUIDE = 12
FONT_TICK = 10

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
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

def main():
    parser = argparse.ArgumentParser(description="Plot Fig2 from fig2.jl output CSVs")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory with panel_ab.csv, panel_c.csv, panel_d.csv, meta.csv (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Output figure path (default: {DEFAULT_OUT_PATH})",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    out_path = args.out

    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}. Run fig2.jl first.")

    ab = pd.read_csv(data_dir / "panel_ab.csv")
    pc = pd.read_csv(data_dir / "panel_c.csv")
    pd_d = pd.read_csv(data_dir / "panel_d.csv")
    meta = pd.read_csv(data_dir / "meta.csv").iloc[0]

    lims_min = meta["lims_min"]
    lims_max = meta["lims_max"]
    r, mae, rmse_all = meta["r"], meta["mae"], meta["rmse_all"]
    bias, loa_lo, loa_hi = meta["bias"], meta["loa_lo"], meta["loa_hi"]
    theta1_fit = meta["theta1_fit"]
    slice_wid = meta["slice_wid"]

    fig, axes = plt.subplots(2, 2, figsize=(5.75, 5), squeeze=True)  
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.xaxis.get_label().set_fontsize(FONT_GUIDE)
        ax.yaxis.get_label().set_fontsize(FONT_GUIDE)

    # Panel A: PDA vs Analytical
    ax = axes[0, 0]
    ax.scatter(ab["pda"], ab["ana"], s=2, alpha=0.25)
    ax.plot([lims_min, lims_max], [lims_min, lims_max], lw=1.5, ls=":", color="gray")
    ax.set_xlim(lims_min, lims_max)
    ax.set_ylim(lims_min, lims_max)
    ax.set_xlabel("PDA log-likelihood")
    ax.set_ylabel("Analytical log-likelihood")
    ax.text(
        lims_min + 0.06 * (lims_max - lims_min),
        lims_max - 0.12 * (lims_max - lims_min),
        f"r = {r:.3f}\nMAE = {mae:.3g}\nRMSE = {rmse_all:.3g}",
        fontsize=FONT_TICK,
        va="top",
    )

    # Panel B: Bland–Altman
    ax = axes[0, 1]
    ax.scatter(ab["mean_ll"], ab["diff"], s=2, alpha=0.25)
    ax.axhline(bias, color="black", lw=2)
    ax.axhline(loa_hi, color="gray", ls="--", lw=1.5)
    ax.axhline(loa_lo, color="gray", ls="--", lw=1.5)
    ax.set_xlabel("Mean log-likelihood")
    ax.set_ylabel("PDA − Analytical")
    xB = ab["mean_ll"].min() + 0.06 * (ab["mean_ll"].max() - ab["mean_ll"].min())
    yB = ab["diff"].max() - 0.12 * (ab["diff"].max() - ab["diff"].min())
    ax.text(xB, yB, f"bias = {bias:.3g}\nLoA = [{loa_lo:.3g}, {loa_hi:.3g}]", fontsize=FONT_TICK, va="top")

    # Panel C: θ1 slice
    ax = axes[1, 0]
    ax.plot(pc["grid_theta1"], pc["ll_ana"], lw=2, label="Analytical")
    ax.plot(pc["grid_theta1"], pc["ll_pda"], lw=2, ls="--", label="PDA")
    ax.axvline(theta1_fit, lw=1.5, ls=":", color="gray", label=r"Fitted $\theta_1$")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel("Total log-likelihood")
    ax.legend(loc="lower right", fontsize=FONT_TICK)
    # ax.text(0.5, 0.03, f"Subject: {slice_wid}", fontsize=FONT_TICK, ha="center", va="top",
    #         transform=ax.transAxes)

    # Panel D: Per-subject RMSE
    ax = axes[1, 1]
    xs = pd_d["rank"].values
    ys = pd_d["rmse"].values
    ax.scatter(xs, ys, s=16, alpha=0.9)
    ax.plot(xs, ys, lw=2)
    ax.set_xlabel("Subjects (sorted by RMSE)")
    ax.set_ylabel("RMSE of Δ log-likelihood")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=500)
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
