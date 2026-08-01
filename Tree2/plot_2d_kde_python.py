#!/usr/bin/env python3
"""
Simulate trial-specific RT1/RT2 samples with Julia, then plot a 2D joint KDE
with marginal distributions, following Tree2/plot_participant_kde.jl.

Examples:
    python Tree2/plot_2d_kde_python.py
    python Tree2/plot_2d_kde_python.py --participant w6eb2a0a --trial 68 --samples 20000
    python Tree2/plot_2d_kde_python.py --samples-csv Tree2/figures/python_2d_kde_samples.csv
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde


def default_tree2_dir() -> Path:
    if Path("Tree2/export_simulated_kde_samples.jl").is_file():
        return Path("Tree2")
    return Path(".")


def default_output_path() -> Path:
    if Path("Tree2/figures").is_dir() or Path("Tree2").is_dir():
        return Path("Tree2/figures/python_2d_kde_logrt.png")
    return Path("figures/python_2d_kde_logrt.png")


def parse_bw(value: str):
    if value in {"scott", "silverman"}:
        return value
    return float(value)


def finite_rt_frame(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Input must contain columns '{x_col}' and '{y_col}'.")
    out = df.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out = out[np.isfinite(out[x_col]) & np.isfinite(out[y_col])]
    out = out[(out[x_col] > 0) & (out[y_col] > 0)]
    return out


def padded_range(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = max((hi - lo) * pad_fraction, 1e-9)
    return lo - pad, hi + pad


def nice_ms_ticks(raw_values: np.ndarray, *, max_ticks: int = 4) -> np.ndarray:
    lo = float(np.min(raw_values))
    hi = float(np.max(raw_values))
    candidates = np.array([500, 1000, 2000, 4000, 6000, 8000, 10000, 12000], dtype=float)
    ticks = candidates[(candidates >= lo) & (candidates <= hi)]
    if len(ticks) >= 2:
        return ticks[:max_ticks] if len(ticks) > max_ticks else ticks
    return np.linspace(lo, hi, min(max_ticks, 3))


def run_julia_simulation(args: argparse.Namespace, samples_csv: Path) -> None:
    exporter = args.tree2_dir / "export_simulated_kde_samples.jl"
    if not exporter.is_file():
        raise FileNotFoundError(f"Could not find Julia exporter: {exporter}")

    cmd = [
        args.julia,
        f"--project={args.tree2_dir}",
        str(exporter),
        "--model", args.model,
        "--participant", args.participant,
        "--trial", str(args.trial),
        "--samples", str(args.samples),
        "--data-file", str(args.data_file),
        "--params-file", str(args.params_file),
        "--output", str(samples_csv),
    ]
    print("Running Julia simulation:", flush=True)
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare_plot_data(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    log_space: bool,
    human_rt1: float | None,
    human_rt2: float | None,
):
    if log_space:
        x = np.log(x_raw)
        y = np.log(y_raw)
        human_x = None if human_rt1 is None else np.log(human_rt1)
        human_y = None if human_rt2 is None else np.log(human_rt2)
        x_label = "RT1 (ms)"
        y_label = "RT2 (ms)"
    else:
        x = x_raw
        y = y_raw
        human_x = human_rt1
        human_y = human_rt2
        x_label = "RT1 (ms)"
        y_label = "RT2 (ms)"
    return x, y, human_x, human_y, x_label, y_label


def set_ms_ticks(ax, *, x_raw: np.ndarray, y_raw: np.ndarray, log_space: bool) -> None:
    if not log_space:
        return
    xticks_ms = nice_ms_ticks(x_raw)
    yticks_ms = nice_ms_ticks(y_raw)
    ax.set_xticks(np.log(xticks_ms))
    ax.set_xticklabels([f"{int(t)}" for t in xticks_ms])
    ax.set_yticks(np.log(yticks_ms))
    ax.set_yticklabels([f"{int(t)}" for t in yticks_ms])


def plot_joint_2d_kde(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    output: Path,
    *,
    human_rt1: float | None = None,
    human_rt2: float | None = None,
    log_space: bool = True,
    grid_n: int = 160,
    bw_method="silverman",
    bins: int = 50,
    max_scatter: int = 1800,
    seed: int = 20260504,
    cmap: str = "viridis",
    title: str = "",
) -> None:
    x, y, human_x, human_y, x_label, y_label = prepare_plot_data(
        x_raw, y_raw, log_space=log_space, human_rt1=human_rt1, human_rt2=human_rt2
    )
    if len(x) < 3:
        raise ValueError(f"Need at least 3 simulated RT pairs; got {len(x)}.")

    kde2d = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    kde_x = gaussian_kde(x, bw_method=bw_method)
    kde_y = gaussian_kde(y, bw_method=bw_method)

    xmin, xmax = padded_range(x)
    ymin, ymax = padded_range(y)
    x_grid = np.linspace(xmin, xmax, grid_n)
    y_grid = np.linspace(ymin, ymax, grid_n)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde2d(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    dens_x = kde_x(x_grid)
    dens_y = kde_y(y_grid)

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig = plt.figure(figsize=(6.2, 5.3), dpi=300)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4.5, 1.15],
        height_ratios=[1.15, 4.5],
        hspace=0.04,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    contour = ax_main.contourf(x_grid, y_grid, zz, levels=22, cmap=cmap)
    ax_main.contour(x_grid, y_grid, zz, levels=8, colors="white", linewidths=0.35, alpha=0.35)

    if len(x) > max_scatter:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), size=max_scatter, replace=False))
    else:
        idx = np.arange(len(x))
    ax_main.scatter(
        x[idx], y[idx],
        s=7,
        marker="x",
        color="white",
        alpha=0.28,
        linewidths=0.45,
        label="Simulated samples",
    )
    if human_x is not None and human_y is not None:
        ax_main.scatter(
            [human_x], [human_y],
            marker="*",
            s=150,
            color="red",
            edgecolor="white",
            linewidth=1.1,
            label="Human RT",
            zorder=5,
        )

    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(ymin, ymax)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper right", frameon=True, framealpha=0.92)
    if title:
        ax_main.set_title(title, pad=8)
    set_ms_ticks(ax_main, x_raw=x_raw, y_raw=y_raw, log_space=log_space)

    ax_top.hist(
        x,
        bins=bins,
        range=(xmin, xmax),
        density=True,
        color="#9ecae1",
        alpha=0.35,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_top.plot(x_grid, dens_x, color="#084594", linewidth=1.7)
    if human_x is not None:
        ax_top.axvline(human_x, color="red", linewidth=1.8)
    ax_top.set_xlim(xmin, xmax)
    ax_top.axis("off")

    ax_right.hist(
        y,
        bins=bins,
        range=(ymin, ymax),
        density=True,
        orientation="horizontal",
        color="#a1d99b",
        alpha=0.35,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_right.plot(dens_y, y_grid, color="#006d2c", linewidth=1.7)
    if human_y is not None:
        ax_right.axhline(human_y, color="red", linewidth=1.8)
    ax_right.set_ylim(ymin, ymax)
    ax_right.axis("off")

    cbar = fig.colorbar(contour, ax=[ax_main, ax_right], fraction=0.035, pad=0.035)
    cbar.set_label("KDE density")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate RT samples with Julia, then draw a 2D joint KDE plot.")
    parser.add_argument("--output", type=Path, default=default_output_path(), help="Output image path.")
    parser.add_argument("--samples-csv", type=Path, default=None, help="Optional path to keep the Julia-exported simulated samples CSV.")
    parser.add_argument("--tree2-dir", type=Path, default=default_tree2_dir(), help="Tree2 directory containing Julia model scripts.")
    parser.add_argument("--julia", default="julia", help="Julia executable.")
    parser.add_argument("--model", default="model6", help="Model name used by model_configs.jl.")
    parser.add_argument("--participant", default="w6eb2a0a", help="Participant id for target trial and fitted parameters.")
    parser.add_argument("--trial", type=int, default=68, help="1-based target trial index.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of Julia simulations before choice-pair filtering.")
    parser.add_argument("--data-file", type=Path, default=default_tree2_dir() / "data" / "Tree2_v3.json", help="Participant data JSON.")
    parser.add_argument("--params-file", type=Path, default=default_tree2_dir() / "results" / "pda" / "model6_pda_BADS_20260125_211706.csv", help="Fitted parameter CSV.")
    parser.add_argument("--x-col", default="rt1", help="Column/key for first-stage RT.")
    parser.add_argument("--y-col", default="rt2", help="Column/key for second-stage RT.")
    parser.add_argument("--raw-space", action="store_true", help="Plot in raw milliseconds instead of log(RT).")
    parser.add_argument("--grid-n", type=int, default=160, help="Number of KDE grid points per axis.")
    parser.add_argument("--bw", default="silverman", help="KDE bandwidth: scott, silverman, or a float.")
    parser.add_argument("--bins", type=int, default=50, help="Marginal histogram bins.")
    parser.add_argument("--max-scatter", type=int, default=1800, help="Max simulated samples shown as scatter points.")
    parser.add_argument("--seed", type=int, default=20260504, help="Scatter subsampling seed.")
    parser.add_argument("--title", default="", help="Optional title shown above the main KDE panel.")
    args = parser.parse_args()

    if args.samples_csv is None:
        with tempfile.TemporaryDirectory(prefix="simulated_2d_kde_") as tmpdir:
            samples_csv = Path(tmpdir) / "samples.csv"
            run_julia_simulation(args, samples_csv)
            df = finite_rt_frame(pd.read_csv(samples_csv), args.x_col, args.y_col)
            plot_from_frame(df, args)
    else:
        run_julia_simulation(args, args.samples_csv)
        df = finite_rt_frame(pd.read_csv(args.samples_csv), args.x_col, args.y_col)
        plot_from_frame(df, args)


def plot_from_frame(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if len(df) < 3:
        raise ValueError(f"Need at least 3 finite positive simulated RT pairs; got {len(df)}.")

    human_rt1 = float(df["human_rt1"].iloc[0]) if "human_rt1" in df.columns else None
    human_rt2 = float(df["human_rt2"].iloc[0]) if "human_rt2" in df.columns else None
    plot_joint_2d_kde(
        df[args.x_col].to_numpy(dtype=float),
        df[args.y_col].to_numpy(dtype=float),
        args.output,
        human_rt1=human_rt1,
        human_rt2=human_rt2,
        log_space=not args.raw_space,
        grid_n=args.grid_n,
        bw_method=parse_bw(args.bw),
        bins=args.bins,
        max_scatter=args.max_scatter,
        seed=args.seed,
        title=args.title,
    )
    print(f"Plotted {len(df)} matching simulated RT pairs")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
