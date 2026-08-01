#!/usr/bin/env python3
"""
Simulate RT1/RT2 samples with the Julia model, then plot a 3D Gaussian KDE
surface in log(RT) space.

Examples:
    python Tree2/plot_3d_kde_python.py
    python Tree2/plot_3d_kde_python.py --participant w6eb2a0a --trial 68 --samples 20000
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib cache inside the workspace/tmp when the home cache is locked.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde


def default_output_path() -> Path:
    if Path("Tree2/figures").is_dir() or Path("Tree2").is_dir():
        return Path("Tree2/figures/python_3d_kde_logrt.png")
    return Path("figures/python_3d_kde_logrt.png")


def default_tree2_dir() -> Path:
    if Path("Tree2/export_simulated_kde_samples.jl").is_file():
        return Path("Tree2")
    return Path(".")


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


def parse_bw(value: str):
    if value in {"scott", "silverman"}:
        return value
    return float(value)


def nice_ms_ticks(raw_values: np.ndarray, *, max_ticks: int = 5) -> np.ndarray:
    lo = float(np.min(raw_values))
    hi = float(np.max(raw_values))
    candidates = np.array([500, 1000, 2000, 4000, 6000, 8000, 10000, 12000], dtype=float)
    ticks = candidates[(candidates >= lo) & (candidates <= hi)]
    if len(ticks) >= 2:
        return ticks[:max_ticks] if len(ticks) > max_ticks else ticks
    return np.linspace(lo, hi, min(max_ticks, 3))


def style_schematic_3d_axes(
    ax,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmax: float,
    show_box_grid: bool = False,
    draw_axis_lines: bool = True,
) -> None:
    # Match the schematic: open 3D space, visible base axes, no grey panes.
    ax.grid(show_box_grid)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 0.0 if not show_box_grid else 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["axisline"]["color"] = (0, 0, 0, 1)
        axis._axinfo["axisline"]["linewidth"] = 1.1

    if draw_axis_lines:
        ax.plot([xmin, xmax], [ymin, ymin], [0, 0], color="black", linewidth=1.25)
        ax.plot([xmin, xmin], [ymin, ymax], [0, 0], color="black", linewidth=1.25)
        ax.plot([xmin, xmin], [ymin, ymin], [0, zmax], color="black", linewidth=1.25)
    ax.tick_params(axis="both", which="major", pad=2, length=3, width=0.8)
    ax.tick_params(axis="z", which="major", pad=2, length=3, width=0.8)


def run_julia_simulation(args: argparse.Namespace, samples_csv: Path) -> None:
    tree2_dir = args.tree2_dir
    exporter = tree2_dir / "export_simulated_kde_samples.jl"
    if not exporter.is_file():
        raise FileNotFoundError(f"Could not find Julia exporter: {exporter}")

    cmd = [
        args.julia,
        f"--project={tree2_dir}",
        str(exporter),
        "--model",
        args.model,
        "--participant",
        args.participant,
        "--trial",
        str(args.trial),
        "--samples",
        str(args.samples),
        "--data-file",
        str(args.data_file),
        "--params-file",
        str(args.params_file),
        "--output",
        str(samples_csv),
    ]
    print("Running Julia simulation:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def plot_3d_kde(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    output: Path,
    *,
    log_space: bool = True,
    grid_n: int = 120,
    bw_method="silverman",
    human_rt1: float | None = None,
    human_rt2: float | None = None,
    elev: float = 28,
    azim: float = -55,
    title: str = "3D Gaussian KDE",
    display_ms_ticks: bool = True,
    schematic_axes: bool = True,
    show_box_grid: bool = False,
) -> None:
    if log_space:
        x = np.log(x_raw)
        y = np.log(y_raw)
        x_label = "RT1 (ms)" if display_ms_ticks else "log(RT1)"
        y_label = "RT2 (ms)" if display_ms_ticks else "log(RT2)"
        human_x = None if human_rt1 is None else np.log(human_rt1)
        human_y = None if human_rt2 is None else np.log(human_rt2)
    else:
        x = x_raw
        y = y_raw
        x_label = "RT1 (ms)"
        y_label = "RT2 (ms)"
        human_x = human_rt1
        human_y = human_rt2

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)

    xmin, xmax = padded_range(x)
    ymin, ymax = padded_range(y)
    x_grid = np.linspace(xmin, xmax, grid_n)
    y_grid = np.linspace(ymin, ymax, grid_n)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(6.4, 5.0), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    zmax = float(np.max(zz)) * 1.05
    if schematic_axes:
        style_schematic_3d_axes(
            ax,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmax=zmax,
            show_box_grid=show_box_grid,
            draw_axis_lines=True,
        )

    ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cm.viridis,
        edgecolor=(0, 0, 0, 0.23),
        linewidth=0.18,
        antialiased=True,
        rstride=2,
        cstride=2,
        alpha=0.98,
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_zlabel("KDE density", labelpad=8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0, float(np.max(zz)) * 1.05)
    ax.view_init(elev=elev, azim=azim)

    if log_space and display_ms_ticks:
        xticks_ms = nice_ms_ticks(x_raw)
        yticks_ms = nice_ms_ticks(y_raw)
        ax.set_xticks(np.log(xticks_ms))
        ax.set_xticklabels([f"{int(t)}" for t in xticks_ms])
        ax.set_yticks(np.log(yticks_ms))
        ax.set_yticklabels([f"{int(t)}" for t in yticks_ms])
    ax.set_zticks([0, zmax])
    ax.set_zticklabels(["0", f"{zmax:.2g}"])

    if schematic_axes:
        style_schematic_3d_axes(
            ax,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmax=zmax,
            show_box_grid=show_box_grid,
            draw_axis_lines=False,
        )
    else:
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 0.65)
            axis._axinfo["grid"]["linewidth"] = 0.6

    if human_x is not None and human_y is not None:
        human_z = float(kde(np.array([[human_x], [human_y]]))[0])
        ax.scatter(
            [human_x],
            [human_y],
            [human_z],
            marker="*",
            s=130,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            label="Human RT",
        )
        ax.legend(loc="upper right", frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate RT1/RT2 samples with Julia, then draw a 3D Gaussian KDE.")
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
    parser.add_argument("--log-axis-labels", action="store_true", help="In log-space plots, label axes as log(RT) instead of ms tick labels.")
    parser.add_argument("--boxed-axes", action="store_true", help="Use Matplotlib's boxed 3D axes instead of schematic open axes.")
    parser.add_argument("--box-grid", action="store_true", help="Show background box grid lines in schematic axis mode.")
    parser.add_argument("--grid-n", type=int, default=120, help="Number of grid points per axis.")
    parser.add_argument("--bw", default="silverman", help="KDE bandwidth: scott, silverman, or a float.")
    parser.add_argument("--elev", type=float, default=22, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58, help="3D camera azimuth.")
    args = parser.parse_args()

    if args.samples_csv is None:
        with tempfile.TemporaryDirectory(prefix="simulated_kde_") as tmpdir:
            samples_csv = Path(tmpdir) / "samples.csv"
            run_julia_simulation(args, samples_csv)
            df = finite_rt_frame(pd.read_csv(samples_csv), args.x_col, args.y_col)
            _plot_from_frame(df, args)
    else:
        run_julia_simulation(args, args.samples_csv)
        df = finite_rt_frame(pd.read_csv(args.samples_csv), args.x_col, args.y_col)
        _plot_from_frame(df, args)


def _plot_from_frame(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if len(df) < 3:
        raise ValueError(f"Need at least 3 finite positive simulated RT pairs; got {len(df)}.")

    human_rt1 = float(df["human_rt1"].iloc[0]) if "human_rt1" in df.columns else None
    human_rt2 = float(df["human_rt2"].iloc[0]) if "human_rt2" in df.columns else None

    plot_3d_kde(
        df[args.x_col].to_numpy(dtype=float),
        df[args.y_col].to_numpy(dtype=float),
        args.output,
        log_space=not args.raw_space,
        grid_n=args.grid_n,
        bw_method=parse_bw(args.bw),
        human_rt1=human_rt1,
        human_rt2=human_rt2,
        elev=args.elev,
        azim=args.azim,
        display_ms_ticks=not args.log_axis_labels,
        schematic_axes=not args.boxed_axes,
        show_box_grid=args.box_grid,
    )
    print(f"Plotted {len(df)} matching simulated RT pairs")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
