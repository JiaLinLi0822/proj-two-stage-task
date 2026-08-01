#!/usr/bin/env python3
"""Human behavioral analysis following Solway & Botvinick (2015).

This script mirrors the paper's descriptive aggregation and inferential
approach:

- plot/report participant-level means at each difficulty/value-difference level;
- test level effects with one-way repeated-measures ANOVA;
- for RT ANOVAs, log-transform trial-level RTs before computing participant
  means, matching the supplementary-material description.
- for first-stage RT curves, include trials with a correct first-stage choice.
- for second-stage RT curves, include trials with a correct second-stage choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Tree2" / "data" / "Tree2.json"
OUT_DIR = ROOT / "Tree2" / "results" / "human_behavior_stats_solway"


def subtree_vals(value2: list[float], choice1: int) -> list[float]:
    return value2[:2] if choice1 == 1 else value2[2:4]


def subtree_relation_code(path: list[float]) -> int:
    idx_desc = np.argsort(path)[::-1]
    best, second, third, worst = idx_desc
    subtree = lambda i: 0 if i < 2 else 1
    if subtree(best) == subtree(second):
        return 1
    if subtree(best) == subtree(third):
        return 2
    if subtree(best) == subtree(worst):
        return 3
    raise ValueError(f"Unexpected path configuration: {path}")


def load_data() -> pd.DataFrame:
    with DATA_PATH.open("r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(records)
    df["path"] = df["rewards"]
    df["best_path_idx"] = df["path"].apply(lambda values: int(np.argmax(values)))
    df["correct1"] = df.apply(
        lambda r: (r["best_path_idx"] < 2 and r["choice1"] == 1)
        or (r["best_path_idx"] >= 2 and r["choice1"] == 2),
        axis=1,
    ).astype(int)
    df["correct2"] = df.apply(
        lambda r: r["value2"][int(r["choice2"]) - 1]
        == max(subtree_vals(r["value2"], int(r["choice1"]))),
        axis=1,
    ).astype(int)
    df["correct_all"] = (df["correct1"] & df["correct2"]).astype(int)
    df["tree_configuration"] = df["path"].apply(subtree_relation_code)
    df["difficulty"] = df["difficulty"].astype(float)
    df["diff2"] = df["diff2"].astype(float)
    df["log_rt1"] = np.log(df["rt1"])
    df["log_rt2"] = np.log(df["rt2"])
    return df


def subject_level_table(df: pd.DataFrame, level: str, dv: str, filt: str | None = None) -> pd.DataFrame:
    data = df if filt is None else df.query(filt)
    per_subject = data.groupby(["wid", level], as_index=False)[dv].mean()
    raw_n = data.groupby(level).size().rename("trials").reset_index()
    summary = (
        per_subject.groupby(level)[dv]
        .agg(mean="mean", std="std", n_subjects="count")
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n_subjects"])
    return summary.merge(raw_n, on=level, how="left")


def repeated_measures_anova(
    df: pd.DataFrame,
    label: str,
    level: str,
    dv: str,
    filt: str | None = None,
) -> dict[str, float | str]:
    data = df if filt is None else df.query(filt)
    per_subject = data.groupby(["wid", level], as_index=False)[dv].mean()
    wide = per_subject.pivot(index="wid", columns=level, values=dv).dropna()
    complete = wide.reset_index().melt(id_vars="wid", var_name=level, value_name=dv)
    model = AnovaRM(complete, depvar=dv, subject="wid", within=[level]).fit()
    row = model.anova_table.iloc[0]
    f_value = float(row["F Value"])
    df_num = float(row["Num DF"])
    df_den = float(row["Den DF"])
    p_value = float(row["Pr > F"])
    eta_p2 = (f_value * df_num) / (f_value * df_num + df_den)
    return {
        "analysis": label,
        "factor": level,
        "dv": dv,
        "n_complete_subjects": int(wide.shape[0]),
        "n_levels": int(wide.shape[1]),
        "df_num": df_num,
        "df_den": df_den,
        "F": f_value,
        "p": p_value,
        "partial_eta_squared": eta_p2,
    }


def repeated_measures_anova_level_mean_imputed(
    df: pd.DataFrame,
    label: str,
    level: str,
    dv: str,
    filt: str | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    """Repeated-measures ANOVA after level-wise population-mean imputation."""
    data = df if filt is None else df.query(filt)
    per_subject = data.groupby(["wid", level], as_index=False)[dv].mean()
    wide = per_subject.pivot(index="wid", columns=level, values=dv).sort_index(axis=1)
    level_means = wide.mean(axis=0, skipna=True)
    missing = wide.isna()
    imputed = wide.fillna(level_means)
    complete = imputed.reset_index().melt(id_vars="wid", var_name=level, value_name=dv)
    model = AnovaRM(complete, depvar=dv, subject="wid", within=[level]).fit()
    row = model.anova_table.iloc[0]
    f_value = float(row["F Value"])
    df_num = float(row["Num DF"])
    df_den = float(row["Den DF"])
    p_value = float(row["Pr > F"])
    eta_p2 = (f_value * df_num) / (f_value * df_num + df_den)
    anova_row = {
        "analysis": label,
        "factor": level,
        "dv": dv,
        "n_subjects": int(imputed.shape[0]),
        "n_levels": int(imputed.shape[1]),
        "n_imputed_cells": int(missing.sum().sum()),
        "df_num": df_num,
        "df_den": df_den,
        "F": f_value,
        "p": p_value,
        "partial_eta_squared": eta_p2,
    }
    imputation_summary = pd.DataFrame(
        {
            level: level_means.index,
            "population_mean_used_for_imputation": level_means.values,
            "n_observed_subjects": (~missing).sum(axis=0).values,
            "n_imputed_subjects": missing.sum(axis=0).values,
        }
    )
    return anova_row, imputed, imputation_summary


def adjacent_trend_table(df: pd.DataFrame, level: str, dv: str, filt: str | None = None) -> pd.DataFrame:
    """Paired adjacent-level contrasts, included as a readable follow-up."""
    data = df if filt is None else df.query(filt)
    wide = data.groupby(["wid", level])[dv].mean().unstack(level)
    rows = []
    levels = list(wide.columns)
    for lo, hi in zip(levels[:-1], levels[1:]):
        paired = wide[[lo, hi]].dropna()
        diff = paired[hi] - paired[lo]
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        rows.append(
            {
                "level_low": lo,
                "level_high": hi,
                "n_subjects": len(diff),
                "mean_high_minus_low": diff.mean(),
                "t": t_stat,
                "p": p_val,
            }
        )
    return pd.DataFrame(rows)


def pairwise_level_table(df: pd.DataFrame, level: str, dv: str, filt: str | None = None) -> pd.DataFrame:
    """All paired level contrasts with Bonferroni-corrected p values."""
    data = df if filt is None else df.query(filt)
    wide = data.groupby(["wid", level])[dv].mean().unstack(level)
    rows = []
    levels = list(wide.columns)
    for i, lo in enumerate(levels[:-1]):
        for hi in levels[i + 1 :]:
            paired = wide[[lo, hi]].dropna()
            diff = paired[hi] - paired[lo]
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            rows.append(
                {
                    "level_low": lo,
                    "level_high": hi,
                    "n_subjects": len(diff),
                    "mean_high_minus_low": diff.mean(),
                    "t": t_stat,
                    "p_uncorrected": p_val,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bonferroni"] = np.minimum(out["p_uncorrected"] * len(out), 1.0)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    analyses = [
        ("stage1_accuracy_by_difficulty", "difficulty", "correct1", None),
        ("stage1_rt_by_difficulty_correct1", "difficulty", "rt1", "correct1 == 1"),
        (
            "stage1_log_rt_by_difficulty_correct1",
            "difficulty",
            "log_rt1",
            "correct1 == 1",
        ),
        ("stage2_accuracy_by_value_difference", "diff2", "correct2", None),
        ("stage2_rt_by_value_difference_correct2", "diff2", "rt2", "correct2 == 1"),
        (
            "stage2_log_rt_by_value_difference_correct2",
            "diff2",
            "log_rt2",
            "correct2 == 1",
        ),
        (
            "stage1_accuracy_by_tree_configuration",
            "tree_configuration",
            "correct1",
            None,
        ),
        (
            "stage1_log_rt_by_tree_configuration_correct1",
            "tree_configuration",
            "log_rt1",
            "correct1 == 1",
        ),
    ]

    complete_case_rows = []
    for name, level, dv, filt in analyses:
        subject_level_table(df, level, dv, filt).to_csv(OUT_DIR / f"{name}.csv", index=False)
        if dv.startswith("log_rt") or "accuracy" in name:
            complete_case_rows.append(repeated_measures_anova(df, name, level, dv, filt))
        adjacent_trend_table(df, level, dv, filt).to_csv(
            OUT_DIR / f"{name}_adjacent_paired_ttests.csv",
            index=False,
        )
        if "tree_configuration" in name:
            pairwise_level_table(df, level, dv, filt).to_csv(
                OUT_DIR / f"{name}_pairwise_paired_ttests.csv",
                index=False,
            )

    pd.DataFrame(complete_case_rows).to_csv(
        OUT_DIR / "repeated_measures_anova_complete_case.csv",
        index=False,
    )

    imputed_rows = []
    stage1_acc_row, stage1_acc_imputed, stage1_acc_imputation = repeated_measures_anova_level_mean_imputed(
        df,
        "stage1_accuracy_by_difficulty_level_mean_imputed",
        "difficulty",
        "correct1",
    )
    imputed_rows.append(stage1_acc_row)
    stage1_acc_imputed.to_csv(
        OUT_DIR / "stage1_accuracy_by_difficulty_level_mean_imputed_matrix.csv"
    )
    stage1_acc_imputation.to_csv(
        OUT_DIR / "stage1_accuracy_by_difficulty_level_mean_imputation_summary.csv",
        index=False,
    )

    stage1_row, stage1_imputed, stage1_imputation = repeated_measures_anova_level_mean_imputed(
        df,
        "stage1_log_rt_by_difficulty_correct1_level_mean_imputed",
        "difficulty",
        "log_rt1",
        "correct1 == 1",
    )
    imputed_rows.append(stage1_row)
    stage1_imputed.to_csv(
        OUT_DIR / "stage1_log_rt_by_difficulty_correct1_level_mean_imputed_matrix.csv"
    )
    stage1_imputation.to_csv(
        OUT_DIR / "stage1_log_rt_by_difficulty_correct1_level_mean_imputation_summary.csv",
        index=False,
    )
    pd.DataFrame(imputed_rows).to_csv(
        OUT_DIR / "repeated_measures_anova_level_mean_imputed.csv",
        index=False,
    )

    final_rows = []
    for row in imputed_rows:
        final_row = dict(row)
        final_row["missing_data_method"] = "level_population_mean_imputation"
        final_rows.append(final_row)
    for row in complete_case_rows:
        analysis = str(row["analysis"])
        if analysis.startswith("stage2_") or "tree_configuration" in analysis:
            final_row = dict(row)
            final_row["n_subjects"] = final_row.pop("n_complete_subjects")
            final_row["n_imputed_cells"] = 0
            final_row["missing_data_method"] = "complete_case"
            final_rows.append(final_row)
    pd.DataFrame(final_rows).to_csv(OUT_DIR / "repeated_measures_anova.csv", index=False)
    print(f"Analyzed {len(df)} trials from {df['wid'].nunique()} subjects.")
    print(f"Wrote Solway-style results to {OUT_DIR}")


if __name__ == "__main__":
    main()
