#!/usr/bin/env python3
"""Statistical summary of Tree2 human behavior.

The derived variables mirror Tree2/plotting/plot.py. Descriptive summaries are
computed as subject-level means and then averaged across subjects; trend tests
use GEE models clustered by participant.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Tree2" / "data" / "Tree2.json"
OUT_DIR = ROOT / "Tree2" / "results" / "human_behavior_stats"


def subtree_vals(value2: list[float], choice1: int) -> list[float]:
    return value2[:2] if choice1 == 1 else value2[2:4]


def load_human_data() -> pd.DataFrame:
    with DATA_PATH.open("r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(records)
    df["path"] = df["rewards"]
    df["best_path_idx"] = df["path"].apply(lambda v: int(np.argmax(v)))
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
    df["difficulty"] = df["difficulty"].astype(float)
    df["diff1"] = df["diff1"].astype(float)
    df["diff2"] = df["diff2"].astype(float)
    return df


def subject_mean_summary(df: pd.DataFrame, level: str, y: str, filt: str | None = None) -> pd.DataFrame:
    data = df if filt is None else df.query(filt)
    per_subject = data.groupby(["wid", level])[y].mean().reset_index(name="subject_mean")
    out = per_subject.groupby(level)["subject_mean"].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    raw_n = data.groupby(level).size().rename("trials").reset_index()
    return out.merge(raw_n, on=level, how="left")


def fit_gee(formula: str, df: pd.DataFrame, family):
    return smf.gee(
        formula,
        groups="wid",
        data=df,
        cov_struct=Exchangeable(),
        family=family,
    ).fit()


def trend_row(label: str, model, predictor: str, kind: str) -> dict[str, float | str]:
    beta = model.params[predictor]
    se = model.bse[predictor]
    z = beta / se
    row: dict[str, float | str] = {
        "analysis": label,
        "predictor": predictor,
        "beta": beta,
        "se": se,
        "z": z,
        "p": model.pvalues[predictor],
    }
    if kind == "accuracy":
        row["odds_ratio_per_unit"] = np.exp(beta)
    else:
        row["percent_change_per_unit"] = (np.exp(beta) - 1) * 100
    return row


def categorical_wald(label: str, formula: str, df: pd.DataFrame, family) -> dict[str, float | str]:
    model = fit_gee(formula, df, family)
    terms = [i for i, name in enumerate(model.params.index) if name != "Intercept"]
    constraint = np.zeros((len(terms), len(model.params)))
    for row, idx in enumerate(terms):
        constraint[row, idx] = 1
    test = model.wald_test(constraint, scalar=True)
    return {
        "analysis": label,
        "df": len(terms),
        "chi2": float(test.statistic),
        "p": float(test.pvalue),
    }


def endpoint_contrast(
    df: pd.DataFrame,
    label: str,
    level: str,
    y: str,
    low: float,
    high: float,
    filt: str | None = None,
) -> dict[str, float | str]:
    data = df if filt is None else df.query(filt)
    wide = data.groupby(["wid", level])[y].mean().unstack(level)
    paired = wide[[low, high]].dropna()
    diff = paired[high] - paired[low]
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    return {
        "analysis": label,
        "n_subjects": len(diff),
        "mean_high_minus_low": diff.mean(),
        "t": t_stat,
        "p": p_val,
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_human_data()
    summaries = {
        "stage1_accuracy_by_difficulty": subject_mean_summary(df, "difficulty", "correct1"),
        "stage1_rt_correct_by_difficulty": subject_mean_summary(
            df, "difficulty", "rt1", "correct1 == 1"
        ),
        "stage2_accuracy_by_value_difference": subject_mean_summary(df, "diff2", "correct2"),
        "stage2_rt_correct_by_value_difference": subject_mean_summary(
            df, "diff2", "rt2", "correct2 == 1"
        ),
        "stage2_accuracy_by_difficulty": subject_mean_summary(df, "difficulty", "correct2"),
        "stage2_rt_correct_by_difficulty": subject_mean_summary(
            df, "difficulty", "rt2", "correct2 == 1"
        ),
    }
    for name, table in summaries.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)

    rt1 = df[df["correct1"] == 1].copy()
    rt1["log_rt1"] = np.log(rt1["rt1"])
    rt2 = df[df["correct2"] == 1].copy()
    rt2["log_rt2"] = np.log(rt2["rt2"])

    trends = pd.DataFrame(
        [
            trend_row(
                "First-stage accuracy ~ difficulty",
                fit_gee("correct1 ~ difficulty", df, Binomial()),
                "difficulty",
                "accuracy",
            ),
            trend_row(
                "Second-stage accuracy ~ value difference",
                fit_gee("correct2 ~ diff2", df, Binomial()),
                "diff2",
                "accuracy",
            ),
            trend_row(
                "Second-stage accuracy ~ difficulty",
                fit_gee("correct2 ~ difficulty", df, Binomial()),
                "difficulty",
                "accuracy",
            ),
            trend_row(
                "log first-stage RT ~ difficulty",
                fit_gee("log_rt1 ~ difficulty", rt1, Gaussian()),
                "difficulty",
                "rt",
            ),
            trend_row(
                "log second-stage RT ~ value difference",
                fit_gee("log_rt2 ~ diff2", rt2, Gaussian()),
                "diff2",
                "rt",
            ),
            trend_row(
                "log second-stage RT ~ difficulty",
                fit_gee("log_rt2 ~ difficulty", rt2, Gaussian()),
                "difficulty",
                "rt",
            ),
        ]
    )
    trends.to_csv(OUT_DIR / "gee_linear_trends.csv", index=False)

    categorical = pd.DataFrame(
        [
            categorical_wald(
                "First-stage accuracy categorical difficulty",
                "correct1 ~ C(difficulty)",
                df,
                Binomial(),
            ),
            categorical_wald(
                "Second-stage accuracy categorical value difference",
                "correct2 ~ C(diff2)",
                df,
                Binomial(),
            ),
            categorical_wald(
                "log first-stage RT categorical difficulty",
                "log_rt1 ~ C(difficulty)",
                rt1,
                Gaussian(),
            ),
            categorical_wald(
                "log second-stage RT categorical value difference",
                "log_rt2 ~ C(diff2)",
                rt2,
                Gaussian(),
            ),
        ]
    )
    categorical.to_csv(OUT_DIR / "gee_categorical_wald.csv", index=False)

    endpoints = pd.DataFrame(
        [
            endpoint_contrast(df, "First-stage accuracy: difficulty 11 minus 2", "difficulty", "correct1", 2.0, 11.0),
            endpoint_contrast(
                df,
                "First-stage RT: difficulty 11 minus 2",
                "difficulty",
                "rt1",
                2.0,
                11.0,
                "correct1 == 1",
            ),
            endpoint_contrast(df, "Second-stage accuracy: diff2 8 minus 1", "diff2", "correct2", 1.0, 8.0),
            endpoint_contrast(
                df,
                "Second-stage RT: diff2 8 minus 1",
                "diff2",
                "rt2",
                1.0,
                8.0,
                "correct2 == 1",
            ),
        ]
    )
    endpoints.to_csv(OUT_DIR / "endpoint_paired_contrasts.csv", index=False)

    print(f"Analyzed {len(df)} trials from {df['wid'].nunique()} subjects.")
    print(f"Difficulty equals diff1 for {(df['difficulty'] == df['diff1']).mean() * 100:.1f}% of trials.")
    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
