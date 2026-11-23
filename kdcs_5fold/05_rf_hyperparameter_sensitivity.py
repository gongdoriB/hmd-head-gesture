#!/usr/bin/env python3
"""
Check whether RandomForest results are sensitive to n_estimators and random seed.

This script gives numerical support for using n_estimators=300 and
random_state=42 in the baseline scripts. The seed value itself is not a
performance-tuned value; it is used for reproducibility.
"""

from __future__ import annotations

import os
import time
from itertools import permutations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MPL_CACHE_DIR = RESULTS_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


LABELS = [
    "Rotating_Left",
    "Rotating_Right",
    "Tilting_Up",
    "Tilting_Down",
    "Leaning_Left",
    "Leaning_Right",
]
SIGNAL_PREFIXES = ["head_ex", "head_ey", "head_ez", "head_avx", "head_avy", "head_avz"]
FEATURE_SUFFIXES = ["mean", "std", "min", "max", "range", "rms", "first", "last", "delta"]
FEATURE_COLUMNS = [f"{prefix}_{suffix}" for prefix in SIGNAL_PREFIXES for suffix in FEATURE_SUFFIXES]
N_ESTIMATORS_GRID = [10, 30, 50, 100, 200, 300, 500, 1000]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]


def load_feature_table() -> pd.DataFrame:
    """Load the window-level feature table."""
    path = RESULTS_DIR / "feature_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    required = {"subject_id", "trial", "label", *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"feature_table.csv is missing required columns: {missing}")

    df = df[df["label"].isin(LABELS)].copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    return df


def build_rf(n_estimators: int, random_state: int) -> Pipeline:
    """Create a RandomForest pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                ),
            ),
        ]
    )


def pooled_trial_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on trials 1 and 2, test on trial 3 for p01+p02."""
    train = df[df["trial"].isin([1, 2])].copy()
    test = df[df["trial"] == 3].copy()
    return train, test


def conservative_split(df: pd.DataFrame, train_subject: str, test_subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on one subject's trials 1/2, test on another subject's trial 3."""
    train = df[(df["subject_id"] == train_subject) & (df["trial"].isin([1, 2]))].copy()
    test = df[(df["subject_id"] == test_subject) & (df["trial"] == 3)].copy()
    return train, test


def evaluate_split(
    analysis: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_estimators: int,
    random_state: int,
) -> dict[str, object]:
    """Train and evaluate one RandomForest setting."""
    model = build_rf(n_estimators=n_estimators, random_state=random_state)
    started = time.perf_counter()
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    fit_seconds = time.perf_counter() - started

    y_pred = model.predict(test_df[FEATURE_COLUMNS])
    y_true = test_df["label"]

    return {
        "analysis": analysis,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "fit_seconds": fit_seconds,
    }


def save_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across random seeds for each split and n_estimators value."""
    summary = (
        results.groupby(["analysis", "n_estimators"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", "std"),
            runs=("macro_f1", "size"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "rf_n_estimators_sensitivity_summary.csv", index=False)
    return summary


def save_plot(summary: pd.DataFrame) -> None:
    """Plot macro F1 versus n_estimators."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for analysis, part in summary.groupby("analysis"):
        ax.errorbar(
            part["n_estimators"],
            part["macro_f1_mean"],
            yerr=part["macro_f1_std"].fillna(0),
            marker="o",
            capsize=3,
            label=analysis,
        )

    ax.set_xscale("log")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("Macro F1")
    ax.set_title("RandomForest Sensitivity to Number of Trees")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_n_estimators_sensitivity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feature_table()

    split_defs = []
    pooled_train, pooled_test = pooled_trial_split(df)
    split_defs.append(("pooled_trial_split", pooled_train, pooled_test))

    subjects = sorted(df["subject_id"].dropna().unique())
    for train_subject, test_subject in permutations(subjects, 2):
        train_df, test_df = conservative_split(df, train_subject, test_subject)
        split_defs.append((f"{train_subject}_train12_to_{test_subject}_trial3", train_df, test_df))

    rows: list[dict[str, object]] = []
    for analysis, train_df, test_df in split_defs:
        for n_estimators in N_ESTIMATORS_GRID:
            for seed in SEEDS:
                rows.append(evaluate_split(analysis, train_df, test_df, n_estimators, seed))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "rf_hyperparameter_sensitivity.csv", index=False)
    summary = save_summary(results)
    save_plot(summary)

    print("RandomForest hyperparameter sensitivity summary:")
    print(
        summary[
            [
                "analysis",
                "n_estimators",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "fit_seconds_mean",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:")
    for output in [
        "rf_hyperparameter_sensitivity.csv",
        "rf_n_estimators_sensitivity_summary.csv",
        "rf_n_estimators_sensitivity.png",
    ]:
        print(f"  {RESULTS_DIR / output}")


if __name__ == "__main__":
    main()
