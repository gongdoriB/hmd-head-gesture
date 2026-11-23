#!/usr/bin/env python3
"""
Run a conservative cross-subject split.

This evaluates whether a model trained on one participant's train trials
(trials 1 and 2) can classify another participant's held-out test trial
(trial 3). Each row is one gesture repetition window.
"""

from __future__ import annotations

import json
import os
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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


def load_feature_table() -> pd.DataFrame:
    """Load the window-level feature table."""
    path = RESULTS_DIR / "feature_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    required = {"subject_id", "trial", "split", "label", *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"feature_table.csv is missing required columns: {missing}")

    df = df[df["label"].isin(LABELS)].copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    return df


def build_rf() -> Pipeline:
    """Create the RandomForest pipeline used in the other RF analyses."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                ),
            ),
        ]
    )


def save_confusion_matrix_png(y_true: pd.Series, y_pred: list[str], title: str, output_path: Path) -> None:
    """Render and save a confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_conservative_split(df: pd.DataFrame, train_subject: str, test_subject: str) -> dict[str, object]:
    """
    Train on one subject's trials 1 and 2, test on the other subject's trial 3.

    This is stricter than training on all trials from the train subject because
    it preserves the original held-out-trial idea and changes the participant.
    """
    train_df = df[(df["subject_id"] == train_subject) & (df["trial"].isin([1, 2]))].copy()
    test_df = df[(df["subject_id"] == test_subject) & (df["trial"] == 3)].copy()

    model = build_rf()
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    y_pred = model.predict(test_df[FEATURE_COLUMNS])
    y_true = test_df["label"]

    output_path = RESULTS_DIR / f"{train_subject}_train12_to_{test_subject}_trial3_rf_confusion_matrix.png"
    save_confusion_matrix_png(
        y_true,
        y_pred,
        f"RF {train_subject} trials 1-2 -> {test_subject} trial 3",
        output_path,
    )

    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "analysis": f"{train_subject}_train12_to_{test_subject}_trial3",
        "model": "RandomForest",
        "train_subject": train_subject,
        "test_subject": test_subject,
        "train_trials": "1,2",
        "test_trials": "3",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "classification_report_json": json.dumps(report, ensure_ascii=False),
        "confusion_matrix_json": json.dumps(matrix.tolist(), ensure_ascii=False),
        "confusion_matrix_png": str(output_path),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feature_table()

    subjects = sorted(df["subject_id"].dropna().unique())
    rows = [
        evaluate_conservative_split(df, train_subject, test_subject)
        for train_subject, test_subject in permutations(subjects, 2)
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "conservative_subject_split_metrics.csv", index=False)

    print("Conservative cross-subject RandomForest results:")
    print(metrics[["analysis", "train_rows", "test_rows", "accuracy", "macro_f1"]].to_string(index=False))
    print("\nSaved:")
    print(f"  {RESULTS_DIR / 'conservative_subject_split_metrics.csv'}")
    for row in rows:
        print(f"  {row['confusion_matrix_png']}")


if __name__ == "__main__":
    main()
