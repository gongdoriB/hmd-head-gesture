#!/usr/bin/env python3
"""
Train on all train trials and evaluate only on held-out test trials.

This is the experiment-design split:
- Train: every subject's trials 1 and 2
- Test: every subject's trial 3

Each row in feature_table.csv is one gesture repetition window.
"""

from __future__ import annotations

import json
import os
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
    """Create the RandomForest pipeline."""
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


def save_confusion_matrix_png(y_true: pd.Series, y_pred: list[str], output_path: Path) -> None:
    """Render and save the confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set_title("RF Train All Train Trials -> Test All Test Trials")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feature_table()

    train_df = df[(df["split"] == "train") | (df["trial"].isin([1, 2]))].copy()
    test_df = df[(df["split"] == "test") | (df["trial"] == 3)].copy()

    model = build_rf()
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    y_pred = model.predict(test_df[FEATURE_COLUMNS]).tolist()
    y_true = test_df["label"]

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    output_png = RESULTS_DIR / "train_all_train_trials_test_all_test_trials_rf_confusion_matrix.png"
    save_confusion_matrix_png(y_true, y_pred, output_png)

    metrics = pd.DataFrame(
        [
            {
                "analysis": "train_all_train_trials_test_all_test_trials",
                "model": "RandomForest",
                "train_subjects": ",".join(sorted(train_df["subject_id"].astype(str).unique())),
                "test_subjects": ",".join(sorted(test_df["subject_id"].astype(str).unique())),
                "train_trials": "1,2",
                "test_trials": "3",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
                "classification_report_json": json.dumps(report, ensure_ascii=False),
                "confusion_matrix_json": json.dumps(matrix.tolist(), ensure_ascii=False),
                "confusion_matrix_png": str(output_png),
            }
        ]
    )
    metrics.to_csv(RESULTS_DIR / "train_all_test_all_metrics.csv", index=False)

    print("Train all train trials -> test all test trials:")
    print(metrics[["analysis", "train_rows", "test_rows", "accuracy", "macro_f1"]].to_string(index=False))
    print("\nSaved:")
    print(f"  {RESULTS_DIR / 'train_all_test_all_metrics.csv'}")
    print(f"  {output_png}")


if __name__ == "__main__":
    main()
