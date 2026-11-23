#!/usr/bin/env python3
"""
Run K-fold style evaluations for the HMD gesture feature table.

Important distinction:
- StratifiedKFold mixes subjects/trials across folds and can be optimistic.
- GroupKFold by subject is stricter because each fold tests on a held-out subject.
- GroupKFold by subject-trial holds out entire subject-trial blocks.
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
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
    required = {"subject_id", "trial", "label", *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"feature_table.csv is missing required columns: {missing}")

    df = df[df["label"].isin(LABELS)].copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    df["subject_trial"] = df["subject_id"].astype(str) + "_trial_" + df["trial"].astype(int).astype(str)
    return df


def build_rf() -> Pipeline:
    """Create the RandomForest pipeline used in the other analyses."""
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


def save_confusion_matrix_png(y_true: list[str], y_pred: list[str], title: str, output_path: Path) -> None:
    """Render and save an aggregate confusion matrix."""
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


def evaluate_cv(df: pd.DataFrame, cv_name: str, splits: list[tuple[list[int], list[int]]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate one cross-validation design."""
    fold_rows: list[dict[str, object]] = []
    all_true: list[str] = []
    all_pred: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        model = build_rf()
        model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
        y_pred = model.predict(test_df[FEATURE_COLUMNS]).tolist()
        y_true = test_df["label"].tolist()

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        fold_rows.append(
            {
                "cv_name": cv_name,
                "fold": fold_idx,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_subjects": ",".join(sorted(train_df["subject_id"].astype(str).unique())),
                "test_subjects": ",".join(sorted(test_df["subject_id"].astype(str).unique())),
                "train_trials": ",".join(map(str, sorted(train_df["trial"].dropna().astype(int).unique()))),
                "test_trials": ",".join(map(str, sorted(test_df["trial"].dropna().astype(int).unique()))),
                "test_groups": ",".join(sorted(test_df["subject_trial"].astype(str).unique())),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            }
        )

    report = classification_report(all_true, all_pred, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(all_true, all_pred, labels=LABELS)
    output_path = RESULTS_DIR / f"{cv_name}_confusion_matrix.png"
    save_confusion_matrix_png(all_true, all_pred, f"{cv_name} Aggregate Confusion Matrix", output_path)

    summary_row = {
        "cv_name": cv_name,
        "folds": len(splits),
        "total_test_rows": len(all_true),
        "accuracy_mean": pd.DataFrame(fold_rows)["accuracy"].mean(),
        "accuracy_std": pd.DataFrame(fold_rows)["accuracy"].std(),
        "macro_f1_mean": pd.DataFrame(fold_rows)["macro_f1"].mean(),
        "macro_f1_std": pd.DataFrame(fold_rows)["macro_f1"].std(),
        "aggregate_accuracy": accuracy_score(all_true, all_pred),
        "aggregate_macro_f1": f1_score(all_true, all_pred, labels=LABELS, average="macro", zero_division=0),
        "classification_report_json": json.dumps(report, ensure_ascii=False),
        "confusion_matrix_json": json.dumps(matrix.tolist(), ensure_ascii=False),
        "confusion_matrix_png": str(output_path),
    }
    return fold_rows, summary_row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feature_table().reset_index(drop=True)
    x = df[FEATURE_COLUMNS]
    y = df["label"]

    cv_designs: list[tuple[str, list[tuple[list[int], list[int]]]]] = []

    stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_designs.append(
        (
            "stratified_5fold_window",
            [(train.tolist(), test.tolist()) for train, test in stratified.split(x, y)],
        )
    )

    subject_groups = df["subject_id"].astype(str)
    subject_group_kfold = GroupKFold(n_splits=subject_groups.nunique())
    cv_designs.append(
        (
            "groupkfold_by_subject",
            [(train.tolist(), test.tolist()) for train, test in subject_group_kfold.split(x, y, groups=subject_groups)],
        )
    )

    subject_trial_groups = df["subject_trial"].astype(str)
    subject_trial_group_kfold = GroupKFold(n_splits=min(5, subject_trial_groups.nunique()))
    cv_designs.append(
        (
            "groupkfold_by_subject_trial",
            [
                (train.tolist(), test.tolist())
                for train, test in subject_trial_group_kfold.split(x, y, groups=subject_trial_groups)
            ],
        )
    )

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cv_name, splits in cv_designs:
        folds, summary = evaluate_cv(df, cv_name, splits)
        fold_rows.extend(folds)
        summary_rows.append(summary)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    fold_df.to_csv(RESULTS_DIR / "kfold_fold_metrics.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "kfold_summary_metrics.csv", index=False)

    print("K-fold summary:")
    print(
        summary_df[
            [
                "cv_name",
                "folds",
                "total_test_rows",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "aggregate_accuracy",
                "aggregate_macro_f1",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:")
    for output in [
        "kfold_fold_metrics.csv",
        "kfold_summary_metrics.csv",
        "stratified_5fold_window_confusion_matrix.png",
        "groupkfold_by_subject_confusion_matrix.png",
        "groupkfold_by_subject_trial_confusion_matrix.png",
    ]:
        print(f"  {RESULTS_DIR / output}")


if __name__ == "__main__":
    main()
