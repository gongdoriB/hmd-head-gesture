#!/usr/bin/env python3
"""
Compare an interpretable rule-based baseline with a RandomForest classifier.

Input rows are gesture repetition windows from results/feature_table.csv.
This script does not perform row-level classification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
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

AXES = ["head_ex", "head_ey", "head_ez"]
RANGE_COLUMNS = [f"{axis}_range" for axis in AXES]
SIGNAL_PREFIXES = ["head_ex", "head_ey", "head_ez", "head_avx", "head_avy", "head_avz"]
FEATURE_SUFFIXES = ["mean", "std", "min", "max", "range", "rms", "first", "last", "delta"]
FEATURE_COLUMNS = [f"{prefix}_{suffix}" for prefix in SIGNAL_PREFIXES for suffix in FEATURE_SUFFIXES]

LABEL_TO_FAMILY = {
    "Rotating_Left": "Rotating",
    "Rotating_Right": "Rotating",
    "Tilting_Up": "Tilting",
    "Tilting_Down": "Tilting",
    "Leaning_Left": "Leaning",
    "Leaning_Right": "Leaning",
}
FAMILY_TO_LABELS = {
    "Rotating": ["Rotating_Left", "Rotating_Right"],
    "Tilting": ["Tilting_Up", "Tilting_Down"],
    "Leaning": ["Leaning_Left", "Leaning_Right"],
}


def load_feature_table() -> pd.DataFrame:
    """Load and validate the window-level feature table."""
    feature_table_path = RESULTS_DIR / "feature_table.csv"
    if not feature_table_path.exists():
        raise FileNotFoundError(f"Missing input file: {feature_table_path}")

    df = pd.read_csv(feature_table_path)
    required = {"subject_id", "trial", "split", "label", *RANGE_COLUMNS, *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"feature_table.csv is missing required columns: {missing}")

    df = df[df["label"].isin(LABELS)].copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    return df


def save_label_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Save mean Euler angle range by label."""
    summary = (
        df.groupby("label", sort=False)[RANGE_COLUMNS]
        .mean()
        .reindex(LABELS)
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "label_axis_summary.csv", index=False)
    return summary


def split_train_test(df: pd.DataFrame, subject_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by experiment design: trials 1 and 2 train, trial 3 test."""
    scoped = df.copy()
    if subject_id is not None:
        scoped = scoped[scoped["subject_id"] == subject_id].copy()

    train = scoped[(scoped["split"] == "train") | (scoped["trial"].isin([1, 2]))].copy()
    test = scoped[(scoped["split"] == "test") | (scoped["trial"] == 3)].copy()
    return train, test


def learn_axis_family_mapping(train_df: pd.DataFrame) -> dict[str, str]:
    """
    Assign each Euler axis to the gesture family with the largest train-set range.

    This keeps the rule interpretable while avoiding hard-coded assumptions about
    which exported Euler axis corresponds to yaw, pitch, or roll.
    """
    working = train_df.copy()
    working["family"] = working["label"].map(LABEL_TO_FAMILY)
    family_axis_ranges = working.groupby("family")[RANGE_COLUMNS].mean()

    remaining_families = set(FAMILY_TO_LABELS)
    axis_to_family: dict[str, str] = {}

    # Greedy one-to-one assignment from strongest observed family-axis effects.
    candidates: list[tuple[float, str, str]] = []
    for family in family_axis_ranges.index:
        for range_column in RANGE_COLUMNS:
            candidates.append((float(family_axis_ranges.loc[family, range_column]), range_column, family))

    for _, range_column, family in sorted(candidates, reverse=True):
        axis = range_column.removesuffix("_range")
        if axis not in axis_to_family and family in remaining_families:
            axis_to_family[axis] = family
            remaining_families.remove(family)

    for axis in AXES:
        if axis not in axis_to_family:
            axis_to_family[axis] = sorted(remaining_families).pop(0)

    return axis_to_family


def learn_direction_mapping(train_df: pd.DataFrame, axis_to_family: dict[str, str]) -> dict[str, dict[str, str]]:
    """
    For each gesture family, learn which sign of axis delta maps to each label.

    Rule explanation:
    1. A window's largest Euler-angle range chooses the dominant motion axis.
    2. The train-set axis-to-family map converts that axis to Rotating/Tilting/Leaning.
    3. The delta on that axis, equivalent to last - first, chooses the direction.
       This uses first/last through the precomputed delta feature.
    """
    direction_map: dict[str, dict[str, str]] = {}

    for axis, family in axis_to_family.items():
        labels = FAMILY_TO_LABELS[family]
        delta_column = f"{axis}_delta"
        mean_deltas = train_df[train_df["label"].isin(labels)].groupby("label")[delta_column].mean()
        ordered = mean_deltas.reindex(labels).sort_values()

        negative_label = ordered.index[0]
        positive_label = ordered.index[-1]
        direction_map[family] = {
            "negative": negative_label,
            "positive": positive_label,
        }

    return direction_map


def predict_rule_based(test_df: pd.DataFrame, axis_to_family: dict[str, str], direction_map: dict[str, dict[str, str]]) -> list[str]:
    """Predict labels with the interpretable dominant-axis and delta-sign rule."""
    predictions: list[str] = []

    for _, row in test_df.iterrows():
        largest_range_column = row[RANGE_COLUMNS].astype(float).idxmax()
        axis = largest_range_column.removesuffix("_range")
        family = axis_to_family[axis]
        delta = float(row[f"{axis}_delta"])

        # If delta is exactly zero or missing, use last-vs-first explicitly as a
        # fallback. This still follows the same directionality rule.
        if not np.isfinite(delta) or delta == 0:
            first = float(row[f"{axis}_first"])
            last = float(row[f"{axis}_last"])
            delta = last - first

        direction = "positive" if delta >= 0 else "negative"
        predictions.append(direction_map[family][direction])

    return predictions


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


def save_confusion_matrix_png(y_true: pd.Series, y_pred: list[str] | np.ndarray, title: str, output_path: Path) -> None:
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


def metrics_row(
    analysis_name: str,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_pred: list[str] | np.ndarray,
) -> dict[str, object]:
    """Create one row for metrics_summary.csv."""
    y_true = test_df["label"]
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "analysis": analysis_name,
        "model": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "classification_report_json": json.dumps(report, ensure_ascii=False),
        "confusion_matrix_json": json.dumps(matrix.tolist(), ensure_ascii=False),
    }


def evaluate_random_forest(
    analysis_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    confusion_matrix_path: Path,
) -> tuple[dict[str, object], np.ndarray]:
    """Train and evaluate RandomForest for one split."""
    model = build_rf()
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    y_pred = model.predict(test_df[FEATURE_COLUMNS])

    save_confusion_matrix_png(test_df["label"], y_pred, f"{analysis_name} RandomForest", confusion_matrix_path)
    return metrics_row(analysis_name, "RandomForest", train_df, test_df, y_pred), y_pred


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feature_table()
    save_label_axis_summary(df)

    metrics: list[dict[str, object]] = []

    pooled_train, pooled_test = split_train_test(df)
    axis_to_family = learn_axis_family_mapping(pooled_train)
    direction_map = learn_direction_mapping(pooled_train, axis_to_family)
    rule_pred = predict_rule_based(pooled_test, axis_to_family, direction_map)

    save_confusion_matrix_png(
        pooled_test["label"],
        rule_pred,
        "Pooled Rule-Based Baseline",
        RESULTS_DIR / "rule_based_confusion_matrix.png",
    )
    metrics.append(metrics_row("pooled_trial_split", "RuleBased", pooled_train, pooled_test, rule_pred))

    rf_metrics, _ = evaluate_random_forest(
        "Pooled Trial Split",
        pooled_train,
        pooled_test,
        RESULTS_DIR / "rf_confusion_matrix.png",
    )
    metrics.append(rf_metrics | {"analysis": "pooled_trial_split"})

    for subject_id in sorted(df["subject_id"].dropna().unique()):
        train_df, test_df = split_train_test(df, subject_id=subject_id)
        subject_metrics, _ = evaluate_random_forest(
            f"{subject_id} Trial Split",
            train_df,
            test_df,
            RESULTS_DIR / f"{subject_id}_rf_confusion_matrix.png",
        )
        metrics.append(subject_metrics | {"analysis": f"{subject_id}_trial_split"})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print("Axis-to-family mapping used by rule baseline:")
    print(json.dumps(axis_to_family, indent=2, ensure_ascii=False))
    print("\nDirection mapping used by rule baseline:")
    print(json.dumps(direction_map, indent=2, ensure_ascii=False))
    print("\nMetrics:")
    print(metrics_df[["analysis", "model", "train_rows", "test_rows", "accuracy", "macro_f1"]].to_string(index=False))
    print("\nSaved:")
    outputs = [
        "label_axis_summary.csv",
        "metrics_summary.csv",
        "rule_based_confusion_matrix.png",
        "rf_confusion_matrix.png",
    ]
    outputs.extend(f"{subject_id}_rf_confusion_matrix.png" for subject_id in sorted(df["subject_id"].dropna().unique()))
    for output in outputs:
        print(f"  {RESULTS_DIR / output}")


if __name__ == "__main__":
    main()
