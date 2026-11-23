#!/usr/bin/env python3
"""
Build a gesture-window feature table for the HMD head gesture experiment.

This script intentionally treats one continuous same-label gesture repetition
as one classification sample. It does not perform row-level classification and
does not train a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_WINDOWS_PER_TRIAL = 60
EXPECTED_ROWS_PER_TRIAL = 9000
FEATURE_COLUMNS = (
    "head_ex",
    "head_ey",
    "head_ez",
    "head_avx",
    "head_avy",
    "head_avz",
)
REQUIRED_COLUMNS = (
    "subject_id",
    "trial",
    "label",
    "timestamp",
    "head_px",
    "head_py",
    "head_pz",
    "head_rx",
    "head_ry",
    "head_rz",
    "head_rw",
    *FEATURE_COLUMNS,
)


def project_root() -> Path:
    """Return the repository/workspace root based on this script location."""
    return Path(__file__).resolve().parents[1]


def find_data_files(root: Path) -> list[Path]:
    """
    Find subject CSV files.

    The current project keeps raw data under data/raw. Older layouts are still
    supported as fallbacks so previous exports can be re-run if needed.
    """
    candidates = [root / "data" / "raw", root / "CollectedData", root]
    search_base = next((candidate for candidate in candidates if candidate.exists()), root)

    filename_pattern = re.compile(r"^p\d+_trial_\d+_(train|test)\.csv$")
    files_by_name: dict[str, Path] = {}

    for directory in sorted(path for path in search_base.iterdir() if path.is_dir()):
        for csv_path in sorted(directory.glob("*.csv")):
            if filename_pattern.match(csv_path.name):
                files_by_name.setdefault(csv_path.name, csv_path)

    return sorted(files_by_name.values())


def parse_filename(path: Path) -> dict[str, object]:
    """Extract subject_id, trial, and split from filenames like p01_trial_1_train.csv."""
    match = re.match(r"^(?P<subject>p\d+)_trial_(?P<trial>\d+)_(?P<split>train|test)\.csv$", path.name)
    if not match:
        return {"subject_id": None, "trial": None, "split": None}

    return {
        "subject_id": match.group("subject"),
        "trial": int(match.group("trial")),
        "split": match.group("split"),
    }


def quality_summary(path: Path, df: pd.DataFrame, parsed: dict[str, object]) -> dict[str, object]:
    """Create one file-level data quality summary row."""
    label_counts = df["label"].value_counts(dropna=False).sort_index()
    timestamp_by_label = df.groupby("label", dropna=False)["timestamp"].agg(["min", "max"])

    row_count = len(df)
    missing_count = int(df.isna().sum().sum())
    row_delta_from_expected = row_count - EXPECTED_ROWS_PER_TRIAL

    return {
        "filename": str(path),
        "subject_id": parsed["subject_id"],
        "trial": parsed["trial"],
        "split": parsed["split"],
        "total_rows": row_count,
        "expected_rows": EXPECTED_ROWS_PER_TRIAL,
        "row_delta_from_expected": row_delta_from_expected,
        "label_counts_json": json.dumps(label_counts.to_dict(), ensure_ascii=False),
        "timestamp_min_by_label_json": json.dumps(timestamp_by_label["min"].to_dict(), ensure_ascii=False),
        "timestamp_max_by_label_json": json.dumps(timestamp_by_label["max"].to_dict(), ensure_ascii=False),
        "missing_value_count": missing_count,
    }


def segment_windows(df: pd.DataFrame) -> pd.Series:
    """
    Return a 1-based window id for every gesture repetition.

    A new window starts when the label changes. In the current export format,
    repeated gestures with the same label can appear back-to-back; those are
    separated by the per-repetition timestamp restarting near 0.
    """
    label_changed = df["label"].ne(df["label"].shift())
    timestamp_restarted = df["timestamp"].lt(df["timestamp"].shift())
    return (label_changed | timestamp_restarted).cumsum()


def rms(values: pd.Series) -> float:
    """Compute root mean square while ignoring missing values."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(np.sqrt(np.mean(np.square(numeric))))


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    """Extract summary features for every configured feature column."""
    features: dict[str, float] = {}

    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(window[column], errors="coerce")
        first = values.iloc[0]
        last = values.iloc[-1]

        features[f"{column}_mean"] = values.mean()
        features[f"{column}_std"] = values.std(ddof=1)
        features[f"{column}_min"] = values.min()
        features[f"{column}_max"] = values.max()
        features[f"{column}_range"] = values.max() - values.min()
        features[f"{column}_rms"] = rms(values)
        features[f"{column}_first"] = first
        features[f"{column}_last"] = last
        features[f"{column}_delta"] = last - first

    return features


def process_file(path: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Read one CSV, inspect quality, segment windows, and extract features."""
    parsed = parse_filename(path)
    df = pd.read_csv(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    if parsed["subject_id"] is None:
        print(f"WARNING: Could not parse subject/trial/split from filename: {path.name}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    quality = quality_summary(path, df, parsed)
    df = df.copy()
    df["window_id"] = segment_windows(df)

    num_windows = int(df["window_id"].nunique())
    if num_windows != EXPECTED_WINDOWS_PER_TRIAL:
        print(
            "WARNING: "
            f"{path.name} produced {num_windows} windows; "
            f"expected {EXPECTED_WINDOWS_PER_TRIAL}."
        )

    print(
        f"{path.name}: rows={len(df)}, windows={num_windows}, "
        f"missing_values={quality['missing_value_count']}"
    )
    print(f"  label_counts={quality['label_counts_json']}")
    print(f"  timestamp_min_by_label={quality['timestamp_min_by_label_json']}")
    print(f"  timestamp_max_by_label={quality['timestamp_max_by_label_json']}")

    repetition_counts: dict[str, int] = {}
    window_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for window_id, window in df.groupby("window_id", sort=True):
        label = str(window["label"].iloc[0])
        repetition_counts[label] = repetition_counts.get(label, 0) + 1

        timestamp_min = window["timestamp"].min()
        timestamp_max = window["timestamp"].max()
        duration = timestamp_max - timestamp_min

        metadata = {
            "subject_id": parsed["subject_id"],
            "trial": parsed["trial"],
            "split": parsed["split"],
            "label": label,
            "window_id": int(window_id),
            "repetition_index_within_label": repetition_counts[label],
            "num_frames": int(len(window)),
            "duration": duration,
        }

        window_rows.append(metadata)
        feature_rows.append({**metadata, **extract_window_features(window)})

    return quality, window_rows, feature_rows


def main() -> None:
    root = project_root()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    files = find_data_files(root)
    if not files:
        raise FileNotFoundError(
            "No CSV files found under CollectedData/pXX or root-level pXX folders."
        )

    quality_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    print(f"Found {len(files)} CSV files.")
    for path in files:
        quality, windows, features = process_file(path)
        quality_rows.append(quality)
        window_rows.extend(windows)
        feature_rows.extend(features)

    quality_df = pd.DataFrame(quality_rows)
    window_df = pd.DataFrame(window_rows)
    feature_df = pd.DataFrame(feature_rows)

    quality_df.to_csv(results_dir / "data_quality_summary.csv", index=False)
    window_df.to_csv(results_dir / "window_summary.csv", index=False)
    feature_df.to_csv(results_dir / "feature_table.csv", index=False)

    print("\nSaved:")
    print(f"  {results_dir / 'data_quality_summary.csv'}")
    print(f"  {results_dir / 'window_summary.csv'}")
    print(f"  {results_dir / 'feature_table.csv'}")
    print(f"\nFeature table rows: {len(feature_df)}")


if __name__ == "__main__":
    main()
