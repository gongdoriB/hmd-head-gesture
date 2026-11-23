#!/usr/bin/env python3
"""Unified HMD head-gesture analysis for Dataset A and Dataset B.

The script applies one preprocessing/model specification to both datasets and
evaluates Random Forest, CNN1D, and LSTM under:

* LOTO (within-user): train on two trials of one subject and test on the third.
* LOSO (cross-user): train on all other subjects and test on the unseen subject.

Every completed fold is cached as an NPZ file, so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # release edit: was parents[2]
DEFAULT_OUTPUT = ROOT / "04_results" / "final_loto_loso"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_OUTPUT / ".cache"))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError:
    tf = None
    keras = None
    layers = None


LABELS = ["rotL", "rotR", "tiltU", "tiltD", "leanL", "leanR"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

A_LABEL_MAP = {
    "Rotating_Left": "rotL",
    "Rotating_Right": "rotR",
    "Tilting_Up": "tiltU",
    "Tilting_Down": "tiltD",
    "Leaning_Left": "leanL",
    "Leaning_Right": "leanR",
}
B_LABEL_MAP = {
    "rotate_left": "rotL",
    "rotate_right": "rotR",
    "tilt_up": "tiltU",
    "tilt_down": "tiltD",
    "lean_left": "leanL",
    "lean_right": "leanR",
}

SEQUENCE_LENGTH = 128
ANGLE_CHANNELS = 3
RF_STATS = ["mean", "std", "min", "max", "range", "rms", "signed_peak", "delta"]


@dataclass
class DatasetBundle:
    name: str
    sequences: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    trials: np.ndarray
    window_ids: np.ndarray


def signed_angle_delta(values: np.ndarray, baseline: float) -> np.ndarray:
    """Circular angle difference in degrees, constrained to [-180, 180)."""
    return ((values - baseline + 180.0) % 360.0) - 180.0


def baseline_relative_signals(values: np.ndarray) -> np.ndarray:
    """Convert Euler channels to relative angles while preserving velocity."""
    values = np.asarray(values, dtype=np.float32).copy()
    baseline_frames = min(5, len(values))
    for channel in range(ANGLE_CHANNELS):
        angle = values[:, channel]
        # Circular mean is stable near the 0/360 boundary.
        radians = np.deg2rad(angle[:baseline_frames])
        baseline = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
        values[:, channel] = signed_angle_delta(angle, baseline)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def resample(values: np.ndarray, target_length: int | None = None) -> np.ndarray:
    """Linearly resample a T x C gesture window to target_length x C."""
    values = np.asarray(values, dtype=np.float32)
    if target_length is None:
        target_length = SEQUENCE_LENGTH
    if len(values) < 2:
        raise ValueError("Gesture window must contain at least two frames")
    old_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    output = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        output[:, channel] = np.interp(new_x, old_x, values[:, channel])
    return output


def dataset_a_files() -> list[Path]:
    bases = [
        ROOT / "02_data" / "dataset_A_jeju" / "raw" / "p01",
        ROOT / "02_data" / "dataset_A_jeju" / "raw" / "p02",
        ROOT / "02_data" / "dataset_A_jeju" / "raw" / "p03",
        ROOT / "02_data" / "dataset_A_jeju" / "raw" / "hmd_p04_data",
        ROOT / "02_data" / "dataset_A_jeju" / "raw" / "hmd_head_gesture_data",
    ]
    pattern = re.compile(r"^p(0[1-8])_trial_(\d+)_(?:train|test)\.csv$")
    files = []
    for base in bases:
        if not base.exists():
            continue
        files.extend(path for path in base.glob("*.csv") if pattern.match(path.name))
    by_key = {}
    for path in sorted(files):
        match = pattern.match(path.name)
        assert match
        by_key[(match.group(1), int(match.group(2)))] = path
    if len(by_key) != 24:
        raise ValueError(f"Dataset A expected 24 subject/trial files, found {len(by_key)}")
    return [by_key[key] for key in sorted(by_key)]


def dataset_b_files() -> list[Path]:
    # Only the 45 static files are analysis inputs; WIP files are preserved separately.
    base = ROOT / "02_data" / "dataset_B_team" / "raw_static"
    pattern = re.compile(r"^hmd_(S\d+)_static_trial(\d+)_\d+_\d+\.csv$")
    files = [path for path in base.glob("*.csv") if pattern.match(path.name)]
    by_key = {}
    for path in sorted(files):
        match = pattern.match(path.name)
        assert match
        by_key[(match.group(1), int(match.group(2)))] = path
    if len(by_key) != 45:
        raise ValueError(f"Dataset B expected 45 static subject/trial files, found {len(by_key)}")
    return [by_key[key] for key in sorted(by_key)]


def load_dataset_a() -> DatasetBundle:
    sequences, labels, subjects, trials, window_ids = [], [], [], [], []
    file_pattern = re.compile(r"^(p\d+)_trial_(\d+)_(?:train|test)\.csv$")
    signal_columns = ["head_ex", "head_ey", "head_ez", "head_avx", "head_avy", "head_avz"]

    for path in dataset_a_files():
        match = file_pattern.match(path.name)
        assert match
        subject, trial = match.group(1), int(match.group(2))
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        label_changed = frame["label"].ne(frame["label"].shift())
        timestamp_restarted = frame["timestamp"].lt(frame["timestamp"].shift())
        frame["window_id"] = (label_changed | timestamp_restarted).cumsum()

        for window_id, window in frame.groupby("window_id", sort=True):
            canonical = A_LABEL_MAP.get(str(window["label"].iloc[0]))
            if canonical is None:
                continue
            values = window[signal_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            values = baseline_relative_signals(values)
            sequences.append(resample(values))
            labels.append(LABEL_TO_INDEX[canonical])
            subjects.append(subject)
            trials.append(trial)
            window_ids.append(f"{subject}_t{trial}_w{int(window_id):03d}")

    bundle = DatasetBundle(
        name="dataset_A",
        sequences=np.stack(sequences).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects),
        trials=np.asarray(trials, dtype=np.int64),
        window_ids=np.asarray(window_ids),
    )
    validate_bundle(bundle, expected_windows=1440, expected_subjects=8, expected_per_class=240)
    return bundle


def load_dataset_b() -> DatasetBundle:
    sequences, labels, subjects, trials, window_ids = [], [], [], [], []
    file_pattern = re.compile(r"^hmd_(S\d+)_static_trial(\d+)_\d+_\d+\.csv$")
    signal_columns = ["euler_pitch", "euler_yaw", "euler_roll", "angvel_x", "angvel_y", "angvel_z"]

    for path in dataset_b_files():
        match = file_pattern.match(path.name)
        assert match
        subject, trial = match.group(1), int(match.group(2))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            canonical = B_LABEL_MAP.get(row.get("label", ""))
            if row.get("phase") != "active" or canonical is None:
                continue
            groups[(canonical, int(row["rep_id"]))].append(row)

        for (canonical, repetition), group_rows in sorted(groups.items()):
            values = np.array(
                [[float(row[column]) for column in signal_columns] for row in group_rows],
                dtype=np.float32,
            )
            values = baseline_relative_signals(values)
            sequences.append(resample(values))
            labels.append(LABEL_TO_INDEX[canonical])
            subjects.append(subject)
            trials.append(trial)
            window_ids.append(f"{subject}_t{trial}_{canonical}_r{repetition:02d}")

    bundle = DatasetBundle(
        name="dataset_B",
        sequences=np.stack(sequences).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects),
        trials=np.asarray(trials, dtype=np.int64),
        window_ids=np.asarray(window_ids),
    )
    validate_bundle(bundle, expected_windows=1620, expected_subjects=15, expected_per_class=270)
    return bundle


def validate_bundle(bundle: DatasetBundle, expected_windows: int, expected_subjects: int, expected_per_class: int) -> None:
    if len(bundle.labels) != expected_windows:
        raise ValueError(f"{bundle.name}: expected {expected_windows} windows, found {len(bundle.labels)}")
    if len(set(bundle.subjects)) != expected_subjects:
        raise ValueError(f"{bundle.name}: expected {expected_subjects} subjects")
    counts = Counter(bundle.labels.tolist())
    if any(counts[index] != expected_per_class for index in range(len(LABELS))):
        raise ValueError(f"{bundle.name}: unbalanced class counts {counts}")
    if bundle.sequences.shape != (expected_windows, SEQUENCE_LENGTH, 6):
        raise ValueError(f"{bundle.name}: unexpected sequence shape {bundle.sequences.shape}")


def rf_features(sequences: np.ndarray) -> np.ndarray:
    """Extract the same 8 statistics from each of the six shared channels."""
    feature_blocks = []
    for channel in range(sequences.shape[2]):
        values = sequences[:, :, channel]
        abs_peak_index = np.abs(values).argmax(axis=1)
        signed_peak = values[np.arange(len(values)), abs_peak_index]
        block = np.column_stack(
            [
                values.mean(axis=1),
                values.std(axis=1),
                values.min(axis=1),
                values.max(axis=1),
                values.max(axis=1) - values.min(axis=1),
                np.sqrt(np.square(values).mean(axis=1)),
                signed_peak,
                values[:, -1] - values[:, 0],
            ]
        )
        feature_blocks.append(block)
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def fold_definitions(bundle: DatasetBundle, protocol: str) -> list[dict[str, object]]:
    folds = []
    if protocol == "loto":
        for subject in sorted(set(bundle.subjects)):
            subject_mask = bundle.subjects == subject
            for heldout_trial in sorted(set(bundle.trials[subject_mask])):
                test = np.where(subject_mask & (bundle.trials == heldout_trial))[0]
                train = np.where(subject_mask & (bundle.trials != heldout_trial))[0]
                folds.append(
                    {
                        "fold_id": f"{subject}_trial{heldout_trial}",
                        "heldout_subject": str(subject),
                        "heldout_trial": int(heldout_trial),
                        "train": train,
                        "test": test,
                    }
                )
    elif protocol == "loso":
        for subject in sorted(set(bundle.subjects)):
            test = np.where(bundle.subjects == subject)[0]
            train = np.where(bundle.subjects != subject)[0]
            folds.append(
                {
                    "fold_id": str(subject),
                    "heldout_subject": str(subject),
                    "heldout_trial": -1,
                    "train": train,
                    "test": test,
                }
            )
    else:
        raise ValueError(protocol)
    return folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass


def standardize_sequences(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def build_model(model_name: str, input_shape: tuple[int, int]) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    if model_name == "cnn1d":
        x = layers.Conv1D(32, 5, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif model_name == "lstm":
        x = layers.LSTM(64)(inputs)
    else:
        raise ValueError(model_name)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_model(
    model_name: str,
    bundle: DatasetBundle,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    epochs: int,
) -> np.ndarray:
    y_train = bundle.labels[train_indices]
    if model_name == "rf":
        features = rf_features(bundle.sequences)
        model = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
        )
        model.fit(features[train_indices], y_train)
        return model.predict(features[test_indices]).astype(np.int64)

    if tf is None:
        raise RuntimeError("TensorFlow is required for cnn1d/lstm")
    set_seed(seed)
    keras.backend.clear_session()
    x_train, x_test = standardize_sequences(bundle.sequences[train_indices], bundle.sequences[test_indices])
    all_train_positions = np.arange(len(train_indices))
    try:
        fit_positions, validation_positions = train_test_split(
            all_train_positions,
            test_size=0.167,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        fit_positions, validation_positions = train_test_split(
            all_train_positions,
            test_size=0.167,
            random_state=seed,
        )
    model = build_model(model_name, (x_train.shape[1], x_train.shape[2]))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    model.fit(
        x_train[fit_positions],
        y_train[fit_positions],
        validation_data=(x_train[validation_positions], y_train[validation_positions]),
        epochs=epochs,
        batch_size=min(32, len(fit_positions)),
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    probabilities = model.predict(x_test, batch_size=64, verbose=0)
    return probabilities.argmax(axis=1).astype(np.int64)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_fold(
    bundle: DatasetBundle,
    protocol: str,
    model_name: str,
    seed: int,
    fold: dict[str, object],
    run_dir: Path,
    epochs: int,
) -> Path:
    fold_key = re.sub(r"[^A-Za-z0-9_-]", "_", str(fold["fold_id"]))
    output = run_dir / f"{bundle.name}__{protocol}__{model_name}__seed{seed}__{fold_key}.npz"
    if output.exists():
        return output
    train_indices = np.asarray(fold["train"], dtype=np.int64)
    test_indices = np.asarray(fold["test"], dtype=np.int64)
    prediction = predict_model(model_name, bundle, train_indices, test_indices, seed, epochs)
    np.savez_compressed(
        output,
        y_true=bundle.labels[test_indices],
        y_pred=prediction,
        test_indices=test_indices,
        train_size=np.asarray([len(train_indices)]),
        test_size=np.asarray([len(test_indices)]),
        fold_id=np.asarray([str(fold["fold_id"])]),
        heldout_subject=np.asarray([str(fold["heldout_subject"])]),
        heldout_trial=np.asarray([int(fold["heldout_trial"])]),
    )
    return output


def collect_results(output_dir: Path, bundles: dict[str, DatasetBundle], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions_by_group: dict[tuple[str, str, str, int], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"true": [], "pred": []}
    )
    for path in sorted((output_dir / "runs").glob("*.npz")):
        parts = path.stem.split("__")
        if len(parts) < 5:
            continue
        dataset, protocol, model_name, seed_token = parts[:4]
        seed = int(seed_token.removeprefix("seed"))
        if dataset not in bundles or seed not in seeds:
            continue
        data = np.load(path)
        y_true, y_pred = data["y_true"], data["y_pred"]
        metrics = metric_row(y_true, y_pred)
        row = {
            "dataset": dataset,
            "protocol": protocol,
            "model": model_name,
            "seed": seed,
            "fold_id": str(data["fold_id"][0]),
            "heldout_subject": str(data["heldout_subject"][0]),
            "heldout_trial": int(data["heldout_trial"][0]),
            "train_size": int(data["train_size"][0]),
            "test_size": int(data["test_size"][0]),
            **metrics,
        }
        rows.append(row)
        key = (dataset, protocol, model_name, seed)
        predictions_by_group[key]["true"].append(y_true)
        predictions_by_group[key]["pred"].append(y_pred)

    fold_metrics = pd.DataFrame(rows)
    aggregate_rows = []
    for (dataset, protocol, model_name, seed), values in sorted(predictions_by_group.items()):
        y_true = np.concatenate(values["true"])
        y_pred = np.concatenate(values["pred"])
        aggregate_rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "model": model_name,
                "seed": seed,
                "n_predictions": len(y_true),
                **metric_row(y_true, y_pred),
            }
        )
    aggregate_metrics = pd.DataFrame(aggregate_rows)
    return fold_metrics, aggregate_metrics


def save_summary(output_dir: Path, fold_metrics: pd.DataFrame, aggregate_metrics: pd.DataFrame) -> pd.DataFrame:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    aggregate_metrics.to_csv(results_dir / "seed_aggregate_metrics.csv", index=False)
    summary = (
        aggregate_metrics.groupby(["dataset", "protocol", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values(["dataset", "protocol", "model"])
    )
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    return summary


def save_confusions(output_dir: Path, bundles: dict[str, DatasetBundle], seeds: list[int]) -> None:
    figures_dir = output_dir / "poster_figures" / "confusion_matrices"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for dataset in bundles:
        for protocol in ["loto", "loso"]:
            for model_name in ["rf", "cnn1d", "lstm"]:
                matrices = []
                for seed in seeds:
                    true_parts, pred_parts = [], []
                    pattern = f"{dataset}__{protocol}__{model_name}__seed{seed}__*.npz"
                    for path in sorted((output_dir / "runs").glob(pattern)):
                        data = np.load(path)
                        true_parts.append(data["y_true"])
                        pred_parts.append(data["y_pred"])
                    if true_parts:
                        matrices.append(confusion_matrix(np.concatenate(true_parts), np.concatenate(pred_parts), labels=range(6)))
                if not matrices:
                    continue
                matrix = np.mean(matrices, axis=0)
                normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
                np.savetxt(figures_dir / f"{dataset}_{model_name}_{protocol}_confusion_matrix.csv", matrix, delimiter=",", fmt="%.3f")
                fig, ax = plt.subplots(figsize=(6.2, 5.4))
                image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
                for i in range(6):
                    for j in range(6):
                        value = normalized[i, j]
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "#222")
                ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
                ax.set_yticks(range(6), LABELS)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(f"{dataset.replace('_', ' ').title()} · {model_name.upper()} · {protocol.upper()}")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(figures_dir / f"{dataset}_{model_name}_{protocol}_confusion_matrix.png", dpi=240)
                plt.close(fig)


def save_main_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    figures_dir = output_dir / "poster_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_order = ["rf", "cnn1d", "lstm"]
    protocol_colors = {"loto": "#0F7C83", "loso": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, dataset in zip(axes, ["dataset_A", "dataset_B"]):
        subset = summary[summary["dataset"] == dataset]
        x = np.arange(len(model_order))
        width = 0.34
        for offset, protocol in [(-width / 2, "loto"), (width / 2, "loso")]:
            values, errors = [], []
            for model_name in model_order:
                row = subset[(subset["protocol"] == protocol) & (subset["model"] == model_name)]
                if row.empty:
                    plt.close(fig)
                    return
                values.append(float(row["macro_f1_mean"].iloc[0]))
                error = float(row["macro_f1_std"].fillna(0).iloc[0])
                errors.append(error)
            bars = ax.bar(x + offset, values, width, yerr=errors, label=protocol.upper(), color=protocol_colors[protocol], capsize=3)
            ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
        ax.set_xticks(x, ["RF", "CNN1D", "LSTM"])
        ax.set_ylim(0, 1.08)
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Macro F1")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("HMD Head-Gesture Recognition under LOTO and LOSO", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "main_macro_f1_comparison.png", dpi=260)
    plt.close(fig)


def save_fold_distribution(output_dir: Path, fold_metrics: pd.DataFrame) -> None:
    figures_dir = output_dir / "poster_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_order = ["rf", "cnn1d", "lstm"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, dataset in zip(axes, ["dataset_A", "dataset_B"]):
        subset = fold_metrics[(fold_metrics["dataset"] == dataset) & (fold_metrics["protocol"] == "loso")]
        data = [subset[subset["model"] == model_name]["macro_f1"].to_numpy() for model_name in model_order]
        if any(len(values) == 0 for values in data):
            plt.close(fig)
            return
        ax.boxplot(data, tick_labels=["RF", "CNN1D", "LSTM"], patch_artist=True, boxprops={"facecolor": "#DCECEF"}, medianprops={"color": "#C43C39", "linewidth": 2})
        rng = np.random.default_rng(42)
        for position, values in enumerate(data, start=1):
            ax.scatter(position + rng.normal(0, 0.045, len(values)), values, s=14, alpha=0.45, color="#174A5B")
        ax.set_ylim(0, 1.04)
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("LOSO Fold Macro F1")
    fig.suptitle("Cross-User Performance Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "loso_fold_distribution.png", dpi=260)
    plt.close(fig)


def write_config(output_dir: Path, args: argparse.Namespace, bundles: dict[str, DatasetBundle]) -> None:
    config = {
        "created_for": "Japan A0 poster analysis",
        "datasets": {
            name: {
                "windows": len(bundle.labels),
                "subjects": sorted(set(bundle.subjects.tolist())),
                "trials": sorted(set(int(value) for value in bundle.trials.tolist())),
                "class_counts": {LABELS[index]: int((bundle.labels == index).sum()) for index in range(6)},
            }
            for name, bundle in bundles.items()
        },
        "signals": ["relative_euler_axis_1", "relative_euler_axis_2", "relative_euler_axis_3", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"],
        "relative_angle_baseline_frames": 5,
        "sequence_length": SEQUENCE_LENGTH,
        "rf_features_per_channel": RF_STATS,
        "rf_trees": 400,
        "cnn": "Conv1D(32,k5)-BN-ReLU-Pool-Conv1D(64,k5)-BN-ReLU-Pool-Conv1D(64,k3)-GAP-Dense(6)",
        "lstm": "LSTM(64)-Dense(6)",
        "normalization": "channel-wise mean/std fitted on each fold training set only",
        "validation": "16.7% stratified split from fold training set; test fold never used for early stopping",
        "epochs_max": args.epochs,
        "early_stopping_patience": 8,
        "seeds": args.seeds,
        "protocols": args.protocols,
        "models": args.models,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=["dataset_A", "dataset_B"], default=["dataset_A", "dataset_B"])
    parser.add_argument("--protocols", nargs="+", choices=["loto", "loso"], default=["loto", "loso"])
    parser.add_argument("--models", nargs="+", choices=["rf", "cnn1d", "lstm"], default=["rf", "cnn1d", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Fixed sequence length after linear resampling (default: 128)",
    )
    parser.add_argument("--limit-folds", type=int, default=0, help="Debug only: run at most N folds per protocol")
    return parser.parse_args()


def main() -> None:
    global SEQUENCE_LENGTH
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2")
    SEQUENCE_LENGTH = int(args.sequence_length)
    output_dir = args.output.resolve()
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)
    (output_dir / ".mpl-cache").mkdir(parents=True, exist_ok=True)
    (output_dir / ".cache").mkdir(parents=True, exist_ok=True)
    loaders = {"dataset_A": load_dataset_a, "dataset_B": load_dataset_b}  # release edit: load only requested datasets
    bundles = {name: loaders[name]() for name in args.datasets}
    write_config(output_dir, args, bundles)

    total = 0
    for bundle in bundles.values():
        for protocol in args.protocols:
            folds = fold_definitions(bundle, protocol)
            if args.limit_folds:
                folds = folds[: args.limit_folds]
            total += len(folds) * len(args.models) * len(args.seeds)
    completed = 0
    print(f"Planned fold runs: {total}", flush=True)
    for bundle in bundles.values():
        print(f"{bundle.name}: sequences={bundle.sequences.shape}, subjects={len(set(bundle.subjects))}", flush=True)
        for protocol in args.protocols:
            folds = fold_definitions(bundle, protocol)
            if args.limit_folds:
                folds = folds[: args.limit_folds]
            for model_name in args.models:
                for seed in args.seeds:
                    for fold in folds:
                        output = run_fold(bundle, protocol, model_name, seed, fold, output_dir / "runs", args.epochs)
                        completed += 1
                        print(
                            f"[{completed}/{total}] {bundle.name} {protocol.upper()} {model_name.upper()} seed={seed} fold={fold['fold_id']} -> {output.name}",
                            flush=True,
                        )

    fold_metrics, aggregate_metrics = collect_results(output_dir, bundles, args.seeds)
    summary = save_summary(output_dir, fold_metrics, aggregate_metrics)
    if not args.limit_folds:
        save_confusions(output_dir, bundles, args.seeds)
        save_main_plot(output_dir, summary)
        save_fold_distribution(output_dir, fold_metrics)
    print("\nSummary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
