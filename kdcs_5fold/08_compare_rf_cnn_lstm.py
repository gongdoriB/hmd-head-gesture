#!/usr/bin/env python3
"""
Compare RF, Keras 1D-CNN, and Keras LSTM on the full shuffled 5-fold window dataset.

RF uses window-level statistical features from results/feature_table.csv.
CNN/LSTM use raw window sequences resampled/padded to 150 frames with 6 channels:
head_ex, head_ey, head_ez, head_avx, head_avy, head_avz.
"""

from __future__ import annotations

import json
import os
import random
import re
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "TensorFlow/Keras is required for CNN/LSTM in this script. "
        "Install it first, for example: python3 -m pip install tensorflow"
    ) from exc


LABELS = [
    "Rotating_Left",
    "Rotating_Right",
    "Tilting_Up",
    "Tilting_Down",
    "Leaning_Left",
    "Leaning_Right",
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
SIGNAL_COLUMNS = ["head_ex", "head_ey", "head_ez", "head_avx", "head_avy", "head_avz"]
SIGNAL_PREFIXES = SIGNAL_COLUMNS
FEATURE_SUFFIXES = ["mean", "std", "min", "max", "range", "rms", "first", "last", "delta"]
FEATURE_COLUMNS = [f"{prefix}_{suffix}" for prefix in SIGNAL_PREFIXES for suffix in FEATURE_SUFFIXES]
SEQUENCE_LENGTH = 150
N_SPLITS = 5
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 6


def set_seed(seed: int) -> None:
    """Make training as reproducible as practical on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # TensorFlow thread settings must be set before runtime initialization.
        pass


def find_data_files(root: Path) -> list[Path]:
    """Find pXX_trial_N_train/test.csv files, deduplicated by filename."""
    candidates = [root / "data" / "raw", root / "CollectedData", root]
    search_base = next((candidate for candidate in candidates if candidate.exists()), root)
    pattern = re.compile(r"^p\d+_trial_\d+_(train|test)\.csv$")
    files_by_name: dict[str, Path] = {}
    for directory in sorted(path for path in search_base.iterdir() if path.is_dir()):
        for csv_path in sorted(directory.glob("*.csv")):
            if pattern.match(csv_path.name):
                files_by_name.setdefault(csv_path.name, csv_path)
    # Keep the same deterministic path ordering used by 01_make_feature_table.py
    # so sequence windows align with rows in results/feature_table.csv.
    return sorted(files_by_name.values())


def segment_windows(df: pd.DataFrame) -> pd.Series:
    """Return a 1-based id for each gesture repetition window."""
    label_changed = df["label"].ne(df["label"].shift())
    timestamp_restarted = df["timestamp"].lt(df["timestamp"].shift())
    return (label_changed | timestamp_restarted).cumsum()


def resample_window(values: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample one window to a fixed sequence length."""
    if len(values) == target_length:
        return values.astype(np.float32)

    old_x = np.linspace(0.0, 1.0, num=len(values))
    new_x = np.linspace(0.0, 1.0, num=target_length)
    resampled = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for channel_idx in range(values.shape[1]):
        resampled[:, channel_idx] = np.interp(new_x, old_x, values[:, channel_idx])
    return resampled


def load_sequence_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load raw CSVs and build fixed-length window sequences."""
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []

    for path in find_data_files(ROOT):
        match = re.match(r"^(?P<subject_id>p\d+)_trial_(?P<trial>\d+)_(?P<split>train|test)\.csv$", path.name)
        if not match:
            continue

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        for column in SIGNAL_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["window_id"] = segment_windows(df)
        repetition_counts: dict[str, int] = {}

        for window_id, window in df.groupby("window_id", sort=True):
            label = str(window["label"].iloc[0])
            if label not in LABEL_TO_INDEX:
                continue

            repetition_counts[label] = repetition_counts.get(label, 0) + 1
            raw_values = window[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)
            raw_values = np.nan_to_num(raw_values, nan=0.0)

            sequences.append(resample_window(raw_values))
            labels.append(LABEL_TO_INDEX[label])
            metadata_rows.append(
                {
                    "subject_id": match.group("subject_id"),
                    "trial": int(match.group("trial")),
                    "split": match.group("split"),
                    "label": label,
                    "window_id": int(window_id),
                    "repetition_index_within_label": repetition_counts[label],
                    "num_frames": int(len(window)),
                }
            )

    x_seq = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)
    return x_seq, y, metadata


def load_feature_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Load RF feature table."""
    df = pd.read_csv(RESULTS_DIR / "feature_table.csv")
    df = df[df["label"].isin(LABELS)].copy()
    y = df["label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    return df, y


def build_rf() -> Pipeline:
    """RandomForest baseline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def standardize_sequences(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit channel-wise standardization on train only."""
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def build_keras_model(model_name: str, input_shape: tuple[int, int], classes: int) -> keras.Model:
    """Build a compact Keras sequence classifier."""
    inputs = keras.Input(shape=input_shape)

    if model_name == "CNN1D":
        # 1D-CNN learns local temporal motion patterns inside each 3-second gesture window.
        x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(96, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif model_name == "LSTM":
        # Bidirectional LSTM reads the whole motion sequence in temporal order.
        x = layers.Bidirectional(layers.LSTM(64))(inputs)
        x = layers.Dropout(0.25)(x)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_keras_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Train Keras CNN/LSTM and return predicted class indices."""
    set_seed(RANDOM_STATE)
    model = build_keras_model(
        model_name=model_name,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        classes=len(LABELS),
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    return probabilities.argmax(axis=1)


def save_confusion(y_true: list[int], y_pred: list[int], model_name: str) -> str:
    """Save aggregate confusion matrix and return path."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    output_path = RESULTS_DIR / f"{model_name.lower()}_stratified_5fold_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set_title(f"{model_name} Stratified 5-fold Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def evaluate_all_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run RF/CNN/LSTM on the same 5 stratified folds."""
    feature_df, y_feature = load_feature_table()
    x_feature = feature_df[FEATURE_COLUMNS]
    x_seq, y_seq, metadata = load_sequence_dataset()

    if len(y_feature) != len(y_seq):
        raise ValueError(f"Feature rows ({len(y_feature)}) and sequence rows ({len(y_seq)}) do not match.")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = [(train_idx, test_idx) for train_idx, test_idx in splitter.split(x_feature, y_feature)]

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    aggregate_predictions: dict[str, dict[str, list[int]]] = {
        "RF": {"true": [], "pred": []},
        "CNN1D": {"true": [], "pred": []},
        "LSTM": {"true": [], "pred": []},
    }
    rf_importances: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        y_test = y_feature[test_idx]

        rf = build_rf()
        rf.fit(x_feature.iloc[train_idx], y_feature[train_idx])
        rf_pred = rf.predict(x_feature.iloc[test_idx])
        rf_importances.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "feature": FEATURE_COLUMNS,
                    "importance": rf.named_steps["rf"].feature_importances_,
                }
            )
        )

        for model_name, pred in [("RF", rf_pred)]:
            fold_rows.append(metrics_row(model_name, fold, y_test, pred, len(train_idx), len(test_idx)))
            aggregate_predictions[model_name]["true"].extend(y_test.tolist())
            aggregate_predictions[model_name]["pred"].extend(pred.tolist())

        x_train_seq, x_test_seq = standardize_sequences(x_seq[train_idx], x_seq[test_idx])
        for model_name in ["CNN1D", "LSTM"]:
            pred = train_keras_model(model_name, x_train_seq, y_seq[train_idx], x_test_seq)
            fold_rows.append(metrics_row(model_name, fold, y_seq[test_idx], pred, len(train_idx), len(test_idx)))
            aggregate_predictions[model_name]["true"].extend(y_seq[test_idx].tolist())
            aggregate_predictions[model_name]["pred"].extend(pred.tolist())

    for model_name, values in aggregate_predictions.items():
        y_true = values["true"]
        y_pred = values["pred"]
        report = classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
        summary_rows.append(
            {
                "model": model_name,
                "folds": N_SPLITS,
                "total_test_rows": len(y_true),
                "accuracy_mean": pd.DataFrame(fold_rows).query("model == @model_name")["accuracy"].mean(),
                "accuracy_std": pd.DataFrame(fold_rows).query("model == @model_name")["accuracy"].std(),
                "macro_f1_mean": pd.DataFrame(fold_rows).query("model == @model_name")["macro_f1"].mean(),
                "macro_f1_std": pd.DataFrame(fold_rows).query("model == @model_name")["macro_f1"].std(),
                "aggregate_accuracy": accuracy_score(y_true, y_pred),
                "aggregate_macro_f1": f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0),
                "classification_report_json": json.dumps(report, ensure_ascii=False),
                "confusion_matrix_json": json.dumps(matrix.tolist(), ensure_ascii=False),
                "confusion_matrix_png": save_confusion(y_true, y_pred, model_name),
            }
        )

    importances = pd.concat(rf_importances, ignore_index=True)
    importance_summary = (
        importances.groupby("feature")
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"))
        .reset_index()
        .sort_values("importance_mean", ascending=False)
    )

    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows), importance_summary


def metrics_row(model_name: str, fold: int, y_true: np.ndarray, y_pred: np.ndarray, train_rows: int, test_rows: int) -> dict[str, object]:
    """Create one fold-level metric row."""
    return {
        "model": model_name,
        "fold": fold,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0),
    }


def save_importance_plot(importance_summary: pd.DataFrame) -> None:
    """Save top RF feature importance plot."""
    top = importance_summary.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"].fillna(0), color="#4c78a8")
    ax.set_xlabel("Mean impurity-based importance")
    ax.set_title("Top RF Feature Importances Across 5 Folds")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_top_feature_importance_5fold.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)

    fold_metrics, summary_metrics, rf_importance = evaluate_all_models()
    fold_metrics.to_csv(RESULTS_DIR / "rf_cnn_lstm_5fold_fold_metrics.csv", index=False)
    summary_metrics.to_csv(RESULTS_DIR / "rf_cnn_lstm_5fold_summary.csv", index=False)
    rf_importance.to_csv(RESULTS_DIR / "rf_feature_importance_5fold.csv", index=False)
    save_importance_plot(rf_importance)

    print("RF/CNN/LSTM 5-fold summary:")
    print(
        summary_metrics[
            [
                "model",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "aggregate_accuracy",
                "aggregate_macro_f1",
            ]
        ].to_string(index=False)
    )
    print("\nTop RF features:")
    print(rf_importance.head(15).to_string(index=False))
    print("\nSaved:")
    for output in [
        "rf_cnn_lstm_5fold_fold_metrics.csv",
        "rf_cnn_lstm_5fold_summary.csv",
        "rf_feature_importance_5fold.csv",
        "rf_top_feature_importance_5fold.png",
        "rf_stratified_5fold_confusion_matrix.png",
        "cnn1d_stratified_5fold_confusion_matrix.png",
        "lstm_stratified_5fold_confusion_matrix.png",
    ]:
        print(f"  {RESULTS_DIR / output}")


if __name__ == "__main__":
    main()
