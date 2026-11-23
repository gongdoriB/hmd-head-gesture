#!/usr/bin/env python3
"""Leakage-safe nested LOSO hyperparameter tuning for the HMD poster study.

Outer LOSO subjects are never used for hyperparameter selection. Candidate
configurations are ranked with subject-grouped inner cross-validation on the
outer-training subjects. The selected configuration is then refit and tested
on the untouched outer subject with three random seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DEFAULT = ROOT / "04_results" / "nested_loso_tuning"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DEFAULT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DEFAULT / ".cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # release edit: main script lives one level up

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, train_test_split

import unified_loto_loso_analysis as base

SEARCH_SPACES: dict[str, list[dict[str, object]]] = {
    "rf": [
        {"id": "rf_baseline", "trees": 400, "max_depth": None, "min_leaf": 1, "max_features": "sqrt"},
        {"id": "rf_leaf2", "trees": 800, "max_depth": None, "min_leaf": 2, "max_features": "sqrt"},
        {"id": "rf_leaf4", "trees": 800, "max_depth": None, "min_leaf": 4, "max_features": "sqrt"},
        {"id": "rf_depth20", "trees": 800, "max_depth": 20, "min_leaf": 2, "max_features": 0.5},
        {"id": "rf_depth12", "trees": 800, "max_depth": 12, "min_leaf": 4, "max_features": "sqrt"},
        {"id": "rf_half_features", "trees": 800, "max_depth": None, "min_leaf": 4, "max_features": 0.5},
    ],
    "cnn1d": [
        {
            "id": "cnn_baseline",
            "filters": [32, 64, 64],
            "kernels": [5, 5, 3],
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "augment": False,
        },
        {
            "id": "cnn_regularized",
            "filters": [32, 64, 64],
            "kernels": [5, 5, 3],
            "dropout": 0.25,
            "learning_rate": 3e-4,
            "augment": False,
        },
        {
            "id": "cnn_regularized_aug",
            "filters": [32, 64, 64],
            "kernels": [5, 5, 3],
            "dropout": 0.25,
            "learning_rate": 3e-4,
            "augment": True,
        },
        {
            "id": "cnn_wide_aug",
            "filters": [64, 96, 128],
            "kernels": [7, 5, 3],
            "dropout": 0.30,
            "learning_rate": 3e-4,
            "augment": True,
        },
    ],
    "lstm": [
        {"id": "lstm_baseline", "units": 64, "dropout": 0.0, "learning_rate": 1e-3, "augment": False},
        {"id": "lstm_regularized", "units": 64, "dropout": 0.30, "learning_rate": 3e-4, "augment": False},
        {"id": "lstm_regularized_aug", "units": 64, "dropout": 0.30, "learning_rate": 3e-4, "augment": True},
        {"id": "lstm_128_aug", "units": 128, "dropout": 0.35, "learning_rate": 3e-4, "augment": True},
    ],
}


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def augment_sequences(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Append one mildly warped/noisy copy of each standardized train sequence."""
    rng = np.random.default_rng(seed)
    augmented = np.empty_like(x)
    base_grid = np.linspace(0.0, 1.0, x.shape[1], dtype=np.float32)
    for index, sequence in enumerate(x):
        increments = rng.lognormal(mean=0.0, sigma=0.10, size=x.shape[1]).astype(np.float32)
        warped_grid = np.cumsum(increments)
        warped_grid = (warped_grid - warped_grid[0]) / max(float(warped_grid[-1] - warped_grid[0]), 1e-6)
        warped = np.column_stack(
            [np.interp(base_grid, warped_grid, sequence[:, channel]) for channel in range(sequence.shape[1])]
        ).astype(np.float32)
        amplitude = rng.uniform(0.90, 1.10, size=(1, sequence.shape[1])).astype(np.float32)
        noise = rng.normal(0.0, 0.015, size=sequence.shape).astype(np.float32)
        augmented[index] = warped * amplitude + noise
    return np.concatenate([x, augmented], axis=0), np.concatenate([y, y], axis=0)


def build_neural_model(model_name: str, config: dict[str, object], input_shape: tuple[int, int]):
    if base.tf is None or base.keras is None:
        raise RuntimeError("TensorFlow is required for CNN1D/LSTM tuning")
    keras = base.keras
    layers = base.layers
    inputs = keras.Input(shape=input_shape)
    if model_name == "cnn1d":
        filters = [int(value) for value in config["filters"]]
        kernels = [int(value) for value in config["kernels"]]
        x = inputs
        for block, (width, kernel) in enumerate(zip(filters, kernels)):
            x = layers.Conv1D(width, kernel, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            if block < 2:
                x = layers.MaxPooling1D(2)(x)
            if float(config["dropout"]) > 0:
                x = layers.Dropout(float(config["dropout"]))(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif model_name == "lstm":
        x = layers.LSTM(int(config["units"]))(inputs)
        if float(config["dropout"]) > 0:
            x = layers.Dropout(float(config["dropout"]))(x)
    else:
        raise ValueError(model_name)
    outputs = layers.Dense(len(base.LABELS), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(config["learning_rate"]), clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def neural_predict(
    model_name: str,
    config: dict[str, object],
    bundle: base.DatasetBundle,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
) -> np.ndarray:
    base.set_seed(seed)
    base.keras.backend.clear_session()
    x_train, x_eval = base.standardize_sequences(bundle.sequences[train_indices], bundle.sequences[eval_indices])
    y_train = bundle.labels[train_indices]
    positions = np.arange(len(train_indices))
    try:
        fit_pos, early_pos = train_test_split(
            positions, test_size=0.15, random_state=seed, stratify=y_train
        )
    except ValueError:
        fit_pos, early_pos = train_test_split(positions, test_size=0.15, random_state=seed)
    x_fit = x_train[fit_pos]
    y_fit = y_train[fit_pos]
    if bool(config.get("augment", False)):
        x_fit, y_fit = augment_sequences(x_fit, y_fit, seed)
    model = build_neural_model(model_name, config, (x_train.shape[1], x_train.shape[2]))
    callbacks = [
        base.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=1e-4, restore_best_weights=True
        )
    ]
    model.fit(
        x_fit,
        y_fit,
        validation_data=(x_train[early_pos], y_train[early_pos]),
        epochs=epochs,
        batch_size=min(32, len(x_fit)),
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    prediction = model.predict(x_eval, batch_size=64, verbose=0).argmax(axis=1).astype(np.int64)
    base.keras.backend.clear_session()
    return prediction


def predict_config(
    model_name: str,
    config: dict[str, object],
    bundle: base.DatasetBundle,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
) -> np.ndarray:
    if model_name == "rf":
        features = base.rf_features(bundle.sequences)
        model = RandomForestClassifier(
            n_estimators=int(config["trees"]),
            max_depth=None if config["max_depth"] is None else int(config["max_depth"]),
            min_samples_leaf=int(config["min_leaf"]),
            max_features=config["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(features[train_indices], bundle.labels[train_indices])
        return model.predict(features[eval_indices]).astype(np.int64)
    return neural_predict(model_name, config, bundle, train_indices, eval_indices, seed, epochs, patience)


def select_config(
    output_dir: Path,
    bundle: base.DatasetBundle,
    outer_fold: dict[str, object],
    model_name: str,
    inner_splits: int,
    tuning_seed: int,
    inner_epochs: int,
    candidate_limit: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    cache_dir = output_dir / "inner_selection"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{bundle.name}__{safe_key(str(outer_fold['fold_id']))}__{model_name}.csv"
    candidates = SEARCH_SPACES[model_name][: candidate_limit or None]
    expected_ids = {str(candidate["id"]) for candidate in candidates}
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(cached["config_id"]) == expected_ids and int(cached["inner_splits"].iloc[0]) == inner_splits:
            best_id = cached.sort_values(["mean_inner_macro_f1", "config_id"], ascending=[False, True]).iloc[0]["config_id"]
            return next(candidate for candidate in candidates if candidate["id"] == best_id), cached

    outer_train = np.asarray(outer_fold["train"], dtype=np.int64)
    groups = bundle.subjects[outer_train]
    splitter = GroupKFold(n_splits=inner_splits)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        scores = []
        for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(
            splitter.split(outer_train, bundle.labels[outer_train], groups), start=1
        ):
            inner_train = outer_train[inner_train_pos]
            inner_val = outer_train[inner_val_pos]
            prediction = predict_config(
                model_name,
                candidate,
                bundle,
                inner_train,
                inner_val,
                tuning_seed,
                inner_epochs,
                patience=5,
            )
            scores.append(float(f1_score(bundle.labels[inner_val], prediction, average="macro", zero_division=0)))
        rows.append(
            {
                "dataset": bundle.name,
                "outer_fold": str(outer_fold["fold_id"]),
                "model": model_name,
                "config_id": str(candidate["id"]),
                "mean_inner_macro_f1": float(np.mean(scores)),
                "std_inner_macro_f1": float(np.std(scores, ddof=0)),
                "inner_splits": inner_splits,
                "tuning_seed": tuning_seed,
                "config_json": json.dumps(candidate, sort_keys=True),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["mean_inner_macro_f1", "config_id"], ascending=[False, True])
    frame.to_csv(cache_path, index=False)
    best_id = str(frame.iloc[0]["config_id"])
    return next(candidate for candidate in candidates if candidate["id"] == best_id), frame


def run_outer_fold(
    output_dir: Path,
    bundle: base.DatasetBundle,
    outer_fold: dict[str, object],
    model_name: str,
    config: dict[str, object],
    seed: int,
    outer_epochs: int,
) -> Path:
    run_dir = output_dir / "outer_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / (
        f"{bundle.name}__loso__{model_name}__seed{seed}__{safe_key(str(outer_fold['fold_id']))}__{config['id']}.npz"
    )
    if path.exists():
        return path
    train_indices = np.asarray(outer_fold["train"], dtype=np.int64)
    test_indices = np.asarray(outer_fold["test"], dtype=np.int64)
    prediction = predict_config(
        model_name,
        config,
        bundle,
        train_indices,
        test_indices,
        seed,
        outer_epochs,
        patience=8,
    )
    np.savez_compressed(
        path,
        y_true=bundle.labels[test_indices],
        y_pred=prediction,
        test_indices=test_indices,
        train_size=np.asarray([len(train_indices)]),
        test_size=np.asarray([len(test_indices)]),
        fold_id=np.asarray([str(outer_fold["fold_id"])]),
        heldout_subject=np.asarray([str(outer_fold["heldout_subject"])]),
        config_id=np.asarray([str(config["id"])]),
    )
    return path


def collect_outer(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((output_dir / "outer_runs").glob("*.npz")):
        parts = path.stem.split("__")
        dataset, _, model_name, seed_part, fold_id, config_id = parts
        data = np.load(path, allow_pickle=False)
        row = {
            "dataset": dataset,
            "protocol": "loso",
            "model": model_name,
            "seed": int(seed_part.replace("seed", "")),
            "fold_id": fold_id,
            "heldout_subject": str(data["heldout_subject"][0]),
            "config_id": config_id,
            "train_size": int(data["train_size"][0]),
            "test_size": int(data["test_size"][0]),
        }
        row.update(metric_row(data["y_true"], data["y_pred"]))
        rows.append(row)
    folds = pd.DataFrame(rows)
    seed_rows = []
    for (dataset, model_name, seed), _ in folds.groupby(["dataset", "model", "seed"]):
        pattern = f"{dataset}__loso__{model_name}__seed{int(seed)}__*.npz"
        paths = sorted((output_dir / "outer_runs").glob(pattern))
        y_true = np.concatenate([np.load(path)["y_true"] for path in paths])
        y_pred = np.concatenate([np.load(path)["y_pred"] for path in paths])
        row = {"dataset": dataset, "model": model_name, "seed": int(seed)}
        row.update(metric_row(y_true, y_pred))
        seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    summary = (
        seeds.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
    )
    return folds, seeds, summary
def save_outputs(output_dir: Path, folds: pd.DataFrame, seeds: pd.DataFrame, summary: pd.DataFrame) -> None:
    result_dir = output_dir / "results"
    figure_dir = output_dir / "poster_figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(result_dir / "tuned_fold_metrics.csv", index=False)
    seeds.to_csv(result_dir / "tuned_seed_metrics.csv", index=False)
    summary.to_csv(result_dir / "tuned_summary_metrics.csv", index=False)

    baseline = pd.read_csv(ROOT / "04_results" / "final_loto_loso" / "results" / "summary_metrics.csv")
    baseline = baseline[baseline["protocol"].eq("loso")][["dataset", "model", "macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "baseline_macro_f1"}
    )
    comparison = summary.merge(baseline, on=["dataset", "model"], how="left")
    comparison["tuning_delta"] = comparison["macro_f1_mean"] - comparison["baseline_macro_f1"]
    comparison.to_csv(result_dir / "baseline_vs_tuned.csv", index=False)

    robust = (
        comparison.groupby("model", as_index=False)
        .agg(
            cross_dataset_mean=("macro_f1_mean", "mean"),
            worst_dataset=("macro_f1_mean", "min"),
            best_dataset=("macro_f1_mean", "max"),
        )
        .sort_values(["worst_dataset", "cross_dataset_mean"], ascending=False)
    )
    robust.to_csv(result_dir / "cross_dataset_robustness.csv", index=False)
    selection_counts = (
        folds[["dataset", "model", "fold_id", "config_id"]]
        .drop_duplicates()
        .groupby(["dataset", "model", "config_id"], as_index=False)
        .size()
        .rename(columns={"size": "outer_folds_selected"})
    )
    selection_counts.to_csv(result_dir / "selected_config_counts.csv", index=False)

    models = ["rf", "cnn1d", "lstm"]
    colors = {"baseline": "#16818A", "tuned": "#E55654"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, dataset in zip(axes, ["dataset_A", "dataset_B"]):
        sub = comparison[comparison["dataset"].eq(dataset)].set_index("model").reindex(models)
        x = np.arange(len(models))
        width = 0.36
        ax.bar(x - width / 2, sub["baseline_macro_f1"], width, color=colors["baseline"], label="Baseline")
        ax.bar(x + width / 2, sub["macro_f1_mean"], width, color=colors["tuned"], label="Nested tuned")
        for xpos, value in zip(x + width / 2, sub["macro_f1_mean"]):
            ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x, ["RF", "CNN1D", "LSTM"])
        ax.set_ylim(0.78, 1.02)
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("LOSO Macro F1")
    axes[1].legend(loc="lower right")
    fig.suptitle("Baseline vs Nested-LOSO Hyperparameter Tuning", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_vs_nested_tuned_macro_f1.png", dpi=260)
    plt.close(fig)

    model_colors = {"rf": "#16818A", "cnn1d": "#E55654", "lstm": "#6B6ECF"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, dataset in zip(axes, ["dataset_A", "dataset_B"]):
        sub = summary[summary["dataset"].eq(dataset)].set_index("model").reindex(models)
        x = np.arange(len(models))
        bars = ax.bar(
            x,
            sub["macro_f1_mean"],
            yerr=sub["macro_f1_std"].fillna(0.0),
            capsize=4,
            color=[model_colors[model] for model in models],
        )
        for bar, value in zip(bars, sub["macro_f1_mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=10)
        ax.set_xticks(x, ["RF", "CNN1D", "LSTM"])
        ax.set_ylim(0.84, 1.02)
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Nested-LOSO Macro F1")
    fig.suptitle("Tuned Cross-User Head-Gesture Recognition", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "nested_tuned_loso_model_comparison.png", dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    fold_means = folds.groupby(["dataset", "model", "fold_id"], as_index=False)["macro_f1"].mean()
    for ax, dataset in zip(axes, ["dataset_A", "dataset_B"]):
        subset = fold_means[fold_means["dataset"].eq(dataset)]
        values = [subset[subset["model"].eq(model)]["macro_f1"].to_numpy() for model in models]
        ax.boxplot(
            values,
            tick_labels=["RF", "CNN1D", "LSTM"],
            patch_artist=True,
            boxprops={"facecolor": "#DCECEF"},
            medianprops={"color": "#C43C39", "linewidth": 2},
        )
        rng = np.random.default_rng(42)
        for position, model_values in enumerate(values, start=1):
            ax.scatter(position + rng.normal(0, 0.045, len(model_values)), model_values, s=16, alpha=0.5, color="#174A5B")
        ax.set_ylim(0.45, 1.03)
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Outer-Subject Macro F1")
    fig.suptitle("Nested-Tuned LOSO Subject Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "nested_tuned_loso_subject_distribution.png", dpi=260)
    plt.close(fig)

    confusion_dir = figure_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ["dataset_A", "dataset_B"]:
        for model_name in models:
            paths = sorted((output_dir / "outer_runs").glob(f"{dataset}__loso__{model_name}__seed*__*.npz"))
            y_true = np.concatenate([np.load(path)["y_true"] for path in paths])
            y_pred = np.concatenate([np.load(path)["y_pred"] for path in paths])
            matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(base.LABELS)), normalize="true")
            pd.DataFrame(matrix, index=base.LABELS, columns=base.LABELS).to_csv(
                confusion_dir / f"{dataset}_{model_name}_nested_loso_confusion_matrix.csv"
            )
            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            for row in range(len(base.LABELS)):
                for column in range(len(base.LABELS)):
                    value = matrix[row, column]
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "#222222", fontsize=9)
            ax.set_xticks(range(len(base.LABELS)), base.LABELS, rotation=35, ha="right")
            ax.set_yticks(range(len(base.LABELS)), base.LABELS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{dataset.replace('_', ' ').title()} · {model_name.upper()} · Nested LOSO")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(confusion_dir / f"{dataset}_{model_name}_nested_loso_confusion_matrix.png", dpi=240)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--datasets", nargs="+", choices=["dataset_A", "dataset_B"], default=["dataset_A", "dataset_B"])
    parser.add_argument("--models", nargs="+", choices=["rf", "cnn1d", "lstm"], default=["rf", "cnn1d", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--inner-epochs", type=int, default=35)
    parser.add_argument("--outer-epochs", type=int, default=60)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--outer-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = {"dataset_A": base.load_dataset_a, "dataset_B": base.load_dataset_b}  # release edit: load only requested datasets
    bundles = {name: loaders[name]() for name in args.datasets}
    config_record = {
        "method": "nested LOSO with subject-grouped inner CV",
        "outer_protocol": "LOSO",
        "inner_splits": args.inner_splits,
        "tuning_seed": args.tuning_seed,
        "outer_seeds": args.seeds,
        "inner_epochs": args.inner_epochs,
        "outer_epochs": args.outer_epochs,
        "candidate_limit": args.candidate_limit,
        "search_spaces": {name: SEARCH_SPACES[name][: args.candidate_limit or None] for name in args.models},
        "selection_metric": "mean inner subject-grouped Macro F1",
        "outer_test_policy": "never used for selection, early stopping, or normalization fitting",
    }
    (output_dir / "tuning_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    tasks = []
    for bundle in bundles.values():
        folds = base.fold_definitions(bundle, "loso")
        if args.outer_limit:
            folds = folds[: args.outer_limit]
        for fold in folds:
            for model_name in args.models:
                tasks.append((bundle, fold, model_name))
    print(f"Nested selection tasks: {len(tasks)}", flush=True)
    for task_number, (bundle, fold, model_name) in enumerate(tasks, start=1):
        selected, scores = select_config(
            output_dir,
            bundle,
            fold,
            model_name,
            args.inner_splits,
            args.tuning_seed,
            args.inner_epochs,
            args.candidate_limit,
        )
        print(
            f"[{task_number}/{len(tasks)}] {bundle.name} outer={fold['fold_id']} {model_name}: "
            f"selected={selected['id']} inner_F1={scores.iloc[0]['mean_inner_macro_f1']:.4f}",
            flush=True,
        )
        for seed in args.seeds:
            run_outer_fold(output_dir, bundle, fold, model_name, selected, seed, args.outer_epochs)

    expected_full = not args.outer_limit and set(args.datasets) == {"dataset_A", "dataset_B"} and set(args.models) == {
        "rf",
        "cnn1d",
        "lstm",
    }
    folds, seeds, summary = collect_outer(output_dir)
    if expected_full:
        expected_rows = (8 + 15) * 3 * len(args.seeds)
        if len(folds) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} outer rows, found {len(folds)}")
        save_outputs(output_dir, folds, seeds, summary)
    print("\nTuned summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nOutputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
