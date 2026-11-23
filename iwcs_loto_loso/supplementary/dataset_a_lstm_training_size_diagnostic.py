#!/usr/bin/env python3
"""Diagnose why Dataset A LSTM scores higher under LOSO than LOTO.

The original protocols confound evaluation target and training-set size:
LOTO trains on 120 windows from one subject, whereas LOSO trains on 1,260
windows from seven subjects. This script keeps the outer LOSO test subject
fixed and varies only the number of training windows using subject/class
balanced sampling. Existing full-LOSO and LOTO predictions are reused.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "04_results" / "dataset_A_lstm_training_size_diagnostic"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # release edit
import unified_loto_loso_analysis as base


SEEDS = [42, 43, 44]
SUBSET_SIZES = [120, 360, 720]
FULL_SIZE = 1260
EPOCHS = 60


def balanced_subject_class_sample(
    bundle: base.DatasetBundle,
    candidate_indices: np.ndarray,
    total_size: int,
    seed: int,
) -> np.ndarray:
    """Select an exactly class-balanced subset spread across train subjects."""
    if total_size % len(base.LABELS):
        raise ValueError("total_size must be divisible by the number of classes")
    per_class = total_size // len(base.LABELS)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    train_subjects = sorted(set(bundle.subjects[candidate_indices].tolist()))

    for label in range(len(base.LABELS)):
        base_quota, remainder = divmod(per_class, len(train_subjects))
        subject_order = list(train_subjects)
        rng.shuffle(subject_order)
        extra_subjects = set(subject_order[:remainder])
        for subject in train_subjects:
            quota = base_quota + int(subject in extra_subjects)
            pool = candidate_indices[
                (bundle.labels[candidate_indices] == label)
                & (bundle.subjects[candidate_indices] == subject)
            ]
            if quota > len(pool):
                raise ValueError(
                    f"Not enough samples for label={label}, subject={subject}: "
                    f"requested {quota}, available {len(pool)}"
                )
            selected.extend(rng.choice(pool, size=quota, replace=False).tolist())

    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    if len(selected_array) != total_size:
        raise AssertionError((len(selected_array), total_size))
    class_counts = np.bincount(bundle.labels[selected_array], minlength=len(base.LABELS))
    if not np.all(class_counts == per_class):
        raise AssertionError(class_counts)
    return selected_array


def run_subsampled_loso(bundle: base.DatasetBundle) -> None:
    run_dir = OUTPUT / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    folds = base.fold_definitions(bundle, "loso")
    for fold in folds:
        heldout = str(fold["heldout_subject"])
        full_train = np.asarray(fold["train"], dtype=np.int64)
        test_indices = np.asarray(fold["test"], dtype=np.int64)
        for size in SUBSET_SIZES:
            for seed in SEEDS:
                output = run_dir / f"loso_size{size}__seed{seed}__{heldout}.npz"
                if output.exists():
                    continue
                sample_seed = seed * 1000 + int(heldout.removeprefix("p"))
                train_indices = balanced_subject_class_sample(
                    bundle, full_train, size, sample_seed
                )
                prediction = base.predict_model(
                    "lstm",
                    bundle,
                    train_indices,
                    test_indices,
                    seed,
                    EPOCHS,
                )
                np.savez_compressed(
                    output,
                    y_true=bundle.labels[test_indices],
                    y_pred=prediction,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    heldout_subject=np.asarray([heldout]),
                    train_size=np.asarray([size]),
                    seed=np.asarray([seed]),
                )
                print(f"[done] LOSO train={size} seed={seed} heldout={heldout}", flush=True)


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def load_subsampled() -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, list[np.ndarray]]]]:
    fold_rows: list[dict[str, object]] = []
    pooled: dict[tuple[int, int], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"true": [], "pred": []}
    )
    for path in sorted((OUTPUT / "runs").glob("loso_size*.npz")):
        data = np.load(path)
        y_true, y_pred = data["y_true"], data["y_pred"]
        size, seed = int(data["train_size"][0]), int(data["seed"][0])
        heldout = str(data["heldout_subject"][0])
        fold_rows.append(
            {
                "protocol": "LOSO",
                "train_size": size,
                "seed": seed,
                "heldout_subject": heldout,
                **metric(y_true, y_pred),
            }
        )
        pooled[(size, seed)]["true"].append(y_true)
        pooled[(size, seed)]["pred"].append(y_pred)
    return fold_rows, pooled


def load_existing_predictions(
    protocol: str,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, list[np.ndarray]]]]:
    source = ROOT / "04_results" / "final_loto_loso" / "runs"
    pattern = f"dataset_A__{protocol.lower()}__lstm__seed*.npz"
    train_size = 120 if protocol == "LOTO" else FULL_SIZE
    fold_rows: list[dict[str, object]] = []
    pooled: dict[tuple[int, int], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"true": [], "pred": []}
    )
    for path in sorted(source.glob(pattern)):
        parts = path.stem.split("__")
        seed = int(parts[3].removeprefix("seed"))
        if seed not in SEEDS:
            continue
        data = np.load(path)
        y_true, y_pred = data["y_true"], data["y_pred"]
        heldout = str(data["heldout_subject"][0])
        fold_rows.append(
            {
                "protocol": protocol,
                "train_size": train_size,
                "seed": seed,
                "heldout_subject": heldout,
                "fold_id": str(data["fold_id"][0]),
                **metric(y_true, y_pred),
            }
        )
        pooled[(train_size, seed)]["true"].append(y_true)
        pooled[(train_size, seed)]["pred"].append(y_pred)
    return fold_rows, pooled


def aggregate_pooled(
    protocol: str,
    pooled: dict[tuple[int, int], dict[str, list[np.ndarray]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (size, seed), values in sorted(pooled.items()):
        y_true = np.concatenate(values["true"])
        y_pred = np.concatenate(values["pred"])
        rows.append(
            {
                "protocol": protocol,
                "train_size": size,
                "seed": seed,
                "n_predictions": int(len(y_true)),
                **metric(y_true, y_pred),
            }
        )
    return rows


def bootstrap_full_minus_120(
    fold_rows: list[dict[str, object]],
    iterations: int = 20000,
) -> dict[str, object]:
    by_size_subject: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in fold_rows:
        if row["protocol"] == "LOSO":
            by_size_subject[(int(row["train_size"]), str(row["heldout_subject"]))].append(
                float(row["macro_f1"])
            )
    subjects = sorted(
        {
            subject
            for size, subject in by_size_subject
            if size == 120 and (FULL_SIZE, subject) in by_size_subject
        }
    )
    small = np.asarray([np.mean(by_size_subject[(120, subject)]) for subject in subjects])
    full = np.asarray([np.mean(by_size_subject[(FULL_SIZE, subject)]) for subject in subjects])
    differences = full - small
    rng = np.random.default_rng(20260811)
    draws = rng.integers(0, len(subjects), size=(iterations, len(subjects)))
    bootstrap = differences[draws].mean(axis=1)
    return {
        "comparison": "LOSO full 1260 minus size-matched LOSO 120",
        "subjects": subjects,
        "mean_difference": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "subjects_improved": int((differences > 0).sum()),
        "n_subjects": len(subjects),
    }


def summarize(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["protocol"]), int(row["train_size"]))].append(row)
    summary = []
    for (protocol, size), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        f1 = np.asarray([float(row["macro_f1"]) for row in rows])
        accuracy = np.asarray([float(row["accuracy"]) for row in rows])
        summary.append(
            {
                "protocol": protocol,
                "train_size": size,
                "seeds": len(rows),
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=1)),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std(ddof=1)),
            }
        )
    return summary


def save_plot(summary: list[dict[str, object]]) -> None:
    loso = sorted(
        (row for row in summary if row["protocol"] == "LOSO"),
        key=lambda row: int(row["train_size"]),
    )
    loto = next(row for row in summary if row["protocol"] == "LOTO")
    sizes = np.asarray([int(row["train_size"]) for row in loso])
    means = np.asarray([float(row["macro_f1_mean"]) for row in loso])
    stds = np.asarray([float(row["macro_f1_std"]) for row in loso])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.errorbar(
        sizes,
        means,
        yerr=stds,
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#D95555",
        label="LOSO (unseen subject)",
    )
    ax.axhline(
        float(loto["macro_f1_mean"]),
        color="#177E89",
        linestyle="--",
        linewidth=2,
        label=f"LOTO, train=120 ({float(loto['macro_f1_mean']):.3f})",
    )
    for index, (x, y) in enumerate(zip(sizes, means)):
        offset = -24 if index == 0 else 9
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlabel("Training windows per fold")
    ax.set_ylabel("Macro F1")
    ax.set_title("Dataset A LSTM: Training-Size Diagnostic")
    ax.set_ylim(0.65, 1.01)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "dataset_A_lstm_training_size_curve.png", dpi=260)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundle = base.load_dataset_a()
    run_subsampled_loso(bundle)

    subset_folds, subset_pooled = load_subsampled()
    loto_folds, loto_pooled = load_existing_predictions("LOTO")
    full_folds, full_pooled = load_existing_predictions("LOSO")
    all_fold_rows = subset_folds + loto_folds + full_folds

    seed_rows = (
        aggregate_pooled("LOSO", subset_pooled)
        + aggregate_pooled("LOTO", loto_pooled)
        + aggregate_pooled("LOSO", full_pooled)
    )
    summary = summarize(seed_rows)
    bootstrap = bootstrap_full_minus_120(all_fold_rows)
    payload = {
        "question": "Why is Dataset A LSTM higher under LOSO than LOTO?",
        "design": {
            "LOTO_original_train_size": 120,
            "LOSO_original_train_size": FULL_SIZE,
            "LOSO_control_sizes": SUBSET_SIZES + [FULL_SIZE],
            "sampling": "exactly class-balanced and approximately subject-balanced",
            "outer_test": "one completely held-out subject for every LOSO condition",
            "seeds": SEEDS,
            "epochs_max": EPOCHS,
        },
        "summary": summary,
        "bootstrap": bootstrap,
        "seed_metrics": seed_rows,
        "fold_metrics": all_fold_rows,
    }
    (OUTPUT / "diagnostic_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plot(summary)
    print(json.dumps({"summary": summary, "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
