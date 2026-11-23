#!/usr/bin/env python3
"""Create the Dataset A 150-frame LOTO/LOSO poster result figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_results" / "dataset_A_loto_loso_150" / "results" / "summary_metrics.csv"
OUTPUT = ROOT / "05_poster" / "IWCS_dataset_A_only" / "figures"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    frame = frame[frame["dataset"].eq("dataset_A")]
    models = ["rf", "cnn1d", "lstm"]
    labels = ["Random Forest", "CNN1D", "LSTM"]
    colors = {"loto": "#177E89", "loso": "#E16A54"}

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(models))
    width = 0.34
    for offset, protocol in [(-width / 2, "loto"), (width / 2, "loso")]:
        values, errors = [], []
        for model in models:
            row = frame[(frame["protocol"].eq(protocol)) & (frame["model"].eq(model))].iloc[0]
            values.append(float(row["macro_f1_mean"]))
            errors.append(float(row["macro_f1_std"]))
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=4,
            label=protocol.upper(),
            color=colors[protocol],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5, fontsize=11, fontweight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.80, 1.01)
    fig.suptitle("HMD Head-Gesture Recognition under LOTO and LOSO", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Macro F1, mean ± SD across three fixed random seeds",
        ha="center",
        va="center",
        fontsize=10,
        color="#536A70",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUTPUT / "fig01_model_performance_loto_loso_150.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "fig01_model_performance_loto_loso_150.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
