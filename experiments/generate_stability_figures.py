"""Generate persistence and parallel-scaling figures for the manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def persistence_figure() -> None:
    source = PROJECT / "results" / "persistence_stability_t1"
    with (source / "discontinuity.csv").open(encoding="utf-8") as handle:
        discontinuity = list(csv.DictReader(handle))
    with (source / "random_trials.csv").open(encoding="utf-8") as handle:
        trials = list(csv.DictReader(handle))
    epsilon = np.asarray([float(row["epsilon"]) for row in discontinuity])
    hard = np.asarray([float(row["hard_amplification"]) for row in discontinuity])
    stable = np.asarray(
        [float(row["persistence_bound_ratio"]) for row in discontinuity]
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].loglog(epsilon, hard, "o-", label="hard baseline amplification")
    axes[0].semilogx(epsilon, stable, "s-", label="persistence / theorem bound")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"one-sided curvature scale $\varepsilon$")
    axes[0].set_ylabel("dimensionless ratio")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for grid, color in (("uniform", "#3465a4"), ("irregular", "#cc0000")):
        ratios = [
            float(row["ratio"]) for row in trials if row["grid"] == grid
        ]
        axes[1].hist(
            ratios,
            bins=np.linspace(0.0, 1.0, 31),
            alpha=0.55,
            label=grid,
            color=color,
        )
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"observed $d_{\rm SC}/(C_x\|e\|_\infty)$")
    axes[1].set_ylabel("trials")
    axes[1].grid(alpha=0.20)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(
            PROJECT / "paper" / "figures" / f"persistence_stability.{extension}",
            dpi=220,
        )
    plt.close(figure)


def parallel_figure() -> None:
    source = PROJECT / "results" / "parallel_runtime_p1" / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))["summary"]
    labels = [
        "serial" if row["backend"] == "serial" else f"{row['backend']}\n{row['workers']}"
        for row in summary
    ]
    speedups = np.asarray([float(row["speedup_vs_serial"]) for row in summary])
    colours = [
        "#555555" if row["backend"] == "serial" else
        "#c44e52" if row["backend"] == "thread" else "#4c72b0"
        for row in summary
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    bars = axis.bar(labels, speedups, color=colours)
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_ylabel("speedup versus serial median")
    axis.set_ylim(0.0, max(2.8, 1.15 * float(np.max(speedups))))
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, speedups, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(
            PROJECT / "paper" / "figures" / f"parallel_scaling.{extension}",
            dpi=220,
        )
    plt.close(figure)


if __name__ == "__main__":
    persistence_figure()
    parallel_figure()
