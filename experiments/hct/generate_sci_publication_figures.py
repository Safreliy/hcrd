"""Generate publication-labelled SCI figures from frozen E33/E35 outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "paper/sci/figures"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def e33_frontier_figure() -> None:
    source = (
        PROJECT
        / "results/hct/shape_contrast_hybrid_e33_confirmation/summary.csv"
    )
    rows = [row for row in _rows(source) if row["signal"].startswith("paper_")]
    names = {
        "paper_f1_cusp": "cusp",
        "paper_f2_onset": "onset",
        "paper_f3_jump": "jump",
        "paper_f4_logistic": "logistic",
    }
    labels = [
        f"{names[row['signal']]}\n{row['design'].replace('_4_8', '')}, {row['n']}"
        for row in rows
    ]
    positions = np.arange(len(rows))
    bar_width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.4), constrained_layout=True)
    axes[0].bar(
        positions - bar_width / 2,
        [float(row["hct_coverage"]) for row in rows],
        bar_width,
        label="SCI (finite-sample)",
    )
    axes[0].bar(
        positions + bar_width / 2,
        [float(row["shapechange_coverage"]) for row in rows],
        bar_width,
        label="ShapeChange bootstrap",
    )
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(ncol=2, loc="lower right")
    axes[0].set_title("Frozen 95% inflection-set comparison (200 trials per cell)")

    axes[1].bar(
        positions - bar_width / 2,
        [float(row["hct_median_width"]) for row in rows],
        bar_width,
        label="SCI",
    )
    axes[1].bar(
        positions + bar_width / 2,
        [float(row["shapechange_median_width"]) for row in rows],
        bar_width,
        label="ShapeChange",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median interval width (log scale)")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=42, ha="right", fontsize=8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "e33_frontier_coverage_width.png", dpi=220)
    figure.savefig(OUTPUT / "e33_frontier_coverage_width.pdf")
    plt.close(figure)


def lidar_figure() -> None:
    result_dir = (
        PROJECT / "results/hct/heteroskedastic_shape_inversion_e35_confirmation"
    )
    data = np.genfromtxt(
        result_dir / "lidar_data_and_fit.csv", delimiter=",", names=True
    )
    sensitivity = _rows(result_dir / "lidar_hct_sensitivity.csv")
    external = _rows(result_dir / "lidar_external_fits.csv")[0]
    point = float(external["sshaped_point_metres"])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.4),
        gridspec_kw={"height_ratios": [2.1, 1.0]},
        constrained_layout=True,
    )
    axes[0].scatter(
        data["range"], data["minus_logratio"], s=12, alpha=0.55, label="observations"
    )
    axes[0].plot(
        data["range"], data["sshaped_fitted"], color="black", linewidth=2, label="S-shaped LSE"
    )
    axes[0].axvline(point, color="tab:orange", linestyle="--")
    axes[0].set(xlabel="range (m)", ylabel="-logratio")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Mercury-plume centre: point estimate and uncertainty sensitivity")

    for index, row in enumerate(sensitivity):
        axes[1].plot(
            [float(row["left_metres"]), float(row["right_metres"])],
            [index, index],
            linewidth=7,
            solid_capstyle="butt",
        )
        axes[1].plot(point, index, marker="|", color="black", markersize=13)
    axes[1].set_yticks(
        np.arange(len(sensitivity)),
        [f"kappa={float(row['kappa']):g}" for row in sensitivity],
    )
    axes[1].set_xlabel("SCI confidence set for plume centre (m)")
    axes[1].set_xlim(float(data["range"].min()), float(data["range"].max()))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "lidar_sci_sensitivity.png", dpi=220)
    figure.savefig(OUTPUT / "lidar_sci_sensitivity.pdf")
    plt.close(figure)


if __name__ == "__main__":
    e33_frontier_figure()
    lidar_figure()
    print(OUTPUT)

