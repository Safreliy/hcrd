"""Render the theorem-linked recovery phase diagram."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "results" / "recovery_phase_r1" / "aggregate.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "paper" / "figures" / "recovery_phase_diagram",
    )
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    configurations = sorted(
        {row["configuration"] for row in rows},
        key=lambda value: int(value.split(",", maxsplit=1)[0].split("=")[1]),
    )
    ratios = sorted({float(row["curvature_ratio"]) for row in rows})
    by_key = {
        (row["configuration"], float(row["curvature_ratio"])): row for row in rows
    }
    exact = np.array(
        [
            [float(by_key[(configuration, ratio)]["exact_recovery_probability"]) for ratio in ratios]
            for configuration in configurations
        ]
    )
    localisation = np.array(
        [
            [float(by_key[(configuration, ratio)]["mean_localisation_error_widths"]) for ratio in ratios]
            for configuration in configurations
        ]
    )

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9})
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 2.75), constrained_layout=True)
    probability_image = axes[0].imshow(
        exact, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    error_limit = max(0.25, float(np.quantile(localisation, 0.95)))
    error_image = axes[1].imshow(
        localisation,
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=error_limit,
        cmap="magma_r",
    )
    theorem_edge = next(
        (index - 0.5 for index, ratio in enumerate(ratios) if ratio > 2.0),
        len(ratios) - 0.5,
    )
    short_labels = [value.replace(",", ", ") for value in configurations]
    for axis, title in zip(
        axes,
        ("(a) Exact first-level knot recovery", "(b) Mean Hausdorff error / lobe width"),
        strict=True,
    ):
        axis.set_title(title, loc="left", pad=5)
        axis.set_xticks(range(len(ratios)), [f"{ratio:g}" for ratio in ratios], rotation=45)
        axis.set_yticks(range(len(configurations)), short_labels)
        axis.set_xlabel(r"normalized curvature $\rho=\gamma/\tau$")
        axis.axvline(theorem_edge, color="white", linewidth=1.4, linestyle="--")
        axis.text(
            0.985,
            0.035,
            r"certified $\rho>2$",
            transform=axis.transAxes,
            color="black",
            fontsize=7.2,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    axes[1].tick_params(labelleft=False)
    figure.colorbar(probability_image, ax=axes[0], fraction=0.05, pad=0.025, label="probability")
    figure.colorbar(error_image, ax=axes[1], fraction=0.05, pad=0.025, label="widths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 240}), (".svg", {})):
        output = args.output.with_suffix(suffix)
        figure.savefig(output, bbox_inches="tight", **options)
        if suffix == ".svg":
            svg = "\n".join(
                line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()
            )
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(svg + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
