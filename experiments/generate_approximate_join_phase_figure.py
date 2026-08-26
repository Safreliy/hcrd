"""Render the approximate-join departure phase diagram."""

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
        default=PROJECT / "results" / "approximate_join_phase_r1" / "aggregate.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "paper" / "figures" / "approximate_join_phase",
    )
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    joins = sorted({float(row["join_ratio"]) for row in rows})
    curvatures = sorted({float(row["curvature_ratio"]) for row in rows})
    configurations = sorted({row["configuration"] for row in rows})

    def pooled(column: str) -> np.ndarray:
        output = np.empty((len(curvatures), len(joins)), dtype=float)
        for row_index, curvature in enumerate(curvatures):
            for column_index, join in enumerate(joins):
                values = [
                    float(row[column])
                    for row in rows
                    if float(row["join_ratio"]) == join
                    and float(row["curvature_ratio"]) == curvature
                ]
                if len(values) != len(configurations):
                    raise RuntimeError("incomplete phase grid")
                output[row_index, column_index] = float(np.mean(values))
        return output

    aware = pooled("aware_exact_probability")
    naive = pooled("naive_exact_probability")
    plt.rcParams.update({"font.size": 8.4, "axes.titlesize": 9.2})
    figure, axes = plt.subplots(
        1, 2, figsize=(7.45, 3.05), sharex=True, sharey=True, layout="constrained"
    )
    images = []
    for axis, values, title in zip(
        axes,
        (aware, naive),
        ("(a) Join-aware tolerance $\\eta+\\tau$", "(b) Noise-only tolerance $\\tau$"),
        strict=True,
    ):
        image = axis.pcolormesh(
            joins,
            curvatures,
            values,
            shading="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        images.append(image)
        line_x = np.linspace(min(joins), max(joins), 200)
        axis.plot(line_x, line_x + 2.0, "w--", linewidth=1.5)
        axis.text(
            0.03,
            0.97,
            r"certified: $\gamma/\tau>\eta/\tau+2$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        axis.set_title(title, loc="left", pad=5)
        axis.set_xlabel(r"join magnitude $\eta/\tau$")
        axis.set_xticks(joins)
        axis.set_yticks(curvatures)
    axes[0].set_ylabel(r"minimum active curvature $\gamma/\tau$")
    colorbar = figure.colorbar(images[0], ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("exact-recovery probability")
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
