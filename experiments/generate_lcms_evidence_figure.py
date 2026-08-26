#!/usr/bin/env python3
"""Generate the manuscript's independent LC--MS evidence figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    e1 = json.loads(
        (ROOT / "results/lcms_eic_e1/confirmation/confirmation_results.json").read_text(
            encoding="utf-8"
        )
    )
    e2 = json.loads(
        (ROOT / "results/ms_metrics_e2/evaluation/e2_results.json").read_text(
            encoding="utf-8"
        )
    )
    cohorts = ("E1 double holdout", "Falkor → Meso", "Meso → Falkor")
    methods = (
        "minimal",
        "domain",
        "hcrd_1",
        "hcrd_8",
        "geometry",
        "area",
    )
    labels = (
        "raw / qscore",
        "domain",
        "HCRD-1",
        "HCRD-8",
        "geometry",
        "area/energy",
    )
    colours = ("0.72", "0.38", "#8CC7E8", "#0072B2", "#009E73", "#E69F00")
    e1_map = {
        "minimal": "raw64",
        "domain": "domain",
        "hcrd_1": "hcrd_1",
        "hcrd_8": "hcrd_8",
        "geometry": "hcrd_geometry",
        "area": "area_only",
    }
    e2_map = {
        "minimal": "qscore",
        "domain": "domain_q",
        "hcrd_1": "hcrd_1_q",
        "hcrd_8": "hcrd_8_q",
        "geometry": "hcrd_geometry_q",
        "area": "area_only_q",
    }
    directions = ("falkor_to_mesoscope", "mesoscope_to_falkor")
    values = np.empty((len(methods), 3))
    for row, method in enumerate(methods):
        values[row, 0] = e1["metrics"][e1_map[method]]["average_precision"]
        for column, direction in enumerate(directions, start=1):
            values[row, column] = e2["directions"][direction]["metrics"][e2_map[method]][
                "average_precision"
            ]

    figure, (bars, forest) = plt.subplots(
        1, 2, figsize=(9.4, 3.45), gridspec_kw={"width_ratios": (1.75, 1.0)}
    )
    centers = np.arange(3)
    width = 0.125
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for row, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        bars.bar(
            centers + offsets[row],
            values[row],
            width=width * 0.92,
            color=colour,
            label=label,
            edgecolor="white",
            linewidth=0.35,
        )
    bars.set_xticks(centers, cohorts)
    bars.set_ylim(0.25, 1.0)
    bars.set_ylabel("average precision")
    bars.set_title("a  independent expert-labelled evaluations", loc="left", fontweight="bold")
    bars.grid(axis="y", alpha=0.18)
    handles, legend_labels = bars.get_legend_handles_labels()

    e1_comparison = e1["ap_comparisons"]["hcrd_8"]
    comparisons = [
        (
            e1_comparison["ap_difference"],
            e1_comparison["cluster_bootstrap_95_ci"],
        )
    ]
    for direction in directions:
        item = e2["directions"][direction]["comparisons"]["hcrd_8_q"]
        comparisons.append((item["ap_difference"], item["bootstrap_95_ci"]))
    point = np.asarray([item[0] for item in comparisons])
    lower = np.asarray([item[1][0] for item in comparisons])
    upper = np.asarray([item[1][1] for item in comparisons])
    y = np.arange(3)
    forest.axvline(0.0, color="0.35", lw=0.9, ls="--")
    forest.errorbar(
        point,
        y,
        xerr=np.vstack([point - lower, upper - point]),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        capsize=3,
        lw=1.4,
    )
    forest.set_yticks(y, ("E1", "E2  F → M", "E2  M → F"))
    forest.invert_yaxis()
    forest.set_xlim(-0.08, 0.19)
    forest.set_xlabel("AP difference: HCRD-8 $-$ primary comparator")
    forest.set_title("b  paired 95% intervals", loc="left", fontweight="bold")
    forest.grid(axis="x", alpha=0.18)
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=7.2,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.34, 0.005),
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.88, wspace=0.35)
    output = ROOT / "paper/figures/lcms_evidence"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
