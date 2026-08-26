"""Generate the C1 held-out Yahoo comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    comparisons = pd.read_csv(
        ROOT / "results" / "tsb_uad_yahoo_c1" / "point_comparisons.csv"
    ).set_index("baseline")
    methods = ["CNN", "HCRD L8-max", "LSTM", "NORMA", "IFOREST1", "POLY", "MP"]
    values = [
        comparisons.loc["CNN", "baseline_mean_auc_pr"],
        comparisons.loc["NORMA", "hcrd_mean_auc_pr"],
        comparisons.loc["LSTM", "baseline_mean_auc_pr"],
        comparisons.loc["NORMA", "baseline_mean_auc_pr"],
        comparisons.loc["IFOREST1", "baseline_mean_auc_pr"],
        comparisons.loc["POLY", "baseline_mean_auc_pr"],
        comparisons.loc["MP", "baseline_mean_auc_pr"],
    ]
    colours = ["#aab4be", "#1c7187", "#aab4be", "#d28e3d", "#aab4be", "#aab4be", "#aab4be"]
    y = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(7.8, 3.7))
    axis.barh(y, values, color=colours, height=0.66)
    axis.set_yticks(y, methods)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("mean per-series AUC-PR")
    axis.set_title("Frozen HCRD confirmation on 134 additional Yahoo point anomalies")
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    for index, value in enumerate(values):
        axis.text(value + 0.012, index, f"{value:.3f}", va="center", fontsize=8)
    axis.text(
        0.99,
        0.02,
        "HCRD - NORMA: +0.190, paired 95% bootstrap [0.129, 0.251]",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        fontsize=8,
        color="#334155",
    )
    figure.tight_layout()
    output = ROOT / "figures" / "tsb_uad_yahoo_confirmation"
    for suffix, options in [
        (".png", {"dpi": 220}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        figure.savefig(output.with_suffix(suffix), bbox_inches="tight", **options)
    plt.close(figure)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
