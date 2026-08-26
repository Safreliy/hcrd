"""Generate the record-level QTDB comparison figure for the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results" / "qtdb_confirmation_r2"
OUTPUT = PROJECT / "paper" / "figures"


def main() -> None:
    comparisons = json.loads((RESULTS / "comparisons.json").read_text())
    labels = [
        row["comparison"].replace("hcrd_quadratic - ", "")
        for row in comparisons
    ]
    labels = [
        {
            "hcrd_gaussian": "Gaussian HCRD",
            "hcrd_raw": "raw HCRD",
            "derivative_threshold": "derivative",
            "official_pu0": "ecgpuwave ch. 0",
            "official_pu1": "ecgpuwave ch. 1",
        }[label]
        for label in labels
    ]
    means = np.array([row["mean_record_difference_ms"] for row in comparisons])
    lower = np.array([row["bootstrap_ci_lower_ms"] for row in comparisons])
    upper = np.array([row["bootstrap_ci_upper_ms"] for row in comparisons])
    y = np.arange(len(labels))
    colors = ["#0072B2" if value < 0 else "#D55E00" for value in means]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="none",
        ecolor="#555555",
        elinewidth=2,
        capsize=4,
    )
    axis.scatter(means, y, c=colors, s=45, zorder=3)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("record-level joint MAE difference (ms): quadratic HCRD minus comparator")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "qtdb_record_differences.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "qtdb_record_differences.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
