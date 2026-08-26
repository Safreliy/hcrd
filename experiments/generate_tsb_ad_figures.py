"""Generate the A1 method illustration and benchmark-stratum figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.analyze_tsb_ad_area_results import _load_baselines  # noqa: E402
from hcrd import aggregate_area_density, multiscale_area_density  # noqa: E402


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def point_example(output: Path) -> None:
    # Transparent, score-independent illustration rule: the first series in the
    # official evaluation order with exactly one labelled point and <= 2,000
    # samples.  This avoids selecting the example by HCRD performance.
    filename = "551_YAHOO_id_1_Synthetic_tr_500_1st_893.csv"
    path = (
        REPOSITORY_ROOT
        / "third_party"
        / "TSB-AD-U-data"
        / "TSB-AD-U"
        / filename
    )
    frame = pd.read_csv(path)
    signal = frame["Data"].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy(dtype=int)
    density = multiscale_area_density(signal, max_levels=8)
    centre = np.median(density, axis=1, keepdims=True)
    scale = np.quantile(density, 0.9, axis=1, keepdims=True) - centre
    scale = np.where(scale > 0.0, scale, 1.0)
    surprise = np.maximum((density - centre) / scale, 0.0)
    score = aggregate_area_density(surprise, aggregation="max")
    score_rank = pd.Series(score).rank(method="average").to_numpy() / len(score)
    anomaly = np.flatnonzero(labels)

    figure, axes = plt.subplots(
        3, 1, figsize=(8.2, 5.8), sharex=True, height_ratios=[1.15, 1.35, 1.0]
    )
    time_axis = np.arange(signal.size)
    axes[0].plot(time_axis, signal, color="#30343b", linewidth=0.8)
    axes[0].scatter(
        anomaly,
        signal[anomaly],
        color="#c43c39",
        s=18,
        zorder=3,
        label="labelled point anomaly",
    )
    axes[0].axvline(
        500,
        color="#718096",
        linestyle="--",
        linewidth=0.8,
        label="end of normal prefix",
    )
    axes[0].set_ylabel("signal")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    image = axes[1].imshow(
        np.log1p(surprise),
        aspect="auto",
        origin="lower",
        extent=(0, signal.size - 1, 0.5, 8.5),
        cmap="magma",
        interpolation="nearest",
    )
    axes[1].set_ylabel("HCRD level")
    axes[1].set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
    colorbar = figure.colorbar(image, ax=axes[1], pad=0.01, aspect=22)
    colorbar.set_label("log area surprise", fontsize=8)

    axes[2].plot(time_axis, score_rank, color="#276678", linewidth=0.9)
    axes[2].scatter(anomaly, score_rank[anomaly], color="#c43c39", s=18, zorder=3)
    axes[2].set_ylabel("max-level\nscore rank")
    axes[2].set_xlabel("sample")
    axes[2].set_ylim(-0.02, 1.04)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("HCRD area spectrum localizes a point anomaly", fontsize=11)
    figure.tight_layout()
    _save(figure, output / "tsb_ad_point_example")
    plt.close(figure)


def benchmark_strata(output: Path) -> None:
    hcrd = pd.read_csv(ROOT / "results" / "tsb_ad_a1" / "evaluation_metrics.csv.gz")
    merged = hcrd.merge(_load_baselines(), on="file", validate="1:1")
    methods = {
        "HCRD (no training)": "vus_pr",
        "Matrix Profile": "MatrixProfile",
        "KShapeAD": "KShapeAD",
        "MMPAD": "MMPAD",
        "Sub-PCA": "Sub-PCA",
        "TSPulse ZS (pretrained)": "TSPulse (ZS)",
        "Time-RCD+MAFT (pretrained)": "Time-RCD+MAFT (FT)",
    }
    point = merged[merged["point_anomaly"] == 1]
    rows = []
    for label, column in methods.items():
        rows.append(
            {
                "method": label,
                "overall": merged[column].mean(),
                "point anomalies": point[column].mean(),
            }
        )
    table = pd.DataFrame(rows)
    y = np.arange(len(table))
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    colours = ["#1c7187" if name.startswith("HCRD") else "#aab4be" for name in table.method]
    for axis, column, title in zip(
        axes,
        ["overall", "point anomalies"],
        ["All 350 evaluation series", "49 point-anomaly series"],
        strict=True,
    ):
        axis.barh(y, table[column], color=colours, height=0.68)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("mean VUS-PR")
        axis.set_xlim(0.0, 0.9)
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        for index, value in enumerate(table[column]):
            axis.text(value + 0.012, index, f"{value:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y, table["method"], fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle("HCRD is specialized for point anomalies, not universal TSAD", fontsize=11)
    figure.tight_layout()
    _save(figure, output / "tsb_ad_strata")
    plt.close(figure)


def main() -> None:
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    point_example(output)
    benchmark_strata(output)
    print(output / "tsb_ad_point_example.png")
    print(output / "tsb_ad_strata.png")


if __name__ == "__main__":
    main()
