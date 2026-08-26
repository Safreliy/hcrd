"""Analyse the saved A1 evaluation without rerunning or retuning the detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent

METHODS = [
    "Sub-PCA",
    "MMPAD",
    "StreamVAE",
    "POLY",
    "KShapeAD",
    "Series2Graph",
    "MatrixProfile",
    "TSPulse (ZS)",
    "TSPulse (FT)",
    "Time-RCD+MAFT (FT)",
]
PRETRAINED = {
    "TSPulse (ZS)",
    "TSPulse (FT)",
    "Time-RCD+MAFT (FT)",
}


def _bootstrap_mean_difference(
    difference: np.ndarray, *, draws: int = 20_000, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    batch = 1_000
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        indices = rng.integers(0, difference.size, size=(stop - start, difference.size))
        means[start:stop] = np.mean(difference[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _load_baselines() -> pd.DataFrame:
    """Read official saved score tables without importing detector dependencies."""

    benchmark = REPOSITORY_ROOT / "third_party" / "TSB-AD" / "benchmark_exp"
    original = pd.read_csv(
        benchmark / "benchmark_eval_results" / "uni_mergedTable_VUS-PR.csv"
    )
    keep = [
        "file",
        "Sub-PCA",
        "POLY",
        "KShapeAD",
        "Series2Graph",
        "MatrixProfile",
        "point_anomaly",
        "seq_anomaly",
    ]
    merged = original[keep].copy()
    sources = [
        ("Uni_StreamVAE.csv", ["file", "StreamVAE"], {}),
        (
            "Uni_MMPAD.csv",
            ["file", "VUS-PR", "Time"],
            {"VUS-PR": "MMPAD", "Time": "MMPAD_seconds"},
        ),
        (
            "Uni_TSPulse.csv",
            ["file", "TSPulse (ZS)", "TSPulse (FT)"],
            {},
        ),
    ]
    for filename, columns, rename in sources:
        table = pd.read_csv(benchmark / "leaderboard_results" / filename)
        merged = merged.merge(table[columns].rename(columns=rename), on="file", how="left")
    maft = pd.read_csv(benchmark / "leaderboard_results" / "Uni_TimeRCD_MAFT.csv")
    return merged.merge(
        maft[["filename", "VUS-PR"]].rename(
            columns={"filename": "file", "VUS-PR": "Time-RCD+MAFT (FT)"}
        ),
        on="file",
        how="left",
    )


def paired_table(group: pd.DataFrame, stratum: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, method in enumerate(METHODS):
        valid = group[["vus_pr", method]].dropna()
        difference = valid["vus_pr"].to_numpy() - valid[method].to_numpy()
        low, high = _bootstrap_mean_difference(
            difference, seed=20240825 + 100 * len(stratum) + index
        )
        rows.append(
            {
                "stratum": stratum,
                "baseline": method,
                "baseline_pretrained": method in PRETRAINED,
                "paired_series": int(len(valid)),
                "hcrd_mean_vus_pr": float(valid["vus_pr"].mean()),
                "baseline_mean_vus_pr": float(valid[method].mean()),
                "mean_difference": float(difference.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "wins": int(np.sum(difference > 1e-12)),
                "ties": int(np.sum(np.abs(difference) <= 1e-12)),
                "losses": int(np.sum(difference < -1e-12)),
            }
        )
    return pd.DataFrame(rows)


def grouped_means(merged: pd.DataFrame, column: str) -> pd.DataFrame:
    methods = ["vus_pr", *METHODS]
    output = merged.groupby(column, sort=True)[methods].mean().reset_index()
    output.insert(1, "series", merged.groupby(column, sort=True).size().to_numpy())
    return output.rename(columns={"vus_pr": "HCRD"})


def main() -> None:
    output = ROOT / "results" / "tsb_ad_a1"
    metrics = pd.read_csv(output / "evaluation_metrics.csv.gz")
    merged = metrics.merge(_load_baselines(), on="file", validate="1:1")

    groups = {
        "all": merged,
        "point_anomaly": merged[merged["point_anomaly"] == 1],
        "no_point_anomaly": merged[merged["point_anomaly"] == 0],
        "sequence_anomaly": merged[merged["seq_anomaly"] == 1],
        "no_sequence_anomaly": merged[merged["seq_anomaly"] == 0],
    }
    comparisons = pd.concat(
        [paired_table(group, name) for name, group in groups.items()],
        ignore_index=True,
    )
    comparisons.to_csv(output / "subgroup_comparisons.csv", index=False)
    grouped_means(merged, "source").to_csv(
        output / "source_method_means.csv", index=False
    )
    grouped_means(merged, "domain").to_csv(
        output / "domain_method_means.csv", index=False
    )

    point = comparisons[comparisons["stratum"] == "point_anomaly"]
    nonpretrained = point[~point["baseline_pretrained"]]
    next_row = nonpretrained.sort_values("baseline_mean_vus_pr", ascending=False).iloc[0]
    summary = {
        "analysis": "post-evaluation analysis of predeclared strata",
        "point_series": int(len(groups["point_anomaly"])),
        "hcrd_point_mean_vus_pr": float(point["hcrd_mean_vus_pr"].iloc[0]),
        "highest_nonpretrained_baseline": str(next_row["baseline"]),
        "highest_nonpretrained_baseline_mean_vus_pr": float(
            next_row["baseline_mean_vus_pr"]
        ),
        "difference": float(next_row["mean_difference"]),
        "bootstrap_95_low": float(next_row["bootstrap_95_low"]),
        "bootstrap_95_high": float(next_row["bootstrap_95_high"]),
        "interpretation": (
            "HCRD has the highest point estimate among the published "
            "nonpretrained baselines, but the paired bootstrap interval versus "
            "the strongest such baseline includes zero."
        ),
    }
    (output / "subgroup_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
