"""Exploratory monotonicity analysis for XJTU-SY HCRD health indicators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


KEYS = ["condition", "bearing_id", "filename", "file_idx", "total_files"]


def _feature_family(name: str) -> str:
    if "log1p_polygon_area" in name:
        return "hcrd_exact_polygon_mass"
    if name.startswith(("h_env_", "h_spec_", "v_env_", "v_spec_")):
        return "hcrd_other"
    return "standard"


def correlations_by_bearing(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, bearing), group in frame.groupby(["condition", "bearing_id"]):
        progress = group["file_idx"].to_numpy(dtype=float) / np.maximum(
            group["total_files"].to_numpy(dtype=float) - 1.0, 1.0
        )
        for feature in features:
            values = group[feature].to_numpy(dtype=float)
            rho = float(spearmanr(progress, values).statistic)
            rows.append(
                {
                    "condition": condition,
                    "bearing_id": bearing,
                    "feature": feature,
                    "family": _feature_family(feature),
                    "spearman_progress": rho,
                    "n_files": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def summarize(correlations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, feature), group in correlations.groupby(["family", "feature"]):
        rho = group["spearman_progress"].to_numpy(dtype=float)
        finite = rho[np.isfinite(rho)]
        positive = float(np.mean(finite > 0))
        negative = float(np.mean(finite < 0))
        rows.append(
            {
                "family": family,
                "feature": feature,
                "n_bearings": int(len(finite)),
                "median_rho": float(np.median(finite)),
                "median_abs_rho": float(np.median(np.abs(finite))),
                "q25_rho": float(np.quantile(finite, 0.25)),
                "q75_rho": float(np.quantile(finite, 0.75)),
                "min_rho": float(np.min(finite)),
                "max_rho": float(np.max(finite)),
                "positive_fraction": positive,
                "same_direction_fraction": max(positive, negative),
            }
        )
    summary = pd.DataFrame(rows)
    summary["exploratory_consistency_score"] = (
        summary["median_abs_rho"] * summary["same_direction_fraction"]
    )
    return summary.sort_values(
        ["exploratory_consistency_score", "median_abs_rho"], ascending=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-features", type=Path, required=True)
    parser.add_argument("--hcrd-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    standard = pd.read_csv(args.standard_features)
    hcrd = pd.read_csv(args.hcrd_features)
    if not standard[KEYS].equals(hcrd[KEYS]):
        raise ValueError("Standard and HCRD feature rows are not aligned")

    standard_features = [
        column
        for column in standard.columns
        if column not in {*KEYS, "rul", "rul_normalized"}
    ]
    hcrd_features = [
        column
        for column in hcrd.columns
        if column not in {*KEYS, "rul", "rul_normalized"}
    ]
    frame = pd.concat(
        [standard[KEYS + standard_features], hcrd[hcrd_features]], axis=1
    )
    correlations = correlations_by_bearing(frame, standard_features + hcrd_features)
    feature_summary = summarize(correlations)
    correlations.to_csv(args.output / "indicator_correlations.csv", index=False)
    feature_summary.to_csv(args.output / "indicator_summary.csv", index=False)

    exact = feature_summary[feature_summary["family"] == "hcrd_exact_polygon_mass"]
    standard_summary = feature_summary[feature_summary["family"] == "standard"]
    result = {
        "analysis_status": "exploratory and post-outcome; not confirmatory",
        "target": "monotonic association with normalized life progress",
        "n_bearings": int(standard["bearing_id"].nunique()),
        "n_files": int(len(standard)),
        "selection_warning": (
            "Features ranked on all 15 bearings must be confirmed without reselection "
            "on an independent run-to-failure dataset."
        ),
        "best_exact_polygon_mass": exact.iloc[0].to_dict(),
        "best_standard_feature": standard_summary.iloc[0].to_dict(),
        "all_positive_exact_polygon_mass_features": exact.loc[
            exact["positive_fraction"] == 1.0, "feature"
        ].tolist(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
