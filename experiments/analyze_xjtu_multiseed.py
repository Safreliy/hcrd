"""Paired, bearing-clustered analysis of the XJTU-SY multi-seed experiment."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


METRICS = ("rmse", "mae", "late_rmse")


def _bootstrap_mean_ci(
    values: np.ndarray, *, draws: int = 200_000, seed: int = 20260824
) -> tuple[float, float]:
    """Percentile CI obtained by resampling independent bearings."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws, dtype=float)
    chunk = 10_000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_pvalue(values: np.ndarray) -> float:
    """Two-sided exact paired randomization p-value for the mean difference."""
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def compare(
    scores: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    bootstrap_draws: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"representation", "seed", "condition", "test_bearing", *METRICS}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    selected = scores[scores["representation"].isin([baseline, candidate])].copy()
    if set(selected["representation"].unique()) != {baseline, candidate}:
        raise ValueError("Both requested representations must be present")

    keys = ["condition", "test_bearing", "seed"]
    wide = selected.pivot(index=keys, columns="representation", values=list(METRICS))
    if wide.isna().any().any():
        raise ValueError("Incomplete baseline/candidate pairing")

    bearing_rows: list[dict[str, object]] = []
    for (condition, bearing), group in wide.groupby(level=["condition", "test_bearing"]):
        row: dict[str, object] = {
            "condition": condition,
            "test_bearing": bearing,
            "n_seeds": int(len(group)),
        }
        for metric in METRICS:
            base = group[(metric, baseline)].to_numpy(dtype=float)
            cand = group[(metric, candidate)].to_numpy(dtype=float)
            row[f"{baseline}_{metric}"] = float(base.mean())
            row[f"{candidate}_{metric}"] = float(cand.mean())
            row[f"delta_{metric}"] = float((cand - base).mean())
        bearing_rows.append(row)
    by_bearing = pd.DataFrame(bearing_rows).sort_values(["condition", "test_bearing"])

    result: dict[str, object] = {
        "analysis_status": "post-outcome robustness analysis",
        "unit_of_generalization": "bearing",
        "baseline": baseline,
        "candidate": candidate,
        "n_bearings": int(len(by_bearing)),
        "n_seeds": int(scores["seed"].nunique()),
        "bootstrap_draws": bootstrap_draws,
        "metrics": {},
    }
    for metric in METRICS:
        delta = by_bearing[f"delta_{metric}"].to_numpy(dtype=float)
        base = by_bearing[f"{baseline}_{metric}"].to_numpy(dtype=float)
        cand = by_bearing[f"{candidate}_{metric}"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(delta, draws=bootstrap_draws)
        nonzero = delta[delta != 0]
        wins = int(np.sum(delta < 0))
        wilcoxon_result = wilcoxon(delta, alternative="two-sided", method="auto")
        result["metrics"][metric] = {
            "baseline_macro_mean": float(base.mean()),
            "candidate_macro_mean": float(cand.mean()),
            "candidate_minus_baseline": float(delta.mean()),
            "relative_change_percent": float(100.0 * delta.mean() / base.mean()),
            "bearing_cluster_bootstrap_95_ci": [ci_low, ci_high],
            "bearings_improved": wins,
            "bearings_total": int(len(delta)),
            "exact_sign_test_p_two_sided": float(
                binomtest(wins, len(nonzero), 0.5, alternative="two-sided").pvalue
            ),
            "wilcoxon_p_two_sided": float(wilcoxon_result.pvalue),
            "exact_mean_sign_flip_p_two_sided": _exact_sign_flip_pvalue(delta),
        }
    return by_bearing, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="standard")
    parser.add_argument("--candidate", default="hybrid_mass6")
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    by_bearing, result = compare(
        scores,
        baseline=args.baseline,
        candidate=args.candidate,
        bootstrap_draws=args.bootstrap_draws,
    )
    by_bearing.to_csv(args.output / "paired_bearing_deltas.csv", index=False)
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
