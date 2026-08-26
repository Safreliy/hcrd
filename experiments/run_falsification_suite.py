"""Out-of-class experiments that actively search for HCRD failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.baselines import gaussian_smooth  # noqa: E402
from hcrd.core import decompose  # noqa: E402
from hcrd.metrics import mse  # noqa: E402
from hcrd.robust import adaptive_gaussian_guided_decompose, robust_decompose  # noqa: E402
from hcrd.signals import alternating_chord_lobes  # noqa: E402


def _oracle_gaussian(observed: np.ndarray, target: np.ndarray) -> np.ndarray:
    estimates = [gaussian_smooth(observed, value) for value in (1, 2, 4, 8, 16, 32, 64)]
    return min(estimates, key=lambda estimate: mse(estimate, target))


def _methods(observed: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "hcrd_raw": decompose(observed, x).levels[0].baseline,
        "hcrd_thresholded": robust_decompose(observed, x).decomposition.levels[0].baseline,
        "hcrd_adaptive_guided": adaptive_gaussian_guided_decompose(
            observed, x
        ).guided.decomposition.levels[0].baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "falsification")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []

    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed + trial)
        x = np.linspace(-1.0, 1.0, 513)

        # Strong curvature can hide a genuine oscillation from any sign-based
        # inflection detector.
        # Here |a|*omega^2 remains below the quadratic curvature 8, satisfying
        # the visibility-limit proposition rather than merely illustrating it.
        amplitude = rng.uniform(0.0002, 0.0015)
        trend = 4.0 * x**2 + rng.normal(0.0, 0.2) * x
        oscillation = amplitude * np.sin(2.0 * np.pi * 10.0 * x)
        observed = trend + oscillation
        methods = _methods(observed, x)
        methods["gaussian_oracle"] = _oracle_gaussian(observed, trend)
        for method, baseline in methods.items():
            detail = observed - baseline
            correlation = float(np.corrcoef(detail, oscillation)[0, 1]) if np.std(detail) else 0.0
            rows.append(
                {
                    "case": "weak_oscillation_strong_curvature",
                    "trial": trial,
                    "method": method,
                    "baseline_mse": mse(baseline, trend),
                    "detail_correlation": correlation,
                    "endpoint_error": "",
                }
            )

        # Endpoint contamination propagates into every chord-based coarse level.
        clean = alternating_chord_lobes(
            seed=args.seed + 100_000 + trial,
            piecewise_baseline=False,
            amplitude_variation=False,
        )
        contaminated = clean.observed.copy()
        outlier = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
        contaminated[rng.choice([0, contaminated.size - 1])] += outlier
        methods = _methods(contaminated, clean.x)
        methods["gaussian_oracle"] = _oracle_gaussian(contaminated, clean.baseline)
        for method, baseline in methods.items():
            rows.append(
                {
                    "case": "endpoint_outlier",
                    "trial": trial,
                    "method": method,
                    "baseline_mse": mse(baseline, clean.baseline),
                    "detail_correlation": "",
                    "endpoint_error": float(
                        max(
                            abs(baseline[0] - clean.baseline[0]),
                            abs(baseline[-1] - clean.baseline[-1]),
                        )
                    ),
                }
            )

        # Divided slopes should retain affine invariance on an irregular grid.
        gaps = rng.uniform(0.5, 1.5, size=512)
        irregular_x = np.concatenate([[0.0], np.cumsum(gaps)])
        irregular_x /= irregular_x[-1]
        irregular_trend = rng.normal() + rng.normal() * irregular_x
        irregular_detail = np.sin(2.0 * np.pi * 8.0 * irregular_x)
        irregular_observed = irregular_trend + irregular_detail
        raw = decompose(irregular_observed, irregular_x).levels[0].baseline
        rows.append(
            {
                "case": "irregular_grid_affine_plus_sine",
                "trial": trial,
                "method": "hcrd_raw",
                "baseline_mse": mse(raw, irregular_trend),
                "detail_correlation": float(
                    np.corrcoef(irregular_observed - raw, irregular_detail)[0, 1]
                ),
                "endpoint_error": "",
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for case in sorted({row["case"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["case"] == case}):
            group = [row for row in rows if row["case"] == case and row["method"] == method]
            summary.append(
                {
                    "case": case,
                    "method": method,
                    "mean_baseline_mse": float(np.mean([float(row["baseline_mse"]) for row in group])),
                    "mean_detail_correlation": (
                        float(np.mean([float(row["detail_correlation"]) for row in group]))
                        if group[0]["detail_correlation"] != ""
                        else None
                    ),
                    "mean_endpoint_error": (
                        float(np.mean([float(row["endpoint_error"]) for row in group]))
                        if group[0]["endpoint_error"] != ""
                        else None
                    ),
                }
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
