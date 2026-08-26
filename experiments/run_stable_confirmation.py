"""Frozen new-seed confirmation of the quadratic curvature guide."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import decompose  # noqa: E402
from hcrd.metrics import (  # noqa: E402
    exact_sign_test,
    jaccard_with_tolerance,
    knot_f1,
    mse,
    paired_bootstrap_ci,
)
from hcrd.robust import (  # noqa: E402
    adaptive_gaussian_guided_decompose,
    gaussian_guided_decompose,
    robust_decompose,
)
from hcrd.signals import alternating_chord_lobes  # noqa: E402
from hcrd.stable import quadratic_curvature_split  # noqa: E402

SEED = 20260950
NOISE_LEVELS = (0.01, 0.03, 0.05, 0.10)
LATENT_SIGNALS = 30
REPETITIONS = 10
REGULARIZATION = 3.0
METHODS = (
    "hcrd_raw",
    "hcrd_robust",
    "hcrd_gaussian_guided",
    "hcrd_adaptive_guided",
    "hcrd_quadratic_guided",
)


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "stable_confirmation_s2",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "frozen_protocol": str(PROJECT / "docs" / "stable_confirmation_protocol.md"),
        "seed": SEED,
        "noise_levels": NOISE_LEVELS,
        "latent_signals": LATENT_SIGNALS,
        "repetitions": REPETITIONS,
        "quadratic_regularization": REGULARIZATION,
        "methods": METHODS,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    trials: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for noise_sigma in NOISE_LEVELS:
        for latent_index in range(LATENT_SIGNALS):
            latent = alternating_chord_lobes(
                seed=SEED + latent_index, noise_sigma=0.0
            )
            observations: list[np.ndarray] = []
            method_knots = {method: [] for method in METHODS}
            method_baselines = {method: [] for method in METHODS}
            quadratic_guides: list[np.ndarray] = []
            for repetition in range(REPETITIONS):
                rng = np.random.default_rng(
                    SEED
                    + 1_000_000 * latent_index
                    + 10_000 * round(1000 * noise_sigma)
                    + repetition
                )
                observed = latent.baseline + latent.detail + rng.normal(
                    0.0, noise_sigma, size=latent.observed.size
                )
                raw = decompose(observed, latent.x, max_levels=1).levels[0]
                robust = robust_decompose(
                    observed, latent.x, z_score=3.5, max_levels=1
                ).decomposition.levels[0]
                gaussian = gaussian_guided_decompose(
                    observed, latent.x, smoothing_sigma=2.0, max_levels=1
                ).decomposition.levels[0]
                adaptive = adaptive_gaussian_guided_decompose(
                    observed, latent.x
                ).guided.decomposition.levels[0]
                split = quadratic_curvature_split(
                    observed, latent.x, regularization=REGULARIZATION
                )
                quadratic = decompose(
                    split.guide,
                    latent.x,
                    atol=1e-12,
                    rtol=64 * np.finfo(float).eps,
                    max_levels=1,
                ).levels[0]
                values = {
                    "hcrd_raw": raw,
                    "hcrd_robust": robust,
                    "hcrd_gaussian_guided": gaussian,
                    "hcrd_adaptive_guided": adaptive,
                    "hcrd_quadratic_guided": quadratic,
                }
                observations.append(observed)
                quadratic_guides.append(split.guide)
                for method, level in values.items():
                    method_knots[method].append(level.knots)
                    method_baselines[method].append(level.baseline)
                    trials.append(
                        {
                            "noise_sigma": noise_sigma,
                            "latent_index": latent_index,
                            "repetition": repetition,
                            "method": method,
                            "target_knot_f1": knot_f1(
                                level.knots, latent.knots, tolerance=1
                            ),
                            "baseline_mse": mse(level.baseline, latent.baseline),
                            "knot_count": level.knots.size,
                        }
                    )
            for method in METHODS:
                similarities = [
                    jaccard_with_tolerance(
                        method_knots[method][first],
                        method_knots[method][second],
                        tolerance=1,
                    )
                    for first, second in combinations(range(REPETITIONS), 2)
                ]
                stability.append(
                    {
                        "noise_sigma": noise_sigma,
                        "latent_index": latent_index,
                        "method": method,
                        "median_pairwise_knot_jaccard": float(np.median(similarities)),
                    }
                )
            guide_ratios = []
            hard_ratios = []
            for first, second in combinations(range(REPETITIONS), 2):
                input_distance = np.linalg.norm(
                    observations[first] - observations[second]
                )
                guide_ratios.append(
                    np.linalg.norm(quadratic_guides[first] - quadratic_guides[second])
                    / input_distance
                )
                hard_ratios.append(
                    np.linalg.norm(
                        method_baselines["hcrd_quadratic_guided"][first]
                        - method_baselines["hcrd_quadratic_guided"][second]
                    )
                    / input_distance
                )
            stability[-1]["max_quadratic_guide_l2_ratio"] = float(np.max(guide_ratios))
            stability[-1]["max_quadratic_hard_l2_ratio"] = float(np.max(hard_ratios))

    for filename, rows in (("trials.csv", trials), ("stability.csv", stability)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=sorted({key for row in rows for key in row}),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    latent_rows = []
    for noise_sigma in NOISE_LEVELS:
        for latent_index in range(LATENT_SIGNALS):
            for method in METHODS:
                group = [
                    row
                    for row in trials
                    if row["noise_sigma"] == noise_sigma
                    and row["latent_index"] == latent_index
                    and row["method"] == method
                ]
                stability_row = next(
                    row
                    for row in stability
                    if row["noise_sigma"] == noise_sigma
                    and row["latent_index"] == latent_index
                    and row["method"] == method
                )
                latent_rows.append(
                    {
                        "noise_sigma": noise_sigma,
                        "latent_index": latent_index,
                        "method": method,
                        "mean_target_knot_f1": float(
                            np.mean([row["target_knot_f1"] for row in group])
                        ),
                        "mean_baseline_mse": float(
                            np.mean([row["baseline_mse"] for row in group])
                        ),
                        "median_pairwise_knot_jaccard": stability_row[
                            "median_pairwise_knot_jaccard"
                        ],
                    }
                )
    with (output / "latent_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latent_rows[0]))
        writer.writeheader()
        writer.writerows(latent_rows)

    comparisons = []
    aggregate = []
    for noise_sigma in NOISE_LEVELS:
        noise_rows = [row for row in latent_rows if row["noise_sigma"] == noise_sigma]
        for method in METHODS:
            group = [row for row in noise_rows if row["method"] == method]
            aggregate.append(
                {
                    "noise_sigma": noise_sigma,
                    "method": method,
                    "mean_target_knot_f1": float(
                        np.mean([row["mean_target_knot_f1"] for row in group])
                    ),
                    "mean_baseline_mse": float(
                        np.mean([row["mean_baseline_mse"] for row in group])
                    ),
                    "median_knot_jaccard": float(
                        np.median([row["median_pairwise_knot_jaccard"] for row in group])
                    ),
                }
            )
        target = sorted(
            [row for row in noise_rows if row["method"] == "hcrd_quadratic_guided"],
            key=lambda row: row["latent_index"],
        )
        for endpoint in ("mean_target_knot_f1", "mean_baseline_mse"):
            endpoint_rows = []
            for method in METHODS[:-1]:
                comparator = sorted(
                    [row for row in noise_rows if row["method"] == method],
                    key=lambda row: row["latent_index"],
                )
                differences = np.array(
                    [a[endpoint] - b[endpoint] for a, b in zip(target, comparator)]
                )
                lower, upper = paired_bootstrap_ci(
                    differences,
                    samples=20_000,
                    seed=SEED + round(1000 * noise_sigma),
                )
                endpoint_rows.append(
                    {
                        "noise_sigma": noise_sigma,
                        "endpoint": endpoint,
                        "comparison": f"hcrd_quadratic_guided - {method}",
                        "mean_difference": float(np.mean(differences)),
                        "bootstrap_ci_lower": lower,
                        "bootstrap_ci_upper": upper,
                        "exact_sign_p": exact_sign_test(differences),
                    }
                )
            adjusted = _holm([row["exact_sign_p"] for row in endpoint_rows])
            for row, p_value in zip(endpoint_rows, adjusted):
                row["holm_adjusted_sign_p"] = p_value
                comparisons.append(row)

    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    print(json.dumps({"trials": len(trials), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
