"""Compare raw and noise-aware knot stability under repeated perturbations."""

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
from hcrd.metrics import jaccard_with_tolerance, knot_f1  # noqa: E402
from hcrd.robust import (  # noqa: E402
    adaptive_gaussian_guided_decompose,
    gaussian_guided_decompose,
    robust_decompose,
)
from hcrd.signals import alternating_chord_lobes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--noise", type=float, nargs="*", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "stability")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for noise_sigma in args.noise:
        for latent_index in range(args.signals):
            latent = alternating_chord_lobes(
                seed=args.seed + latent_index,
                noise_sigma=0.0,
            )
            raw_knots: list[np.ndarray] = []
            robust_knots: list[np.ndarray] = []
            guided_knots: list[np.ndarray] = []
            adaptive_knots: list[np.ndarray] = []
            raw_baselines: list[np.ndarray] = []
            robust_baselines: list[np.ndarray] = []
            guided_baselines: list[np.ndarray] = []
            adaptive_baselines: list[np.ndarray] = []
            for repetition in range(args.repetitions):
                rng = np.random.default_rng(
                    args.seed + 1_000_000 * latent_index + 10_000 * round(noise_sigma * 1000) + repetition
                )
                observed = latent.baseline + latent.detail + rng.normal(
                    0.0, noise_sigma, size=latent.observed.size
                )
                raw = decompose(observed, latent.x).levels[0]
                robust = robust_decompose(observed, latent.x, z_score=3.5).decomposition.levels[0]
                guided = gaussian_guided_decompose(
                    observed, latent.x, smoothing_sigma=2.0
                ).decomposition.levels[0]
                adaptive = adaptive_gaussian_guided_decompose(
                    observed, latent.x
                ).guided.decomposition.levels[0]
                raw_knots.append(raw.knots)
                robust_knots.append(robust.knots)
                guided_knots.append(guided.knots)
                adaptive_knots.append(adaptive.knots)
                raw_baselines.append(raw.baseline)
                robust_baselines.append(robust.baseline)
                guided_baselines.append(guided.baseline)
                adaptive_baselines.append(adaptive.baseline)

            for method, knot_sets, baselines in (
                ("hcrd_raw", raw_knots, raw_baselines),
                ("hcrd_robust", robust_knots, robust_baselines),
                ("hcrd_guided", guided_knots, guided_baselines),
                ("hcrd_adaptive_guided", adaptive_knots, adaptive_baselines),
            ):
                similarities = [
                    jaccard_with_tolerance(knot_sets[i], knot_sets[j], tolerance=1)
                    for i, j in combinations(range(args.repetitions), 2)
                ]
                baseline_stack = np.stack(baselines)
                rows.append(
                    {
                        "noise_sigma": noise_sigma,
                        "latent_index": latent_index,
                        "method": method,
                        "mean_pairwise_jaccard": float(np.mean(similarities)),
                        "mean_knot_count": float(np.mean([len(knots) for knots in knot_sets])),
                        "mean_target_knot_f1": float(
                            np.mean([knot_f1(knots, latent.knots, tolerance=1) for knots in knot_sets])
                        ),
                        "mean_pointwise_baseline_variance": float(np.mean(np.var(baseline_stack, axis=0))),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stability_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for noise_sigma in args.noise:
        for method in (
            "hcrd_raw",
            "hcrd_robust",
            "hcrd_guided",
            "hcrd_adaptive_guided",
        ):
            group = [
                row for row in rows if row["noise_sigma"] == noise_sigma and row["method"] == method
            ]
            summary.append(
                {
                    "noise_sigma": noise_sigma,
                    "method": method,
                    "median_jaccard": float(np.median([row["mean_pairwise_jaccard"] for row in group])),
                    "mean_knot_count": float(np.mean([row["mean_knot_count"] for row in group])),
                    "mean_target_knot_f1": float(
                        np.mean([row["mean_target_knot_f1"] for row in group])
                    ),
                    "mean_baseline_variance": float(
                        np.mean([row["mean_pointwise_baseline_variance"] for row in group])
                    ),
                }
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
