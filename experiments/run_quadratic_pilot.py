"""Exploratory grid for the stable quadratic-curvature HCRD guide."""

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
from hcrd.metrics import jaccard_with_tolerance, knot_f1, mse  # noqa: E402
from hcrd.signals import alternating_chord_lobes  # noqa: E402
from hcrd.stable import quadratic_curvature_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--noise", type=float, nargs="*", default=[0.01, 0.03, 0.05, 0.10]
    )
    parser.add_argument(
        "--regularizations",
        type=float,
        nargs="*",
        default=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "quadratic_pilot"
    )
    args = parser.parse_args()
    if args.signals < 1 or args.repetitions < 2:
        raise ValueError("signals >= 1 and repetitions >= 2 are required")

    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "exploratory parameter pilot; not confirmatory evidence",
        "seed": args.seed,
        "signals": args.signals,
        "repetitions": args.repetitions,
        "noise_sigmas": args.noise,
        "regularizations": args.regularizations,
        "guide": "argmin_z 0.5||y-z||_2^2 + 0.5 lambda ||D_curvature z||_2^2",
        "hard_output": "one centred HCRD level applied to the guide",
        "warning": (
            "The guide and residual are globally nonexpansive; the subsequent "
            "hard HCRD baseline is not asserted to be globally stable."
        ),
        "selection_note": "Freeze any selected lambda before new-seed confirmation.",
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    trials: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for noise_sigma in args.noise:
        for regularization in args.regularizations:
            for latent_index in range(args.signals):
                latent = alternating_chord_lobes(
                    seed=args.seed + latent_index, noise_sigma=0.0
                )
                observations: list[np.ndarray] = []
                guides: list[np.ndarray] = []
                baselines: list[np.ndarray] = []
                knot_sets: list[np.ndarray] = []
                for repetition in range(args.repetitions):
                    rng = np.random.default_rng(
                        args.seed
                        + 1_000_000 * latent_index
                        + 10_000 * round(1000 * noise_sigma)
                        + repetition
                    )
                    observed = latent.baseline + latent.detail + rng.normal(
                        0.0, noise_sigma, size=latent.observed.size
                    )
                    split = quadratic_curvature_split(
                        observed, latent.x, regularization=regularization
                    )
                    level = decompose(
                        split.guide,
                        latent.x,
                        atol=1e-12,
                        rtol=64 * np.finfo(float).eps,
                        max_levels=1,
                    ).levels[0]
                    observations.append(observed)
                    guides.append(split.guide)
                    baselines.append(level.baseline)
                    knot_sets.append(level.knots)
                    trials.append(
                        {
                            "noise_sigma": noise_sigma,
                            "regularization": regularization,
                            "latent_index": latent_index,
                            "repetition": repetition,
                            "baseline_mse": mse(level.baseline, latent.baseline),
                            "guide_mse": mse(split.guide, latent.baseline),
                            "target_knot_f1": knot_f1(
                                level.knots, latent.knots, tolerance=1
                            ),
                            "knot_count": level.knots.size,
                        }
                    )
                for first, second in combinations(range(args.repetitions), 2):
                    input_distance = np.linalg.norm(
                        observations[first] - observations[second]
                    )
                    pairs.append(
                        {
                            "noise_sigma": noise_sigma,
                            "regularization": regularization,
                            "latent_index": latent_index,
                            "knot_jaccard": jaccard_with_tolerance(
                                knot_sets[first], knot_sets[second], tolerance=1
                            ),
                            "guide_l2_ratio": np.linalg.norm(
                                guides[first] - guides[second]
                            )
                            / input_distance,
                            "hcrd_baseline_l2_ratio": np.linalg.norm(
                                baselines[first] - baselines[second]
                            )
                            / input_distance,
                        }
                    )

    for filename, rows in (("trials.csv", trials), ("pairs.csv", pairs)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = []
    for noise_sigma in args.noise:
        for regularization in args.regularizations:
            trial_group = [
                row
                for row in trials
                if row["noise_sigma"] == noise_sigma
                and row["regularization"] == regularization
            ]
            pair_group = [
                row
                for row in pairs
                if row["noise_sigma"] == noise_sigma
                and row["regularization"] == regularization
            ]
            summary.append(
                {
                    "noise_sigma": noise_sigma,
                    "regularization": regularization,
                    "mean_baseline_mse": float(
                        np.mean([row["baseline_mse"] for row in trial_group])
                    ),
                    "mean_guide_mse": float(
                        np.mean([row["guide_mse"] for row in trial_group])
                    ),
                    "mean_target_knot_f1": float(
                        np.mean([row["target_knot_f1"] for row in trial_group])
                    ),
                    "mean_knot_count": float(
                        np.mean([row["knot_count"] for row in trial_group])
                    ),
                    "median_knot_jaccard": float(
                        np.median([row["knot_jaccard"] for row in pair_group])
                    ),
                    "max_guide_l2_ratio": float(
                        np.max([row["guide_l2_ratio"] for row in pair_group])
                    ),
                    "max_hcrd_baseline_l2_ratio": float(
                        np.max([row["hcrd_baseline_l2_ratio"] for row in pair_group])
                    ),
                }
            )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"trials": len(trials), "pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
