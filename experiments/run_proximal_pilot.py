"""Exploratory regularization grid for the proximal-guided HCRD companion."""

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
from hcrd.stable import proximal_curvature_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--noise", type=float, nargs="*", default=[0.01, 0.03, 0.05, 0.10]
    )
    parser.add_argument(
        "--ratios", type=float, nargs="*", default=[0.3, 1.0, 3.0, 10.0, 30.0]
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "proximal_pilot"
    )
    args = parser.parse_args()
    if args.signals < 1 or args.repetitions < 2:
        raise ValueError("signals >= 1 and repetitions >= 2 are required")

    protocol = {
        "status": "exploratory parameter pilot; not confirmatory evidence",
        "seed": args.seed,
        "signals": args.signals,
        "repetitions": args.repetitions,
        "noise_sigmas": args.noise,
        "regularization_rule": "lambda = ratio * known noise sigma",
        "ratios": args.ratios,
        "metrics": [
            "baseline_mse",
            "target_knot_f1_tolerance_1",
            "pairwise_knot_jaccard_tolerance_1",
            "pairwise_guide_l2_ratio",
            "pairwise_hcrd_baseline_l2_ratio",
        ],
        "selection_note": (
            "Any chosen ratio must be written to a separate frozen confirmation "
            "protocol and evaluated on new seeds."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    trial_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for noise_sigma in args.noise:
        for ratio in args.ratios:
            regularization = ratio * noise_sigma
            for latent_index in range(args.signals):
                latent = alternating_chord_lobes(
                    seed=args.seed + latent_index,
                    noise_sigma=0.0,
                )
                observations: list[np.ndarray] = []
                guides: list[np.ndarray] = []
                baselines: list[np.ndarray] = []
                knots: list[np.ndarray] = []
                for repetition in range(args.repetitions):
                    rng = np.random.default_rng(
                        args.seed
                        + 1_000_000 * latent_index
                        + 10_000 * round(noise_sigma * 1000)
                        + repetition
                    )
                    observed = latent.baseline + latent.detail + rng.normal(
                        0.0, noise_sigma, size=latent.observed.size
                    )
                    split = proximal_curvature_split(
                        observed,
                        latent.x,
                        regularization=regularization,
                    )
                    level = decompose(
                        split.guide,
                        latent.x,
                        atol=1e-8,
                        rtol=64 * np.finfo(float).eps,
                        max_levels=1,
                    ).levels[0]
                    observations.append(observed)
                    guides.append(split.guide)
                    baselines.append(level.baseline)
                    knots.append(level.knots)
                    trial_rows.append(
                        {
                            "noise_sigma": noise_sigma,
                            "ratio": ratio,
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
                    pair_rows.append(
                        {
                            "noise_sigma": noise_sigma,
                            "ratio": ratio,
                            "regularization": regularization,
                            "latent_index": latent_index,
                            "first_repetition": first,
                            "second_repetition": second,
                            "knot_jaccard": jaccard_with_tolerance(
                                knots[first], knots[second], tolerance=1
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

    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    with (args.output / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = []
    for noise_sigma in args.noise:
        for ratio in args.ratios:
            trials = [
                row
                for row in trial_rows
                if row["noise_sigma"] == noise_sigma and row["ratio"] == ratio
            ]
            pairs = [
                row
                for row in pair_rows
                if row["noise_sigma"] == noise_sigma and row["ratio"] == ratio
            ]
            summary.append(
                {
                    "noise_sigma": noise_sigma,
                    "ratio": ratio,
                    "mean_baseline_mse": float(
                        np.mean([row["baseline_mse"] for row in trials])
                    ),
                    "mean_guide_mse": float(
                        np.mean([row["guide_mse"] for row in trials])
                    ),
                    "mean_target_knot_f1": float(
                        np.mean([row["target_knot_f1"] for row in trials])
                    ),
                    "mean_knot_count": float(
                        np.mean([row["knot_count"] for row in trials])
                    ),
                    "median_knot_jaccard": float(
                        np.median([row["knot_jaccard"] for row in pairs])
                    ),
                    "max_guide_l2_ratio": float(
                        np.max([row["guide_l2_ratio"] for row in pairs])
                    ),
                    "max_hcrd_baseline_l2_ratio": float(
                        np.max([row["hcrd_baseline_l2_ratio"] for row in pairs])
                    ),
                }
            )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"trials": len(trial_rows), "pairs": len(pair_rows)}, indent=2))


if __name__ == "__main__":
    main()
