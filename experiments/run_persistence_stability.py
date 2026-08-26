"""Computational audit of the signed-curvature persistence stability bound."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results" / "persistence_stability_t1"
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import decompose  # noqa: E402
from hcrd.persistence import (  # noqa: E402
    curvature_lipschitz_constant,
    curvature_persistence,
    curvature_persistence_distance,
)

SEED = 20261003
TRIALS_PER_GRID = 500


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for grid in ("uniform", "irregular"):
        for trial in range(TRIALS_PER_GRID):
            size = int(rng.integers(12, 65))
            x = (
                None
                if grid == "uniform"
                else np.concatenate(
                    ([0.0], np.cumsum(rng.uniform(0.4, 1.6, size - 1)))
                )
            )
            signal = rng.normal(size=size)
            requested_radius = float(10 ** rng.uniform(-7.0, -1.0))
            perturbation = rng.uniform(-1.0, 1.0, size=size)
            perturbation *= requested_radius / np.max(np.abs(perturbation))
            radius = float(np.max(np.abs(perturbation)))
            first = curvature_persistence(signal, x)
            second = curvature_persistence(signal + perturbation, x)
            distance = curvature_persistence_distance(first, second)
            bound = curvature_lipschitz_constant(size, x) * radius
            robust_bars = sum(
                bar.lifetime > 2.0 * bound
                for diagram in (first.positive, first.negative)
                for bar in diagram.bars
            )
            available_bars = len(second.positive.bars) + len(second.negative.bars)
            rows.append(
                {
                    "grid": grid,
                    "trial": trial,
                    "size": size,
                    "radius": radius,
                    "distance": distance,
                    "bound": bound,
                    "ratio": distance / bound if bound > 0 else 0.0,
                    "robust_finite_bars": robust_bars,
                    "perturbed_finite_bars": available_bars,
                    "bound_satisfied": distance <= bound + 2e-12,
                }
            )

    discontinuity_rows: list[dict[str, float]] = []
    for epsilon in 10.0 ** np.arange(-1, -9, -1, dtype=float):
        first_signal = np.asarray([-2.0, 0.0, 2.0 + epsilon, -2.0])
        second_signal = np.asarray([-2.0, 0.0, 2.0 - epsilon, -2.0])
        input_distance = float(np.max(np.abs(first_signal - second_signal)))
        first_baseline = decompose(
            first_signal, max_levels=1, atol=0.0, rtol=0.0
        ).levels[0].baseline
        second_baseline = decompose(
            second_signal, max_levels=1, atol=0.0, rtol=0.0
        ).levels[0].baseline
        hard_distance = float(np.max(np.abs(first_baseline - second_baseline)))
        persistence_distance = curvature_persistence_distance(
            curvature_persistence(first_signal),
            curvature_persistence(second_signal),
        )
        bound = 4.0 * input_distance
        discontinuity_rows.append(
            {
                "epsilon": float(epsilon),
                "input_distance": input_distance,
                "hard_baseline_distance": hard_distance,
                "hard_amplification": hard_distance / input_distance,
                "persistence_distance": persistence_distance,
                "persistence_bound": bound,
                "persistence_bound_ratio": persistence_distance / bound,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "random_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "discontinuity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(discontinuity_rows[0]))
        writer.writeheader()
        writer.writerows(discontinuity_rows)

    ratios = np.asarray([float(row["ratio"]) for row in rows])
    summary = {
        "seed": SEED,
        "trials": len(rows),
        "uniform_trials": TRIALS_PER_GRID,
        "irregular_trials": TRIALS_PER_GRID,
        "violations": sum(not bool(row["bound_satisfied"]) for row in rows),
        "maximum_bound_ratio": float(np.max(ratios)),
        "median_bound_ratio": float(np.median(ratios)),
        "q95_bound_ratio": float(np.quantile(ratios, 0.95)),
        "discontinuity": discontinuity_rows,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
