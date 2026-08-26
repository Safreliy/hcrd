"""Comparison against EMD, VMD, and L1 trend filtering."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.baselines import gaussian_smooth  # noqa: E402
from hcrd.core import decompose  # noqa: E402
from hcrd.external_baselines import emd_residue, l1_trend_filter_path, vmd_low_frequency  # noqa: E402
from hcrd.metrics import (  # noqa: E402
    exact_sign_test,
    mse,
    paired_bootstrap_ci,
    scaled_mse,
)
from hcrd.robust import adaptive_gaussian_guided_decompose, robust_decompose  # noqa: E402
from hcrd.signals import alternating_chord_lobes  # noqa: E402


def _oracle(candidates: list[tuple[str, np.ndarray]], target: np.ndarray) -> tuple[str, np.ndarray]:
    return min(candidates, key=lambda item: mse(item[1], target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.03, 0.10])
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "external_comparison")
    parser.add_argument("--suite", choices=("exact", "variable", "all"), default="all")
    parser.add_argument("--primary-noise", type=float, default=0.0)
    parser.add_argument("--reference", default="hcrd_centered")
    args = parser.parse_args()
    if args.trials < 2:
        raise SystemExit("at least two trials are required outside smoke testing")

    rows: list[dict[str, object]] = []
    suites = ("exact", "variable") if args.suite == "all" else (args.suite,)
    for suite in suites:
        for noise_sigma in args.noise:
            for trial in range(args.trials):
                seed = args.seed + (100_000_000 if suite == "variable" else 0) + round(
                    10_000 * noise_sigma
                ) + trial
                signal = alternating_chord_lobes(
                    seed=seed,
                    noise_sigma=noise_sigma,
                    piecewise_baseline=suite == "variable",
                    amplitude_variation=suite == "variable",
                )
                hcrd = decompose(signal.observed, signal.x).levels[0].baseline
                thresholded = robust_decompose(signal.observed, signal.x).decomposition.levels[0].baseline
                adaptive = adaptive_gaussian_guided_decompose(
                    signal.observed, signal.x
                ).guided.decomposition.levels[0].baseline
                l1_values = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
                l1_path = l1_trend_filter_path(signal.observed, l1_values)
                l1_parameter, l1_estimate = _oracle(
                    [
                        (f"lambda={value:g}", estimate)
                        for value, estimate in zip(l1_values, l1_path, strict=True)
                    ],
                    signal.baseline,
                )
                gaussian_parameter, gaussian_estimate = _oracle(
                    [
                        (f"sigma={value:g}", gaussian_smooth(signal.observed, value))
                        for value in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
                    ],
                    signal.baseline,
                )
                estimates = {
                    "hcrd_centered": ("none", hcrd),
                    "hcrd_thresholded": ("z=3.5", thresholded),
                    "hcrd_adaptive_guided": ("pilot-calibrated", adaptive),
                    "emd_residue": ("default", emd_residue(signal.observed, signal.x)),
                    "vmd_low_mode": ("K=5;alpha=2000", vmd_low_frequency(signal.observed)),
                    "l1_trend_oracle": (l1_parameter, l1_estimate),
                    "gaussian_oracle": (gaussian_parameter, gaussian_estimate),
                }
                for method, (parameter, estimate) in estimates.items():
                    rows.append(
                        {
                            "suite": suite,
                            "noise_sigma": noise_sigma,
                            "trial": trial,
                            "seed": seed,
                            "method": method,
                            "parameter": parameter,
                            "baseline_mse": mse(estimate, signal.baseline),
                            "baseline_nmse": scaled_mse(
                                estimate, signal.baseline, signal.observed
                            ),
                        }
                    )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for suite in suites:
        for noise_sigma in args.noise:
            for method in sorted({row["method"] for row in rows}):
                losses = [
                    float(row["baseline_mse"])
                    for row in rows
                    if row["suite"] == suite
                    and row["noise_sigma"] == noise_sigma
                    and row["method"] == method
                ]
                summary.append(
                    {
                        "suite": suite,
                        "noise_sigma": noise_sigma,
                        "method": method,
                        "mean_mse": float(np.mean(losses)),
                        "median_mse": float(np.median(losses)),
                    }
                )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    primary = [
        row
        for row in rows
        if row["suite"] == "exact"
        and np.isclose(float(row["noise_sigma"]), args.primary_noise)
    ]
    methods = sorted({str(row["method"]) for row in primary})
    by_method = {
        method: sorted(
            [row for row in primary if row["method"] == method],
            key=lambda row: int(row["trial"]),
        )
        for method in methods
    }
    if args.reference not in by_method:
        raise RuntimeError(f"reference method {args.reference!r} is absent from primary rows")
    reference = np.asarray(
        [float(row["baseline_mse"]) for row in by_method[args.reference]]
    )
    comparisons: list[dict[str, object]] = []
    for method in methods:
        if method == args.reference:
            continue
        competitor = np.asarray(
            [float(row["baseline_mse"]) for row in by_method[method]]
        )
        differences = reference - competitor
        differences[np.abs(differences) < 1e-12] = 0.0
        low, high = paired_bootstrap_ci(differences, seed=args.seed)
        comparisons.append(
            {
                "reference": args.reference,
                "competitor": method,
                "mean_difference_reference_minus_competitor": float(np.mean(differences)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "reference_win_rate": float(np.mean(differences < 0)),
                "exact_sign_p": exact_sign_test(differences),
            }
        )
    order = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["exact_sign_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(
            1.0,
            (len(comparisons) - rank) * float(comparisons[index]["exact_sign_p"]),
        )
        running = max(running, adjusted)
        comparisons[index]["holm_adjusted_p"] = running
    with (args.output / "primary_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": "docs/experiment_protocol.md version 0.5",
        "primary_noise": args.primary_noise,
        "reference": args.reference,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"rows": len(rows), "comparisons": len(comparisons), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
