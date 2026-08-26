"""Finite-sample phase diagram for the generative chord-lobe theorem."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import find_convexity_knots  # noqa: E402
from hcrd.metrics import knot_f1, mse  # noqa: E402
from hcrd.recovery import (  # noqa: E402
    alternating_parabolic_chord_lobes,
    amplitude_for_curvature_ratio,
    finite_sample_recovery_thresholds,
)


DEFAULT_CONFIGURATIONS = ((2, 4), (4, 8), (8, 16), (16, 8))
DEFAULT_RATIOS = (0.50, 0.75, 1.00, 1.15, 1.30, 1.45, 1.60, 1.75, 1.90, 2.05, 2.25)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_configuration(value: str) -> tuple[int, int]:
    try:
        lobes, width = (int(part) for part in value.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("configuration must be LOBES:WIDTH") from error
    if lobes < 2 or width < 2:
        raise argparse.ArgumentTypeError("LOBES and WIDTH must both be at least two")
    return lobes, width


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)
    )
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def _localisation_error(estimate: np.ndarray, truth: np.ndarray, width: int, n: int) -> float:
    estimate_inner = estimate[1:-1]
    truth_inner = truth[1:-1]
    if estimate_inner.size == 0 or truth_inner.size == 0:
        return float(n / width)
    first = max(float(np.min(np.abs(estimate_inner - value))) for value in truth_inner)
    second = max(float(np.min(np.abs(truth_inner - value))) for value in estimate_inner)
    return max(first, second) / width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--configuration",
        action="append",
        type=_parse_configuration,
        dest="configurations",
        help="repeatable LOBES:WIDTH specification",
    )
    parser.add_argument("--ratio", type=float, nargs="*", default=list(DEFAULT_RATIOS))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "recovery_phase_r1",
    )
    args = parser.parse_args()
    if args.trials < 20:
        raise SystemExit("at least 20 trials are required")
    if args.sigma <= 0 or not 0 < args.delta < 1:
        raise SystemExit("sigma must be positive and delta must lie in (0,1)")
    if not args.ratio or any(value <= 0 for value in args.ratio):
        raise SystemExit("all curvature ratios must be positive")

    configurations = tuple(args.configurations or DEFAULT_CONFIGURATIONS)
    rows: list[dict[str, object]] = []
    for configuration_index, (lobes, width) in enumerate(configurations):
        n = lobes * width + 1
        threshold, sample_radius = finite_sample_recovery_thresholds(
            n, args.sigma, delta=args.delta
        )
        for ratio_index, ratio in enumerate(args.ratio):
            amplitude = amplitude_for_curvature_ratio(ratio, threshold, width)
            for trial in range(args.trials):
                seed = (
                    args.seed
                    + 100_000_000 * configuration_index
                    + 1_000_000 * ratio_index
                    + trial
                )
                signal = alternating_parabolic_chord_lobes(
                    seed=seed,
                    lobes=lobes,
                    samples_per_lobe=width,
                    amplitude=amplitude,
                    noise_sigma=args.sigma,
                )
                start = time.perf_counter_ns()
                knots = find_convexity_knots(
                    signal.observed,
                    signal.x,
                    atol=threshold,
                    rtol=0.0,
                    boundary_rule="minimum_curvature",
                )
                estimate = np.interp(
                    signal.x, signal.x[knots], signal.observed[knots]
                )
                elapsed_ns = time.perf_counter_ns() - start
                detail_estimate = signal.observed - estimate
                baseline_sup = float(np.max(np.abs(estimate - signal.baseline)))
                detail_sup = float(np.max(np.abs(detail_estimate - signal.detail)))
                exact = bool(np.array_equal(knots, signal.knots))
                joint_certificate = bool(
                    exact
                    and baseline_sup <= sample_radius + 1e-12
                    and detail_sup <= 2.0 * sample_radius + 1e-12
                )
                rows.append(
                    {
                        "configuration": f"K={lobes},m={width},n={n}",
                        "lobes": lobes,
                        "samples_per_lobe": width,
                        "n": n,
                        "curvature_ratio": ratio,
                        "trial": trial,
                        "seed": seed,
                        "threshold": threshold,
                        "sample_radius": sample_radius,
                        "amplitude": amplitude,
                        "exact_knots": int(exact),
                        "joint_certificate": int(joint_certificate),
                        "knot_f1": knot_f1(knots, signal.knots, tolerance=0),
                        "localisation_error_widths": _localisation_error(
                            knots, signal.knots, width, n
                        ),
                        "baseline_mse": mse(estimate, signal.baseline),
                        "detail_mse": mse(detail_estimate, signal.detail),
                        "baseline_sup_error": baseline_sup,
                        "detail_sup_error": detail_sup,
                        "retained_knots": int(knots.size),
                        "compression_ratio": float(n / knots.size),
                        "elapsed_microseconds": elapsed_ns / 1000.0,
                    }
                )

    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["configuration"]), float(row["curvature_ratio"])), []
        ).append(row)
    aggregate: list[dict[str, object]] = []
    for (configuration, ratio), group in grouped.items():
        exact_count = sum(int(row["exact_knots"]) for row in group)
        joint_count = sum(int(row["joint_certificate"]) for row in group)
        exact_low, exact_high = _wilson(exact_count, len(group))
        joint_low, joint_high = _wilson(joint_count, len(group))
        aggregate.append(
            {
                "configuration": configuration,
                "curvature_ratio": ratio,
                "trials": len(group),
                "exact_recovery_probability": exact_count / len(group),
                "exact_wilson_low": exact_low,
                "exact_wilson_high": exact_high,
                "joint_certificate_probability": joint_count / len(group),
                "joint_wilson_low": joint_low,
                "joint_wilson_high": joint_high,
                "mean_knot_f1": float(np.mean([float(row["knot_f1"]) for row in group])),
                "mean_localisation_error_widths": float(
                    np.mean([float(row["localisation_error_widths"]) for row in group])
                ),
                "mean_baseline_mse": float(
                    np.mean([float(row["baseline_mse"]) for row in group])
                ),
                "median_compression_ratio": float(
                    np.median([float(row["compression_ratio"]) for row in group])
                ),
                "median_elapsed_microseconds": float(
                    np.median([float(row["elapsed_microseconds"]) for row in group])
                ),
            }
        )
    aggregate.sort(key=lambda row: (str(row["configuration"]), float(row["curvature_ratio"])))

    theorem_rows = [row for row in aggregate if float(row["curvature_ratio"]) > 2.0]
    summary = {
        "configurations": [f"K={lobes},m={width},n={lobes * width + 1}" for lobes, width in configurations],
        "curvature_ratios": list(args.ratio),
        "trials_per_cell": args.trials,
        "delta": args.delta,
        "theorem_boundary_strict": 2.0,
        "minimum_exact_probability_above_boundary": min(
            float(row["exact_recovery_probability"]) for row in theorem_rows
        ),
        "minimum_exact_wilson_low_above_boundary": min(
            float(row["exact_wilson_low"]) for row in theorem_rows
        ),
        "minimum_joint_probability_above_boundary": min(
            float(row["joint_certificate_probability"]) for row in theorem_rows
        ),
        "minimum_joint_wilson_low_above_boundary": min(
            float(row["joint_wilson_low"]) for row in theorem_rows
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "trials.csv", rows)
    _write_csv(args.output / "aggregate.csv", aggregate)
    with (args.output / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": "docs/recovery_phase_diagram_protocol.md",
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROJECT / "docs" / "recovery_phase_diagram_protocol.md"),
    }
    with (args.output / "metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
