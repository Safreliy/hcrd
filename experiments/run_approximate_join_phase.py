"""Phase experiment for approximate sampled chord-lobe joins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import discrete_curvature, find_convexity_knots  # noqa: E402
from hcrd.metrics import knot_f1, mse  # noqa: E402
from hcrd.recovery import (  # noqa: E402
    alternating_parabolic_chord_lobes,
    amplitudes_for_recovery_ratios,
    approximate_join_tolerance,
    finite_sample_recovery_thresholds,
)


CONFIGURATIONS = ((4, 8), (8, 8), (8, 16))
JOIN_RATIOS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
CURVATURE_RATIOS = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)
    )
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def _localisation_error(
    estimate: np.ndarray, truth: np.ndarray, width: int, n: int
) -> float:
    estimate_inner = estimate[1:-1]
    truth_inner = truth[1:-1]
    if estimate_inner.size == 0 or truth_inner.size == 0:
        return float(n / width)
    truth_to_estimate = max(
        float(np.min(np.abs(estimate_inner - value))) for value in truth_inner
    )
    estimate_to_truth = max(
        float(np.min(np.abs(truth_inner - value))) for value in estimate_inner
    )
    return max(truth_to_estimate, estimate_to_truth) / width


def _evaluate(
    signal, tolerance: float, sample_radius: float
) -> tuple[np.ndarray, dict[str, object]]:
    knots = find_convexity_knots(
        signal.observed,
        signal.x,
        atol=tolerance,
        rtol=0.0,
        boundary_rule="minimum_curvature",
    )
    baseline = np.interp(signal.x, signal.x[knots], signal.observed[knots])
    detail = signal.observed - baseline
    baseline_sup = float(np.max(np.abs(baseline - signal.baseline)))
    detail_sup = float(np.max(np.abs(detail - signal.detail)))
    return knots, {
        "baseline_mse": mse(baseline, signal.baseline),
        "detail_mse": mse(detail, signal.detail),
        "baseline_sup": baseline_sup,
        "detail_sup": detail_sup,
        "reconstruction_bounds": int(
            baseline_sup <= sample_radius + 1e-12
            and detail_sup <= 2.0 * sample_radius + 1e-12
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "approximate_join_phase_r1",
    )
    args = parser.parse_args()
    if args.trials < 20 or args.sigma <= 0 or not 0 < args.delta < 1:
        raise SystemExit("require trials>=20, sigma>0, and delta in (0,1)")

    rows: list[dict[str, object]] = []
    for config_index, (lobes, width) in enumerate(CONFIGURATIONS):
        n = lobes * width + 1
        tau, sample_radius = finite_sample_recovery_thresholds(
            n, args.sigma, delta=args.delta
        )
        for join_index, join_ratio in enumerate(JOIN_RATIOS):
            eta = join_ratio * tau
            aware_tolerance = approximate_join_tolerance(eta, tau)
            for curvature_index, curvature_ratio in enumerate(CURVATURE_RATIOS):
                amplitudes = amplitudes_for_recovery_ratios(
                    lobes=lobes,
                    samples_per_lobe=width,
                    curvature_ratio=curvature_ratio,
                    join_ratio=join_ratio,
                    curvature_threshold=tau,
                )
                certified = curvature_ratio > join_ratio + 2.0
                for trial in range(args.trials):
                    seed = (
                        args.seed
                        + 100_000_000 * config_index
                        + 10_000_000 * join_index
                        + 1_000_000 * curvature_index
                        + trial
                    )
                    signal = alternating_parabolic_chord_lobes(
                        seed=seed,
                        lobes=lobes,
                        samples_per_lobe=width,
                        amplitudes=amplitudes,
                        noise_sigma=args.sigma,
                    )
                    aware_knots, aware = _evaluate(
                        signal, aware_tolerance, sample_radius
                    )
                    naive_knots, naive = _evaluate(signal, tau, sample_radius)
                    curvature_event = bool(
                        np.max(np.abs(discrete_curvature(signal.noise, signal.x)))
                        <= tau + 1e-12
                    )
                    sample_event = bool(
                        np.max(np.abs(signal.noise)) <= sample_radius + 1e-12
                    )
                    aware_exact = bool(np.array_equal(aware_knots, signal.knots))
                    naive_exact = bool(np.array_equal(naive_knots, signal.knots))
                    common = {
                        "configuration": f"K={lobes},m={width},n={n}",
                        "lobes": lobes,
                        "samples_per_lobe": width,
                        "n": n,
                        "join_ratio": join_ratio,
                        "curvature_ratio": curvature_ratio,
                        "certified_region": int(certified),
                        "trial": trial,
                        "seed": seed,
                        "tau": tau,
                        "eta": eta,
                        "aware_tolerance": aware_tolerance,
                        "minimum_amplitude": float(np.min(amplitudes)),
                        "maximum_amplitude": float(np.max(amplitudes)),
                        "curvature_event": int(curvature_event),
                        "sample_event": int(sample_event),
                        "aware_exact": int(aware_exact),
                        "aware_joint_certificate": int(
                            aware_exact
                            and curvature_event
                            and sample_event
                            and bool(aware["reconstruction_bounds"])
                        ),
                        "aware_knot_f1": knot_f1(
                            aware_knots, signal.knots, tolerance=0
                        ),
                        "aware_localisation_error_widths": _localisation_error(
                            aware_knots, signal.knots, width, n
                        ),
                        "aware_retained_knots": int(aware_knots.size),
                        "naive_exact": int(naive_exact),
                        "naive_knot_f1": knot_f1(
                            naive_knots, signal.knots, tolerance=0
                        ),
                        "naive_localisation_error_widths": _localisation_error(
                            naive_knots, signal.knots, width, n
                        ),
                        "naive_retained_knots": int(naive_knots.size),
                        "theorem_implication_violation": int(
                            certified and curvature_event and not aware_exact
                        ),
                        "noise_only_theorem_implication_violation": int(
                            certified and curvature_event and not naive_exact
                        ),
                    }
                    rows.append({**common, **{f"aware_{k}": v for k, v in aware.items()}, **{f"naive_{k}": v for k, v in naive.items()}})

    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["configuration"]),
                float(row["join_ratio"]),
                float(row["curvature_ratio"]),
            ),
            [],
        ).append(row)
    aggregate: list[dict[str, object]] = []
    for (configuration, join_ratio, curvature_ratio), group in grouped.items():
        aware_count = sum(int(row["aware_exact"]) for row in group)
        naive_count = sum(int(row["naive_exact"]) for row in group)
        aware_low, aware_high = _wilson(aware_count, len(group))
        naive_low, naive_high = _wilson(naive_count, len(group))
        aggregate.append(
            {
                "configuration": configuration,
                "join_ratio": join_ratio,
                "curvature_ratio": curvature_ratio,
                "certified_region": int(curvature_ratio > join_ratio + 2.0),
                "trials": len(group),
                "aware_exact_probability": aware_count / len(group),
                "aware_wilson_low": aware_low,
                "aware_wilson_high": aware_high,
                "naive_exact_probability": naive_count / len(group),
                "naive_wilson_low": naive_low,
                "naive_wilson_high": naive_high,
                "aware_minus_naive_exact": (aware_count - naive_count) / len(group),
                "aware_mean_knot_f1": float(
                    np.mean([float(row["aware_knot_f1"]) for row in group])
                ),
                "naive_mean_knot_f1": float(
                    np.mean([float(row["naive_knot_f1"]) for row in group])
                ),
                "aware_mean_localisation_error_widths": float(
                    np.mean(
                        [float(row["aware_localisation_error_widths"]) for row in group]
                    )
                ),
                "naive_mean_localisation_error_widths": float(
                    np.mean(
                        [float(row["naive_localisation_error_widths"]) for row in group]
                    )
                ),
                "theorem_implication_violations": sum(
                    int(row["theorem_implication_violation"]) for row in group
                ),
                "noise_only_theorem_implication_violations": sum(
                    int(row["noise_only_theorem_implication_violation"])
                    for row in group
                ),
            }
        )
    aggregate.sort(
        key=lambda row: (
            str(row["configuration"]),
            float(row["join_ratio"]),
            float(row["curvature_ratio"]),
        )
    )
    certified_rows = [row for row in aggregate if int(row["certified_region"])]
    positive_join_rows = [row for row in aggregate if float(row["join_ratio"]) > 0]
    summary = {
        "draws": len(rows),
        "configurations": [
            f"K={lobes},m={width},n={lobes * width + 1}"
            for lobes, width in CONFIGURATIONS
        ],
        "join_ratios": list(JOIN_RATIOS),
        "curvature_ratios": list(CURVATURE_RATIOS),
        "trials_per_configuration_cell": args.trials,
        "strict_boundary": "curvature_ratio > join_ratio + 2",
        "minimum_aware_exact_probability_certified": min(
            float(row["aware_exact_probability"]) for row in certified_rows
        ),
        "minimum_aware_wilson_low_certified": min(
            float(row["aware_wilson_low"]) for row in certified_rows
        ),
        "theorem_implication_violations": sum(
            int(row["theorem_implication_violations"]) for row in aggregate
        ),
        "noise_only_theorem_implication_violations": sum(
            int(row["noise_only_theorem_implication_violations"])
            for row in aggregate
        ),
        "maximum_aware_minus_naive_positive_eta": max(
            float(row["aware_minus_naive_exact"]) for row in positive_join_rows
        ),
        "minimum_aware_minus_naive_positive_eta": min(
            float(row["aware_minus_naive_exact"]) for row in positive_join_rows
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "trials.csv", rows)
    _write_csv(args.output / "aggregate.csv", aggregate)
    _write_json(args.output / "summary.json", summary)
    protocol = PROJECT / "docs" / "approximate_join_phase_protocol.md"
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": protocol.relative_to(PROJECT).as_posix(),
        "protocol_sha256": _sha256(protocol),
        "runner_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.output / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
