"""Preregistered chord-lobe recovery benchmark.

Comparators are tuned by an oracle against the latent baseline on each trial.
This is intentionally favourable to the comparison methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.baselines import (  # noqa: E402
    affine_trend,
    fourier_lowpass,
    gaussian_smooth,
    interpolate_knots,
    moving_average,
    rdp_indices,
)
from hcrd.core import decompose  # noqa: E402
from hcrd.metrics import (  # noqa: E402
    exact_sign_test,
    knot_f1,
    mse,
    nmse,
    paired_bootstrap_ci,
    scaled_mse,
)
from hcrd.robust import (  # noqa: E402
    adaptive_gaussian_guided_decompose,
    gaussian_guided_decompose,
    robust_decompose,
)
from hcrd.signals import alternating_chord_lobes  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _best(candidates: list[tuple[str, np.ndarray]], target: np.ndarray) -> tuple[str, np.ndarray]:
    return min(candidates, key=lambda item: nmse(item[1], target))


def _oracle_methods(signal, max_knots: int) -> dict[str, tuple[str, np.ndarray]]:
    n = signal.observed.size
    windows = [value for value in (5, 9, 17, 33, 65, 129) if value <= n]
    gaussians = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    bins = sorted(set([1, 2, 3, 5, 8, 13, 21, 34, min(55, n // 2 + 1)]))

    methods: dict[str, tuple[str, np.ndarray]] = {}
    methods["affine_ls"] = ("degree=1", affine_trend(signal.observed, signal.x))
    methods["moving_average_oracle"] = _best(
        [(f"window={window}", moving_average(signal.observed, window)) for window in windows],
        signal.baseline,
    )
    methods["gaussian_oracle"] = _best(
        [(f"sigma={sigma:g}", gaussian_smooth(signal.observed, sigma)) for sigma in gaussians],
        signal.baseline,
    )
    methods["fourier_oracle"] = _best(
        [
            (f"bins={retained}", fourier_lowpass(signal.observed, retained))
            for retained in bins
            if retained <= n // 2 + 1
        ],
        signal.baseline,
    )
    amplitude = max(1e-12, float(np.ptp(signal.observed)))
    epsilons = amplitude * np.geomspace(1e-3, 1.0, 30)
    rdp_candidates: list[tuple[str, np.ndarray]] = []
    for epsilon in epsilons:
        knots = rdp_indices(signal.x, signal.observed, float(epsilon))
        if knots.size <= max_knots:
            rdp_candidates.append(
                (f"epsilon={epsilon:.8g};knots={knots.size}", interpolate_knots(signal.x, signal.observed, knots))
            )
    if not rdp_candidates:
        knots = rdp_indices(signal.x, signal.observed, float(epsilons[-1]))
        rdp_candidates.append(
            (f"epsilon={epsilons[-1]:.8g};knots={knots.size}", interpolate_knots(signal.x, signal.observed, knots))
        )
    methods["rdp_oracle_budget"] = _best(rdp_candidates, signal.baseline)
    return methods


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.03, 0.10])
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--suite", choices=("exact", "variable", "all"), default="all")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "synthetic_v03")
    args = parser.parse_args()
    if args.trials < 2:
        raise SystemExit("at least two trials are required")

    rows: list[dict[str, object]] = []
    suites = ("exact", "variable") if args.suite == "all" else (args.suite,)
    for suite in suites:
      for noise_sigma in args.noise:
        for trial in range(args.trials):
            suite_offset = 0 if suite == "exact" else 100_000_000
            seed = args.seed + suite_offset + 10_000 * round(noise_sigma * 1000) + trial
            signal = alternating_chord_lobes(
                seed=seed,
                noise_sigma=noise_sigma,
                piecewise_baseline=suite == "variable",
                amplitude_variation=suite == "variable",
            )

            centred = decompose(signal.observed, signal.x, boundary_rule="minimum_curvature")
            legacy = decompose(signal.observed, signal.x, boundary_rule="legacy")
            robust = robust_decompose(signal.observed, signal.x, z_score=3.5)
            decompositions = {
                "hcrd_centered": centred,
                "hcrd_legacy": legacy,
                "hcrd_robust": robust.decomposition,
            }
            for method, result in decompositions.items():
                estimate = result.levels[0].baseline
                detail = signal.observed - estimate
                rows.append(
                    {
                        "trial": trial,
                        "suite": suite,
                        "seed": seed,
                        "noise_sigma": noise_sigma,
                        "method": method,
                        "parameter": "z=3.5" if method == "hcrd_robust" else "none",
                        "baseline_mse": mse(estimate, signal.baseline),
                        "baseline_nmse": scaled_mse(estimate, signal.baseline, signal.observed),
                        "detail_mse": mse(detail, signal.detail),
                        "detail_nmse": scaled_mse(detail, signal.detail, signal.observed),
                        "knot_f1": knot_f1(result.levels[0].knots, signal.knots, tolerance=1),
                        "knots": int(result.levels[0].knots.size),
                    }
                )

            guided_candidates = []
            for smoothing_sigma in (1.0, 2.0, 4.0, 8.0):
                guided = gaussian_guided_decompose(
                    signal.observed, signal.x, smoothing_sigma=smoothing_sigma
                )
                guided_candidates.append(
                    (
                        nmse(guided.decomposition.levels[0].baseline, signal.baseline),
                        smoothing_sigma,
                        guided,
                    )
                )
            _, best_sigma, best_guided = min(guided_candidates, key=lambda item: item[0])
            for method, guided in (
                ("hcrd_guided_fixed", gaussian_guided_decompose(signal.observed, signal.x, smoothing_sigma=2.0)),
                ("hcrd_guided_oracle", best_guided),
                (
                    "hcrd_guided_adaptive",
                    adaptive_gaussian_guided_decompose(signal.observed, signal.x).guided,
                ),
            ):
                estimate = guided.decomposition.levels[0].baseline
                detail = signal.observed - estimate
                rows.append(
                    {
                        "trial": trial,
                        "suite": suite,
                        "seed": seed,
                        "noise_sigma": noise_sigma,
                        "method": method,
                        "parameter": (
                            "sigma=2"
                            if method == "hcrd_guided_fixed"
                            else f"sigma={best_sigma:g}"
                            if method == "hcrd_guided_oracle"
                            else f"sigma={guided.smoothing_sigma:g};adaptive"
                        ),
                        "baseline_mse": mse(estimate, signal.baseline),
                        "baseline_nmse": scaled_mse(estimate, signal.baseline, signal.observed),
                        "detail_mse": mse(detail, signal.detail),
                        "detail_nmse": scaled_mse(detail, signal.detail, signal.observed),
                        "knot_f1": knot_f1(
                            guided.decomposition.levels[0].knots, signal.knots, tolerance=1
                        ),
                        "knots": int(guided.decomposition.levels[0].knots.size),
                    }
                )

            max_knots = int(centred.levels[0].knots.size)
            for method, (parameter, estimate) in _oracle_methods(signal, max_knots).items():
                detail = signal.observed - estimate
                rows.append(
                    {
                        "trial": trial,
                        "suite": suite,
                        "seed": seed,
                        "noise_sigma": noise_sigma,
                        "method": method,
                        "parameter": parameter,
                        "baseline_mse": mse(estimate, signal.baseline),
                        "baseline_nmse": scaled_mse(estimate, signal.baseline, signal.observed),
                        "detail_mse": mse(detail, signal.detail),
                        "detail_nmse": scaled_mse(detail, signal.detail, signal.observed),
                        "knot_f1": "",
                        "knots": "",
                    }
                )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "trials.csv", rows)

    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["suite"]), float(row["noise_sigma"]), str(row["method"])), []
        ).append(row)
    aggregate = []
    for (suite, noise, method), group in sorted(grouped.items()):
        losses = np.array([float(row["baseline_mse"]) for row in group])
        aggregate.append(
            {
                "suite": suite,
                "noise_sigma": noise,
                "method": method,
                "mean_baseline_mse": float(np.mean(losses)),
                "median_baseline_mse": float(np.median(losses)),
                "std_baseline_mse": float(np.std(losses, ddof=1)),
            }
        )
    _write_csv(output / "aggregate.csv", aggregate)

    primary = [
        row
        for row in rows
        if row["suite"] == "exact" and float(row["noise_sigma"]) == 0.0
    ]
    by_method = {
        method: sorted(
            [row for row in primary if row["method"] == method], key=lambda row: int(row["trial"])
        )
        for method in sorted({str(row["method"]) for row in primary})
    }
    reference = np.array([float(row["baseline_mse"]) for row in by_method["hcrd_centered"]])
    comparisons = []
    for method, method_rows in by_method.items():
        if method == "hcrd_centered":
            continue
        competitor = np.array([float(row["baseline_mse"]) for row in method_rows])
        differences = reference - competitor
        differences[np.abs(differences) < 1e-12] = 0.0
        low, high = paired_bootstrap_ci(differences, seed=args.seed)
        comparisons.append(
            {
                "competitor": method,
                "mean_difference_hcrd_minus_competitor": float(np.mean(differences)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "hcrd_win_rate": float(np.mean(differences < 0)),
                "exact_sign_p": exact_sign_test(differences),
            }
        )
    ordered = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["exact_sign_p"])
    running = 0.0
    total = len(comparisons)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(comparisons[index]["exact_sign_p"]))
        running = max(running, adjusted)
        comparisons[index]["holm_adjusted_p"] = running
    _write_csv(output / "primary_comparisons.csv", comparisons)

    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": "docs/experiment_protocol.md version 0.3; see docs/protocol_amendments.md",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
