"""Frozen E3 comparison against oracle slow-tail Iterative Filtering."""

from __future__ import annotations

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "results" / "iterative_filtering_e3"
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import decompose  # noqa: E402
from hcrd.external_baselines import iterative_filtering_slow_tail_path  # noqa: E402
from hcrd.metrics import exact_sign_test, mse, paired_bootstrap_ci  # noqa: E402
from hcrd.robust import robust_decompose  # noqa: E402
from hcrd.signals import alternating_chord_lobes  # noqa: E402

SEED = 20261202
NOISE_LEVELS = (0.0, 0.03, 0.10)
TRIALS = 50
WORKERS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle(candidates: list[np.ndarray], target: np.ndarray) -> tuple[int, np.ndarray]:
    index = min(range(len(candidates)), key=lambda item: mse(candidates[item], target))
    return index + 1, candidates[index]


def _run_trial(task: tuple[int, float, int]) -> list[dict[str, object]]:
    trial, noise_sigma, noise_index = task
    signal_seed = SEED + 10_000 * noise_index + trial
    signal = alternating_chord_lobes(
        seed=signal_seed,
        noise_sigma=noise_sigma,
        piecewise_baseline=False,
        amplitude_variation=False,
    )
    centred = decompose(signal.observed, signal.x).levels[0].baseline
    thresholded = robust_decompose(
        signal.observed, signal.x, z_score=3.5, max_levels=1
    ).decomposition.levels[0].baseline
    started = time.perf_counter()
    candidates, reconstruction_error = iterative_filtering_slow_tail_path(
        signal.observed, maximum_tail_components=4
    )
    if_seconds = time.perf_counter() - started
    tail_count, oracle = _oracle(candidates, signal.baseline)
    estimates = (
        ("hcrd_centered", "none", centred),
        ("hcrd_thresholded", "z=3.5", thresholded),
        ("if_final", "tail=1", candidates[0]),
        ("if_oracle_tail", f"tail={tail_count}", oracle),
    )
    return [
        {
            "noise_sigma": noise_sigma,
            "trial": trial,
            "signal_seed": signal_seed,
            "method": method,
            "parameter": parameter,
            "baseline_mse": mse(estimate, signal.baseline),
            "if_reconstruction_error": reconstruction_error,
            "if_seconds": if_seconds,
        }
        for method, parameter, estimate in estimates
    ]


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    protocol = PROJECT / "docs" / "iterative_filtering_confirmation_protocol.md"
    if not protocol.exists():
        raise FileNotFoundError("frozen E3 protocol is missing")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("E3 must run under the frozen Python 3.12 environment")
    tasks = [
        (trial, noise_sigma, noise_index)
        for noise_index, noise_sigma in enumerate(NOISE_LEVELS)
        for trial in range(TRIALS)
    ]
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for completed, trial_rows in enumerate(
            executor.map(_run_trial, tasks, chunksize=1), start=1
        ):
            rows.extend(trial_rows)
            if completed % 10 == 0:
                print(json.dumps({"completed": completed, "total": len(tasks)}), flush=True)
    elapsed = time.perf_counter() - started
    rows.sort(
        key=lambda row: (
            float(row["noise_sigma"]), int(row["trial"]), str(row["method"])
        )
    )

    methods = sorted({str(row["method"]) for row in rows})
    summary: list[dict[str, object]] = []
    for noise_sigma in NOISE_LEVELS:
        for method in methods:
            selected = [
                float(row["baseline_mse"])
                for row in rows
                if float(row["noise_sigma"]) == noise_sigma and row["method"] == method
            ]
            summary.append(
                {
                    "noise_sigma": noise_sigma,
                    "method": method,
                    "trials": len(selected),
                    "mean_mse": float(np.mean(selected)),
                    "median_mse": float(np.median(selected)),
                }
            )

    comparisons: list[dict[str, object]] = []
    for comparison_index, noise_sigma in enumerate(NOISE_LEVELS):
        reference_method = "hcrd_centered" if noise_sigma == 0.0 else "hcrd_thresholded"
        reference = np.asarray(
            [
                float(row["baseline_mse"])
                for row in rows
                if float(row["noise_sigma"]) == noise_sigma
                and row["method"] == reference_method
            ]
        )
        competitor = np.asarray(
            [
                float(row["baseline_mse"])
                for row in rows
                if float(row["noise_sigma"]) == noise_sigma
                and row["method"] == "if_oracle_tail"
            ]
        )
        differences = reference - competitor
        differences[np.abs(differences) < 1e-12] = 0.0
        lower, upper = paired_bootstrap_ci(
            differences, samples=20_000, seed=SEED + comparison_index
        )
        comparisons.append(
            {
                "noise_sigma": noise_sigma,
                "reference": reference_method,
                "competitor": "if_oracle_tail",
                "mean_difference_reference_minus_competitor": float(np.mean(differences)),
                "bootstrap_95_low": lower,
                "bootstrap_95_high": upper,
                "reference_win_rate": float(np.mean(differences < 0)),
                "ties": int(np.sum(differences == 0)),
                "exact_sign_p": exact_sign_test(differences),
            }
        )
    adjusted = _holm([float(row["exact_sign_p"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["superiority_supported"] = bool(
            float(row["bootstrap_95_high"]) < 0.0 and value < 0.05
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUTPUT / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    metadata = {
        "protocol": str(protocol),
        "protocol_sha256": _sha256(protocol),
        "seed": SEED,
        "noise_levels": NOISE_LEVELS,
        "trials_per_condition": TRIALS,
        "workers": WORKERS,
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "iterativefiltering": importlib.metadata.version("iterativefiltering"),
        "pyfftw": importlib.metadata.version("pyfftw"),
        "maximum_reconstruction_error": max(
            float(row["if_reconstruction_error"]) for row in rows
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "comparisons": comparisons, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
