"""Protocol P1: controlled batch-parallel HCRD runtime benchmark."""

from __future__ import annotations

import os

# Prevent compiled libraries inside feature extraction from creating hidden
# worker teams.  These variables must be set before importing NumPy.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import gc
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.datasets import load_cwru_drive_end  # noqa: E402
from hcrd.features import (  # noqa: E402
    normalise_window,
    representation_features_batch,
)

CONFIGURATIONS = (
    ("serial", 1),
    ("thread", 4),
    ("thread", 8),
    ("process", 2),
    ("process", 4),
    ("process", 8),
    ("process", 16),
)


def deterministic_windows(
    signal: np.ndarray, *, length: int, count: int
) -> list[np.ndarray]:
    available = signal.size // length
    if available < count:
        raise ValueError("record is too short for the frozen P1 workload")
    blocks = np.rint(np.linspace(0, available - 1, count)).astype(int)
    if np.unique(blocks).size != count:
        raise RuntimeError("window selection produced duplicate blocks")
    return [
        signal[block * length : (block + 1) * length].copy()
        for block in blocks
    ]


def array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def wait_for_idle(maximum_percent: float, timeout_seconds: int) -> list[float]:
    started = time.monotonic()
    recent: list[float] = []
    while time.monotonic() - started < timeout_seconds:
        value = float(psutil.cpu_percent(interval=1.0))
        recent.append(value)
        recent = recent[-5:]
        if len(recent) == 5 and statistics.mean(recent) <= maximum_percent:
            return recent
        if len(recent) == 5 and int(time.monotonic() - started) % 15 == 0:
            print(
                json.dumps(
                    {
                        "waiting_for_idle_cpu": round(statistics.mean(recent), 2),
                        "threshold": maximum_percent,
                    }
                ),
                flush=True,
            )
    raise RuntimeError(
        f"CPU did not remain below {maximum_percent}% within {timeout_seconds}s"
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "raw" / "cwru")
    parser.add_argument(
        "--reference", type=Path, default=PROJECT / "results" / "cwru_r1" / "features_hcrd.npy"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "parallel_runtime_p1"
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-background-cpu", type=float, default=20.0)
    parser.add_argument("--idle-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    protocol = PROJECT / "docs" / "parallel_runtime_protocol.md"
    if not protocol.exists():
        raise FileNotFoundError("the frozen P1 protocol is missing")
    manifest_path = args.data / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    windows: list[np.ndarray] = []
    for item in manifest:
        signal, _ = load_cwru_drive_end(args.data / f"{int(item['record_id'])}.mat")
        windows.extend(deterministic_windows(signal, length=2048, count=24))
    if len(windows) != 384:
        raise RuntimeError(f"expected 384 frozen windows, found {len(windows)}")

    idle_samples = wait_for_idle(args.max_background_cpu, args.idle_timeout)
    print(json.dumps({"idle_cpu_samples": idle_samples}), flush=True)

    # Compute once outside the timed region to establish an exact reference.
    serial_reference = representation_features_batch(
        windows, "hcrd", backend="serial", workers=1
    )
    digest = array_digest(serial_reference)
    previous_reference_equal: bool | None = None
    previous_reference_digest: str | None = None
    if args.reference.exists():
        previous = np.load(args.reference)
        previous_reference_equal = bool(np.array_equal(previous, serial_reference))
        previous_reference_digest = array_digest(previous)
        if not previous_reference_equal:
            raise RuntimeError("P1 features differ from the saved R1 HCRD features")

    # Short code/import warm-up.  Process startup is intentionally repeated and
    # timed in every full trial below.
    warmup = windows[:32]
    for backend, workers in (("serial", 1), ("thread", 4), ("process", 4)):
        candidate = representation_features_batch(
            warmup, "hcrd", backend=backend, workers=workers
        )
        if not np.array_equal(candidate, serial_reference[: len(warmup)]):
            raise RuntimeError(f"{backend} warm-up changed numerical output")

    rng = random.Random(20260824)
    rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        order = list(CONFIGURATIONS)
        rng.shuffle(order)
        for backend, workers in order:
            gc.collect()
            pre_cpu = float(psutil.cpu_percent(interval=0.5))
            started = time.perf_counter_ns()
            features = representation_features_batch(
                windows,
                "hcrd",
                backend=backend,
                workers=workers,
            )
            elapsed = (time.perf_counter_ns() - started) / 1e9
            trial_digest = array_digest(features)
            exact = bool(np.array_equal(features, serial_reference))
            if not exact or trial_digest != digest:
                raise RuntimeError(f"{backend}/{workers} changed numerical output")
            row = {
                "repetition": repetition,
                "backend": backend,
                "workers": workers,
                "seconds": elapsed,
                "windows_per_second": len(windows) / elapsed,
                "pre_cpu_percent": pre_cpu,
                "digest": trial_digest,
                "bitwise_equal": exact,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    serial_times = [
        float(row["seconds"])
        for row in rows
        if row["backend"] == "serial" and row["workers"] == 1
    ]
    serial_median = statistics.median(serial_times)
    summary: list[dict[str, object]] = []
    for backend, workers in CONFIGURATIONS:
        selected = [
            float(row["seconds"])
            for row in rows
            if row["backend"] == backend and row["workers"] == workers
        ]
        median = statistics.median(selected)
        summary.append(
            {
                "backend": backend,
                "workers": workers,
                "median_seconds": median,
                "q1_seconds": percentile(selected, 0.25),
                "q3_seconds": percentile(selected, 0.75),
                "median_windows_per_second": len(windows) / median,
                "speedup_vs_serial": serial_median / median,
                "parallel_efficiency": (
                    serial_median / median / workers if backend != "serial" else 1.0
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    import csv

    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "P1",
        "exploratory_scaling_characterization": True,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "windows": len(windows),
        "samples_per_window": 2048,
        "features": int(serial_reference.shape[1]),
        "repetitions": args.repetitions,
        "initial_idle_cpu_samples": idle_samples,
        "reference_digest": digest,
        "previous_reference_exists": args.reference.exists(),
        "previous_reference_equal": previous_reference_equal,
        "previous_reference_digest": previous_reference_digest,
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
