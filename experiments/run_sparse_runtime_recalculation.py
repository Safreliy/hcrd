"""Protocol P2R: per-trial-load-gated sparse HCRD recalculation."""

from __future__ import annotations

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import csv
import gc
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
sys.path.insert(0, str(PROJECT))

from hcrd.datasets import load_cwru_drive_end  # noqa: E402
from hcrd.features import normalise_window  # noqa: E402
from hcrd.parallel import decompose_batch, decompose_sparse_batch  # noqa: E402

from experiments.run_parallel_runtime import deterministic_windows  # noqa: E402
from experiments.run_sparse_runtime import hierarchy_digest, percentile  # noqa: E402

CONFIGURATIONS = (
    ("dense", "serial", 1),
    ("sparse", "serial", 1),
    ("sparse", "process", 2),
    ("sparse", "process", 4),
    ("sparse", "process", 8),
)


def strict_idle_gate(maximum_percent: float, timeout_seconds: int) -> list[float]:
    """Require five consecutive one-second samples below the threshold."""
    started = time.monotonic()
    recent: list[float] = []
    last_notice = -1
    while time.monotonic() - started < timeout_seconds:
        value = float(psutil.cpu_percent(interval=1.0))
        recent.append(value)
        recent = recent[-5:]
        if len(recent) == 5 and max(recent) <= maximum_percent:
            return recent
        elapsed_bucket = int((time.monotonic() - started) // 15)
        if elapsed_bucket > last_notice and elapsed_bucket > 0:
            print(
                json.dumps(
                    {
                        "waiting_for_strict_idle_cpu": recent,
                        "threshold": maximum_percent,
                    }
                ),
                flush=True,
            )
            last_notice = elapsed_bucket
    raise RuntimeError(
        f"CPU did not give five samples <= {maximum_percent}% "
        f"within {timeout_seconds}s"
    )


def run_configuration(
    windows: list[np.ndarray], representation: str, backend: str, workers: int
) -> tuple[object, ...]:
    if representation == "dense":
        return decompose_batch(windows, backend="serial", workers=1)
    return decompose_sparse_batch(
        windows,
        backend=backend,  # type: ignore[arg-type]
        workers=workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "raw" / "cwru")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "sparse_runtime_p2r"
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-background-cpu", type=float, default=20.0)
    parser.add_argument("--idle-timeout", type=int, default=300)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    protocol = PROJECT / "docs" / "sparse_runtime_recalculation_protocol.md"
    if not protocol.exists():
        raise FileNotFoundError("frozen P2R protocol is missing")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    windows: list[np.ndarray] = []
    for item in manifest:
        signal, _ = load_cwru_drive_end(args.data / f"{int(item['record_id'])}.mat")
        windows.extend(
            normalise_window(window)
            for window in deterministic_windows(signal, length=2048, count=24)
        )
    if len(windows) != 384:
        raise RuntimeError(f"expected 384 frozen windows, found {len(windows)}")

    dense_reference = decompose_batch(windows, backend="serial", workers=1)
    sparse_reference = decompose_sparse_batch(windows, backend="serial", workers=1)
    reference_digest = hierarchy_digest(dense_reference)
    if hierarchy_digest(sparse_reference) != reference_digest:
        raise RuntimeError("sparse knot hierarchies differ from dense output")
    for dense, sparse, window in zip(
        dense_reference, sparse_reference, windows, strict=True
    ):
        if dense.depth != sparse.depth:
            raise RuntimeError("dense/sparse hierarchy depths differ")
        for left, right in zip(dense.knot_sets, sparse.knot_sets, strict=True):
            if not np.array_equal(left, right):
                raise RuntimeError("dense/sparse knot sets differ")
        if sparse.stored_knot_count > 2 * (window.size - 1) + sparse.depth:
            raise RuntimeError("sparse hierarchy violates the storage bound")
    del dense_reference, sparse_reference
    gc.collect()

    warmup = windows[:32]
    warmup_digest = hierarchy_digest(
        decompose_sparse_batch(warmup, backend="serial", workers=1)
    )
    for representation, backend, workers in CONFIGURATIONS:
        candidate = run_configuration(warmup, representation, backend, workers)
        if hierarchy_digest(candidate) != warmup_digest:
            raise RuntimeError(f"{representation}/{backend}/{workers} warm-up differs")
        del candidate

    rng = random.Random(20260826)
    rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        order = list(CONFIGURATIONS)
        rng.shuffle(order)
        for representation, backend, workers in order:
            gc.collect()
            load_samples = strict_idle_gate(
                args.max_background_cpu, args.idle_timeout
            )
            started = time.perf_counter_ns()
            result = run_configuration(windows, representation, backend, workers)
            seconds = (time.perf_counter_ns() - started) / 1e9
            digest = hierarchy_digest(result)
            if digest != reference_digest:
                raise RuntimeError(f"{representation}/{backend}/{workers} changed knots")
            row = {
                "repetition": repetition,
                "representation": representation,
                "backend": backend,
                "workers": workers,
                "seconds": seconds,
                "signals_per_second": len(windows) / seconds,
                "pre_cpu_samples": json.dumps(load_samples),
                "pre_cpu_max": max(load_samples),
                "digest": digest,
                "exact_knots": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del result

    dense_serial = statistics.median(
        float(row["seconds"])
        for row in rows
        if row["representation"] == "dense"
    )
    sparse_serial = statistics.median(
        float(row["seconds"])
        for row in rows
        if row["representation"] == "sparse" and row["backend"] == "serial"
    )
    summary: list[dict[str, object]] = []
    for representation, backend, workers in CONFIGURATIONS:
        values = [
            float(row["seconds"])
            for row in rows
            if row["representation"] == representation
            and row["backend"] == backend
            and int(row["workers"]) == workers
        ]
        median = statistics.median(values)
        summary.append(
            {
                "representation": representation,
                "backend": backend,
                "workers": workers,
                "median_seconds": median,
                "q1_seconds": percentile(values, 0.25),
                "q3_seconds": percentile(values, 0.75),
                "median_signals_per_second": len(windows) / median,
                "speedup_vs_dense_serial": dense_serial / median,
                "speedup_vs_sparse_serial": sparse_serial / median,
                "parallel_efficiency_vs_sparse_serial": (
                    sparse_serial / median / workers
                    if representation == "sparse" and backend != "serial"
                    else 1.0
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "P2R",
        "protocol_path": str(protocol),
        "replaces_timing_claims_from": "P2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "windows": len(windows),
        "samples_per_window": windows[0].size,
        "repetitions": args.repetitions,
        "load_gate": {
            "consecutive_samples": 5,
            "sample_seconds": 1,
            "maximum_each_percent": args.max_background_cpu,
            "applied_before_every_trial": True,
        },
        "reference_knot_digest": reference_digest,
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
