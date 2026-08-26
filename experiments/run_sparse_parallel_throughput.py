"""Protocol P3: load-gated sparse HCRD process throughput."""

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
from hcrd.parallel import decompose_sparse_batch  # noqa: E402

from experiments.run_parallel_runtime import deterministic_windows  # noqa: E402
from experiments.run_sparse_runtime import hierarchy_digest, percentile  # noqa: E402
from experiments.run_sparse_runtime_recalculation import strict_idle_gate  # noqa: E402

CONFIGURATIONS = (("serial", 1), ("process", 2), ("process", 4), ("process", 8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "raw" / "cwru")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "sparse_parallel_p3"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-multiplier", type=int, default=10)
    parser.add_argument("--max-background-cpu", type=float, default=20.0)
    parser.add_argument("--idle-timeout", type=int, default=300)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")
    if args.batch_multiplier != 10:
        raise ValueError("P3 freezes the batch multiplier at ten")

    protocol = PROJECT / "docs" / "sparse_parallel_throughput_protocol.md"
    if not protocol.exists():
        raise FileNotFoundError("frozen P3 protocol is missing")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    base_windows: list[np.ndarray] = []
    for item in manifest:
        signal, _ = load_cwru_drive_end(args.data / f"{int(item['record_id'])}.mat")
        base_windows.extend(
            normalise_window(window)
            for window in deterministic_windows(signal, length=2048, count=24)
        )
    if len(base_windows) != 384:
        raise RuntimeError(f"expected 384 frozen windows, found {len(base_windows)}")
    windows = base_windows * args.batch_multiplier

    reference = decompose_sparse_batch(windows, backend="serial", workers=1)
    reference_digest = hierarchy_digest(reference)
    for hierarchy, window in zip(reference, windows, strict=True):
        if hierarchy.stored_knot_count > 2 * (window.size - 1) + hierarchy.depth:
            raise RuntimeError("sparse hierarchy violates the storage bound")
    del reference
    gc.collect()

    warmup = base_windows[:64]
    warmup_digest = hierarchy_digest(
        decompose_sparse_batch(warmup, backend="serial", workers=1)
    )
    for backend, workers in CONFIGURATIONS:
        candidate = decompose_sparse_batch(warmup, backend=backend, workers=workers)
        if hierarchy_digest(candidate) != warmup_digest:
            raise RuntimeError(f"{backend}/{workers} warm-up differs")
        del candidate

    rng = random.Random(20260827)
    rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        order = list(CONFIGURATIONS)
        rng.shuffle(order)
        for backend, workers in order:
            gc.collect()
            load_samples = strict_idle_gate(
                args.max_background_cpu, args.idle_timeout
            )
            started = time.perf_counter_ns()
            result = decompose_sparse_batch(windows, backend=backend, workers=workers)
            seconds = (time.perf_counter_ns() - started) / 1e9
            digest = hierarchy_digest(result)
            if digest != reference_digest:
                raise RuntimeError(f"{backend}/{workers} changed knots")
            row = {
                "repetition": repetition,
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

    serial = statistics.median(
        float(row["seconds"]) for row in rows if row["backend"] == "serial"
    )
    summary: list[dict[str, object]] = []
    for backend, workers in CONFIGURATIONS:
        values = [
            float(row["seconds"])
            for row in rows
            if row["backend"] == backend and int(row["workers"]) == workers
        ]
        median = statistics.median(values)
        summary.append(
            {
                "backend": backend,
                "workers": workers,
                "median_seconds": median,
                "q1_seconds": percentile(values, 0.25),
                "q3_seconds": percentile(values, 0.75),
                "median_signals_per_second": len(windows) / median,
                "speedup_vs_serial": serial / median,
                "parallel_efficiency": serial / median / workers,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "P3",
        "protocol_path": str(protocol),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "windows": len(windows),
        "base_windows": len(base_windows),
        "batch_multiplier": args.batch_multiplier,
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
