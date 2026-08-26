"""Protocol P2: dense versus sparse HCRD hierarchy construction."""

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
sys.path.insert(0, str(PROJECT))

from hcrd.datasets import load_cwru_drive_end
from hcrd.features import normalise_window
from hcrd.parallel import decompose_batch, decompose_sparse_batch

from experiments.run_parallel_runtime import deterministic_windows, wait_for_idle
CONFIGURATIONS = (
    ("dense", "serial", 1),
    ("sparse", "serial", 1),
    ("sparse", "thread", 4),
    ("sparse", "thread", 8),
    ("sparse", "process", 2),
    ("sparse", "process", 4),
    ("sparse", "process", 8),
    ("sparse", "process", 16),
)


def hierarchy_digest(results: tuple[object, ...]) -> str:
    digest = hashlib.sha256()
    for result in results:
        knot_sets = result.knot_sets
        digest.update(np.asarray([len(knot_sets)], dtype="<i8").tobytes())
        for knots in knot_sets:
            canonical = np.asarray(knots, dtype="<i8", order="C")
            digest.update(np.asarray([canonical.size], dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _run(
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
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "sparse_runtime_p2")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-background-cpu", type=float, default=20.0)
    parser.add_argument("--idle-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")
    protocol = PROJECT / "docs" / "sparse_runtime_protocol.md"
    if not protocol.exists():
        raise FileNotFoundError("frozen P2 protocol is missing")
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

    idle_samples = wait_for_idle(args.max_background_cpu, args.idle_timeout)
    print(json.dumps({"idle_cpu_samples": idle_samples}), flush=True)

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
            raise RuntimeError("sparse hierarchy violates the halving storage bound")
    del dense_reference, sparse_reference
    gc.collect()

    warmup = windows[:32]
    for representation, backend, workers in CONFIGURATIONS:
        candidate = _run(warmup, representation, backend, workers)
        if hierarchy_digest(candidate) != hierarchy_digest(
            decompose_sparse_batch(warmup, backend="serial", workers=1)
        ):
            raise RuntimeError(f"{representation}/{backend}/{workers} warm-up differs")

    rng = random.Random(20260825)
    rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        order = list(CONFIGURATIONS)
        rng.shuffle(order)
        for representation, backend, workers in order:
            gc.collect()
            pre_cpu = float(psutil.cpu_percent(interval=0.5))
            started = time.perf_counter_ns()
            result = _run(windows, representation, backend, workers)
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
                "pre_cpu_percent": pre_cpu,
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
        "protocol": "P2",
        "protocol_path": str(protocol),
        "exploratory_scaling_characterization": True,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "windows": len(windows),
        "samples_per_window": windows[0].size,
        "repetitions": args.repetitions,
        "initial_idle_cpu_samples": idle_samples,
        "reference_knot_digest": reference_digest,
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
