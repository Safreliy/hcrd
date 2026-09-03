"""Benchmark the matrix-free SCI contrast family on uniform designs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from shapecontrast import build_shape_contrast_family  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - optional measurement only
    psutil = None


def _rss_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.Process().memory_info().rss)


def benchmark(sample_sizes: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in sample_sizes:
        gc.collect()
        before = _rss_bytes()
        x = np.arange(1, n + 1, dtype=float) / (n + 1)
        y = x - (x - 0.5) ** 3

        started = perf_counter()
        family = build_shape_contrast_family(
            x, separation_multipliers=(1, 2, 4)
        )
        built = perf_counter()
        estimates = family.means(y)
        evaluated = perf_counter()
        after = _rss_bytes()

        dense_bytes = family.contrast_count * n * np.dtype(float).itemsize
        rows.append(
            {
                "n": n,
                "contrast_count": family.contrast_count,
                "build_seconds": built - started,
                "evaluate_seconds": evaluated - built,
                "stored_mebibytes": family.stored_bytes / 2**20,
                "estimate_vector_mebibytes": estimates.nbytes / 2**20,
                "dense_operator_gibibytes": dense_bytes / 2**30,
                "rss_increase_mebibytes": (
                    None if before is None or after is None else (after - before) / 2**20
                ),
                "finite": bool(np.all(np.isfinite(estimates))),
            }
        )
        del estimates, family, y, x
    return rows


def _write(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "experiment": "matrix-free SCI scaling",
        "separation_multipliers": [1, 2, 4],
        "rows": rows,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "rss_measurement": "psutil" if psutil is not None else "unavailable",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Matrix-free SCI scaling",
        "",
        "The benchmark uses the full `(1, 2, 4)` separation family. Times are",
        "hardware-specific. Stored array sizes and dense-matrix counterfactuals",
        "follow directly from the constructed family.",
        "",
        "| n | contrasts | build (s) | evaluate (s) | stored (MiB) | dense matrix (GiB) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {n} | {contrast_count} | {build_seconds:.4f} | "
            "{evaluate_seconds:.4f} | {stored_mebibytes:.2f} | "
            "{dense_operator_gibibytes:.2f} |".format(**row)
        )
    report.extend(
        [
            "",
            "The compact family stores scale metadata and start indices. On a",
            "uniform design, contrast evaluation uses prefix sums. No",
            "contrast-by-observation matrix is created.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[1_000, 10_000, 100_000, 1_000_000],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/sci/matrix_free_scaling",
    )
    args = parser.parse_args()
    if any(value < 3 for value in args.sample_sizes):
        raise ValueError("every sample size must be at least three")
    rows = benchmark(args.sample_sizes)
    _write(rows, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
