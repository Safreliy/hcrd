"""Parallel wrapper for the pinned 65-feature XJTU-SY reference extractor.

The upstream implementation is MIT licensed and remains a separate checkout:
https://github.com/thfmn/xjtu-sy-bearing (commit pinned below). This wrapper
records provenance and does not vendor or silently modify the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/thfmn/xjtu-sy-bearing.git"
UPSTREAM_COMMIT = "7d7231c582961741bde629da6731e6c169d88785"
CONDITIONS = {
    "35Hz12kN": [f"Bearing1_{index}" for index in range(1, 6)],
    "37.5Hz11kN": [f"Bearing2_{index}" for index in range(1, 6)],
    "40Hz10kN": [f"Bearing3_{index}" for index in range(1, 6)],
}


def _numeric_csvs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.csv"), key=lambda path: int(path.stem))


def _extract_bearing(task: tuple[str, str, str, str]) -> list[dict[str, object]]:
    data_root_text, baseline_repo_text, condition, bearing_id = task
    baseline_repo = Path(baseline_repo_text)
    sys.path.insert(0, str(baseline_repo))
    from src.features.fusion import FeatureExtractor

    extractor = FeatureExtractor(
        mode="all", condition=condition, sampling_rate=25_600.0
    )
    files = _numeric_csvs(Path(data_root_text) / condition / bearing_id)
    if not files:
        raise FileNotFoundError(f"no CSV files for {bearing_id}")
    output: list[dict[str, object]] = []
    for file_index, path in enumerate(files):
        signal = pd.read_csv(path).to_numpy(dtype=np.float32)
        features = extractor.extract(signal)
        output.append(
            {
                "condition": condition,
                "bearing_id": bearing_id,
                "filename": path.name,
                "file_idx": file_index,
                "total_files": len(files),
                "rul": 125.0 * (1.0 - file_index / max(1, len(files) - 1)),
                **{
                    name: float(value)
                    for name, value in zip(
                        extractor.feature_names, features, strict=True
                    )
                },
            }
        )
    return output


def _git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT / "data" / "raw" / "xjtu_sy" / "XJTU-SY_Bearing_Datasets",
    )
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "xjtu_x1" / "standard_features.csv.gz",
    )
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    commit = _git_commit(args.baseline_repo)
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"reference extractor must be at {UPSTREAM_COMMIT}; found {commit}"
        )
    tasks = [
        (str(args.data), str(args.baseline_repo), condition, bearing)
        for condition, bearings in CONDITIONS.items()
        for bearing in bearings
    ]
    started = time.perf_counter()
    if args.workers == 1:
        groups = [_extract_bearing(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            groups = list(executor.map(_extract_bearing, tasks, chunksize=1))
    records = [record for group in groups for record in group]
    frame = pd.DataFrame.from_records(records).sort_values(
        ["condition", "bearing_id", "file_idx"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, compression="gzip")
    elapsed = time.perf_counter() - started
    metadata = {
        "protocol": "X1",
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": commit,
        "upstream_license": "MIT",
        "rows": len(frame),
        "feature_count": len(frame.columns) - 6,
        "workers": args.workers,
        "seconds": elapsed,
        "milliseconds_per_file": 1000.0 * elapsed / max(1, len(frame)),
        "python": platform.python_version(),
        "output_sha256": _sha256(args.output),
    }
    args.output.with_name("standard_features_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
