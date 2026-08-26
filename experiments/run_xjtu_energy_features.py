"""Extract multiscale HCRD energy features from XJTU-SY vibration files.

Protocol: docs/xjtu_sy_rul_protocol.md (X1). Raw third-party data are never
written to the release; only named features, metadata, and hashes are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.energy import (  # noqa: E402
    multiscale_energy_feature_names,
    multiscale_energy_features,
)

CONDITIONS = {
    "35Hz12kN": [f"Bearing1_{index}" for index in range(1, 6)],
    "37.5Hz11kN": [f"Bearing2_{index}" for index in range(1, 6)],
    "40Hz10kN": [f"Bearing3_{index}" for index in range(1, 6)],
}
SAMPLING_RATE = 25_600.0
PROFILE_BINS = 256
MAX_LEVELS = 6


def _numeric_csvs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.csv"), key=lambda path: int(path.stem))


def _block_rms_profile(values: np.ndarray, bins: int = PROFILE_BINS) -> np.ndarray:
    if values.ndim != 1 or values.size % bins != 0:
        raise ValueError("signal length must be divisible by profile bin count")
    blocks = values.reshape(bins, values.size // bins)
    return np.sqrt(np.mean(blocks**2, axis=1))


def _log_power_profile(values: np.ndarray, bins: int = PROFILE_BINS) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
    spectrum = spectrum[1:]
    usable = (spectrum.size // bins) * bins
    if usable < bins:
        raise ValueError("spectrum is too short for requested bin count")
    binned = spectrum[:usable].reshape(bins, usable // bins).mean(axis=1)
    return np.log1p(binned)


def _profile_features(profile: np.ndarray, prefix: str) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, profile.size)
    values = multiscale_energy_features(
        profile, grid, max_levels=MAX_LEVELS
    )
    names = multiscale_energy_feature_names(MAX_LEVELS)
    return {f"{prefix}{name}": float(value) for name, value in zip(names, values)}


def extract_file_features(path: Path) -> dict[str, float]:
    signal = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if signal.shape != (32_768, 2):
        raise ValueError(f"unexpected XJTU-SY shape {signal.shape} in {path}")
    output: dict[str, float] = {}
    for channel, channel_name in enumerate(("h", "v")):
        values = signal[:, channel]
        output.update(
            _profile_features(_block_rms_profile(values), f"{channel_name}_env_")
        )
        output.update(
            _profile_features(_log_power_profile(values), f"{channel_name}_spec_")
        )
    return output


def _extract_bearing(task: tuple[str, str, str]) -> list[dict[str, object]]:
    data_root_text, condition, bearing_id = task
    folder = Path(data_root_text) / condition / bearing_id
    files = _numeric_csvs(folder)
    if not files:
        raise FileNotFoundError(f"no CSV files under {folder}")
    records: list[dict[str, object]] = []
    denominator = max(1, len(files) - 1)
    for file_index, path in enumerate(files):
        records.append(
            {
                "condition": condition,
                "bearing_id": bearing_id,
                "filename": path.name,
                "file_idx": file_index,
                "total_files": len(files),
                "rul_normalized": 1.0 - file_index / denominator,
                **extract_file_features(path),
            }
        )
    return records


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
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "xjtu_x1" / "hcrd_energy_features.csv.gz",
    )
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    tasks = [
        (str(args.data), condition, bearing)
        for condition, bearings in CONDITIONS.items()
        for bearing in bearings
    ]
    started = time.perf_counter()
    if args.workers == 1:
        bearing_records = [_extract_bearing(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            bearing_records = list(executor.map(_extract_bearing, tasks, chunksize=1))
    records = [record for group in bearing_records for record in group]
    frame = pd.DataFrame.from_records(records).sort_values(
        ["condition", "bearing_id", "file_idx"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, compression="gzip")
    elapsed = time.perf_counter() - started
    metadata = {
        "protocol": "X1",
        "exploratory_stage": "feature extraction before model outcome",
        "rows": len(frame),
        "columns": len(frame.columns),
        "profile_bins": PROFILE_BINS,
        "max_levels": MAX_LEVELS,
        "profiles": ["block_rms", "log_power"],
        "channels": ["horizontal", "vertical"],
        "workers": args.workers,
        "seconds": elapsed,
        "milliseconds_per_file": 1000.0 * elapsed / max(1, len(frame)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "output_sha256": _sha256(args.output),
    }
    metadata_path = args.output.with_name("hcrd_energy_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
