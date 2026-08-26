"""Run the frozen independent PRONOSTIA health-indicator confirmation H1."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd import decompose_sparse, level_energies  # noqa: E402
from hcrd.metrics import exact_sign_test, paired_bootstrap_ci  # noqa: E402


PROFILE_BINS = 256
MAX_LEVELS = 6
SAMPLING_RATE = 25_600.0
ARCHIVE_SHA256 = "e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f"


def _file_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot parse acquisition number from {path.name}")
    return int(match.group(1))


def _extract_file(path: Path, delimiter: str) -> dict[str, float]:
    horizontal = np.loadtxt(path, delimiter=delimiter, usecols=(4,), dtype=np.float64)
    if horizontal.shape != (2560,) or not np.all(np.isfinite(horizontal)):
        raise ValueError(f"unexpected PRONOSTIA signal in {path}")
    centered = horizontal - float(np.mean(horizontal))
    variance = float(np.mean(centered**2))
    fourth = float(np.mean(centered**4))
    kurtosis = fourth / variance**2 if variance > 0.0 else 0.0
    spectrum = np.fft.rfft(centered)
    frequency = np.fft.rfftfreq(horizontal.size, d=1.0 / SAMPLING_RATE)
    band_power = float(np.sum(np.abs(spectrum[frequency <= 1000.0]) ** 2))

    blocks = horizontal.reshape(PROFILE_BINS, horizontal.size // PROFILE_BINS)
    profile = np.sqrt(np.mean(blocks**2, axis=1))
    hierarchy = decompose_sparse(
        profile,
        np.linspace(0.0, 1.0, PROFILE_BINS),
        atol=1e-12,
        rtol=64 * np.finfo(float).eps,
        max_levels=MAX_LEVELS,
    )
    energies = level_energies(hierarchy)
    level_three_mass = energies[2].polygon_area if len(energies) >= 3 else 0.0
    return {
        "hcrd_level3_log1p_polygon_mass": float(np.log1p(level_three_mass)),
        "horizontal_rms": float(np.sqrt(np.mean(horizontal**2))),
        "horizontal_variance": variance,
        "horizontal_kurtosis": kurtosis,
        "horizontal_0_1khz_band_power": float(np.log1p(band_power)),
    }


def _extract_bearing(task: tuple[str, str]) -> list[dict[str, object]]:
    partition, folder_text = task
    folder = Path(folder_text)
    files = sorted(folder.glob("acc_*.csv"), key=_file_number)
    if len(files) < 20:
        raise ValueError(f"too few acquisitions in {folder}")
    with files[0].open(encoding="utf-8") as handle:
        first_line = handle.readline()
    delimiter = ";" if ";" in first_line else ","
    denominator = len(files) - 1
    rows: list[dict[str, object]] = []
    for index, path in enumerate(files):
        rows.append(
            {
                "partition": partition,
                "bearing_id": folder.name,
                "filename": path.name,
                "file_idx": index,
                "total_files": len(files),
                "life_progress": index / denominator,
                **_extract_file(path, delimiter),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bearing_scores(features: pd.DataFrame) -> pd.DataFrame:
    indicator_columns = [
        "hcrd_level3_log1p_polygon_mass",
        "horizontal_rms",
        "horizontal_variance",
        "horizontal_kurtosis",
        "horizontal_0_1khz_band_power",
    ]
    rows: list[dict[str, object]] = []
    for (partition, bearing), group in features.groupby(["partition", "bearing_id"]):
        progress = group["life_progress"].to_numpy(dtype=float)
        for indicator in indicator_columns:
            values = group[indicator].to_numpy(dtype=float)
            rows.append(
                {
                    "partition": partition,
                    "bearing_id": bearing,
                    "indicator": indicator,
                    "spearman_trendability": float(spearmanr(progress, values).statistic),
                    "pearson_trendability": float(pearsonr(progress, values).statistic),
                    "acquisitions": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT / "data" / "raw" / "pronostia" / "FEMTO_Bearing.zip",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=PROJECT
        / "data"
        / "raw"
        / "pronostia"
        / "dataset"
        / "training"
        / "Learning_set",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=PROJECT
        / "data"
        / "raw"
        / "pronostia"
        / "dataset"
        / "validation"
        / "Full_Test_Set",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "pronostia_h1"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    archive_hash = _sha256(args.archive)
    if archive_hash.lower() != ARCHIVE_SHA256:
        raise RuntimeError(f"archive hash mismatch: {archive_hash}")

    tasks = [
        *(('training', str(path)) for path in sorted(args.training.glob("Bearing*"))),
        *(('full_test', str(path)) for path in sorted(args.validation.glob("Bearing*"))),
    ]
    if len(tasks) != 17:
        raise RuntimeError(f"expected 17 complete trajectories, found {len(tasks)}")
    started = time.perf_counter()
    if args.workers == 1:
        groups = [_extract_bearing(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            groups = list(executor.map(_extract_bearing, tasks, chunksize=1))
    features = pd.DataFrame([row for group in groups for row in group]).sort_values(
        ["partition", "bearing_id", "file_idx"]
    )
    scores = _bearing_scores(features)
    args.output.mkdir(parents=True, exist_ok=True)
    features_path = args.output / "features.csv.gz"
    features.to_csv(features_path, index=False, compression="gzip")
    scores.to_csv(args.output / "bearing_scores.csv", index=False)

    wide = scores.pivot(
        index="bearing_id", columns="indicator", values="spearman_trendability"
    )
    hcrd = wide["hcrd_level3_log1p_polygon_mass"].to_numpy(dtype=float)
    rms = wide["horizontal_rms"].to_numpy(dtype=float)
    difference = np.abs(hcrd) - np.abs(rms)
    interval = paired_bootstrap_ci(difference, samples=200_000)
    aggregates = []
    for indicator, group in scores.groupby("indicator"):
        values = group["spearman_trendability"].to_numpy(dtype=float)
        aggregates.append(
            {
                "indicator": indicator,
                "mean_spearman": float(np.mean(values)),
                "median_spearman": float(np.median(values)),
                "median_absolute_spearman": float(np.median(np.abs(values))),
                "positive_bearings": int(np.sum(values > 0.0)),
                "bearings": int(len(values)),
            }
        )
    success = bool(np.all(hcrd > 0.0) and interval[0] > 0.0)
    summary = {
        "protocol": "H1 independent confirmation",
        "protocol_file": "docs/pronostia_health_indicator_protocol.md",
        "success": success,
        "primary": {
            "contrast": "abs(Spearman HCRD level-3 polygon mass) - abs(Spearman horizontal RMS)",
            "mean_difference": float(np.mean(difference)),
            "bearing_bootstrap_95_ci": list(interval),
            "bearings_improved": int(np.sum(difference > 0.0)),
            "bearings_total": int(len(difference)),
            "exact_sign_test_p_two_sided": exact_sign_test(-difference),
            "hcrd_positive_on_all_bearings": bool(np.all(hcrd > 0.0)),
        },
        "aggregates": sorted(aggregates, key=lambda row: row["indicator"]),
        "metadata": {
            "complete_bearings": len(tasks),
            "acquisitions": int(len(features)),
            "archive_sha256": archive_hash,
            "features_sha256": _sha256(features_path),
            "workers": args.workers,
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
