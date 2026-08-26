"""Run the frozen 80-record QTDB R2 confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "qtdb"
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.ecg import (  # noqa: E402
    DelineationResult,
    QRSBoundary,
    derivative_qrs_delineate,
    hcrd_qrs_delineate,
    parse_qrs_boundaries,
)
from hcrd.metrics import exact_sign_test, paired_bootstrap_ci  # noqa: E402

FAILURE_PENALTY_MS = 160.0
METHODS = (
    "hcrd_quadratic",
    "hcrd_gaussian",
    "hcrd_raw",
    "derivative_threshold",
    "official_pu0",
    "official_pu1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(records: list[str]) -> None:
    manifest = json.loads(
        (DATA / "download_manifest_confirmation_locked.json").read_text()
    )
    if {row["record"] for row in manifest} != set(records) or len(manifest) != 400:
        raise RuntimeError("locked download manifest does not match the frozen records")
    for row in manifest:
        path = DATA / f"{row['record']}.{row['extension']}"
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"QTDB integrity check failed for {path.name}")


def _read_boundaries(record: str, annotator: str) -> list[QRSBoundary]:
    annotation = wfdb.rdann(str(DATA / record), annotator)
    return parse_qrs_boundaries(annotation.sample, annotation.symbol, annotation.num)


def _official(
    target: QRSBoundary,
    candidates: list[QRSBoundary],
    tolerance_samples: int,
) -> DelineationResult:
    if not candidates:
        return DelineationResult(None, None, 0.0, 0)
    match = min(candidates, key=lambda item: abs(item.fiducial - target.fiducial))
    if abs(match.fiducial - target.fiducial) > tolerance_samples:
        return DelineationResult(None, None, 0.0, 0)
    return DelineationResult(match.onset, match.offset, 0.0, 1)


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _trial_row(
    record: str,
    beat_index: int,
    method: str,
    target: QRSBoundary,
    result: DelineationResult,
    sampling_frequency: float,
) -> dict[str, object]:
    success = result.succeeded
    scale = 1000.0 / sampling_frequency
    if success and result.onset is not None and result.offset is not None:
        onset_error = (result.onset - target.onset) * scale
        offset_error = (result.offset - target.offset) * scale
        duration_error = (
            (result.offset - result.onset) - (target.offset - target.onset)
        ) * scale
        joint = (abs(onset_error) + abs(offset_error)) / 2.0
    else:
        onset_error = np.nan
        offset_error = np.nan
        duration_error = np.nan
        joint = FAILURE_PENALTY_MS
    return {
        "record": record,
        "beat_index": beat_index,
        "method": method,
        "sampling_frequency": sampling_frequency,
        "target_onset": target.onset,
        "target_fiducial": target.fiducial,
        "target_offset": target.offset,
        "predicted_onset": result.onset,
        "predicted_offset": result.offset,
        "success": int(success),
        "onset_error_ms": onset_error,
        "offset_error_ms": offset_error,
        "duration_error_ms": duration_error,
        "joint_absolute_error_ms": joint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "qtdb_confirmation_r2",
    )
    args = parser.parse_args()
    output = args.output
    split = json.loads((DATA / "record_split.json").read_text())
    records = split["confirmation_locked"]
    if len(records) != 80 or set(records) & set(split["pilot"]):
        raise RuntimeError("frozen QTDB split is invalid")
    _verify_manifest(records)
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "frozen_protocol": str(PROJECT / "docs" / "qtdb_confirmation_protocol.md"),
        "records": records,
        "methods": METHODS,
        "failure_penalty_ms": FAILURE_PENALTY_MS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "wfdb": wfdb.__version__,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    trials: list[dict[str, object]] = []
    for record in records:
        waveform = wfdb.rdrecord(str(DATA / record))
        sampling_frequency = float(waveform.fs)
        if waveform.p_signal.shape[1] != 2:
            raise RuntimeError(f"expected two channels in {record}")
        manual = _read_boundaries(record, "q1c")
        automatic_zero = _read_boundaries(record, "pu0")
        automatic_one = _read_boundaries(record, "pu1")
        tolerance = int(round(0.1 * sampling_frequency))
        for beat_index, target in enumerate(manual):
            results = {
                "hcrd_quadratic": hcrd_qrs_delineate(
                    waveform.p_signal[:, 0],
                    target.fiducial,
                    sampling_frequency,
                    guide="quadratic",
                    regularization=10.0,
                    amplitude_ratio=0.2,
                ),
                "hcrd_gaussian": hcrd_qrs_delineate(
                    waveform.p_signal[:, 0],
                    target.fiducial,
                    sampling_frequency,
                    guide="gaussian",
                    gaussian_sigma=2.0,
                    amplitude_ratio=0.3,
                ),
                "hcrd_raw": hcrd_qrs_delineate(
                    waveform.p_signal[:, 1],
                    target.fiducial,
                    sampling_frequency,
                    guide="raw",
                    amplitude_ratio=0.1,
                ),
                "derivative_threshold": derivative_qrs_delineate(
                    waveform.p_signal[:, 0],
                    target.fiducial,
                    sampling_frequency,
                    gaussian_sigma=2.0,
                    threshold_ratio=0.05,
                ),
                "official_pu0": _official(target, automatic_zero, tolerance),
                "official_pu1": _official(target, automatic_one, tolerance),
            }
            for method, result in results.items():
                trials.append(
                    _trial_row(
                        record,
                        beat_index,
                        method,
                        target,
                        result,
                        sampling_frequency,
                    )
                )

    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)

    record_rows = []
    for record in records:
        for method in METHODS:
            group = [
                row for row in trials if row["record"] == record and row["method"] == method
            ]
            successful = [row for row in group if row["success"]]
            record_rows.append(
                {
                    "record": record,
                    "method": method,
                    "beats": len(group),
                    "failure_rate": 1.0 - len(successful) / len(group),
                    "mean_joint_absolute_error_ms": float(
                        np.mean([row["joint_absolute_error_ms"] for row in group])
                    ),
                    "mean_successful_onset_absolute_error_ms": float(
                        np.mean([abs(row["onset_error_ms"]) for row in successful])
                    )
                    if successful
                    else None,
                    "mean_successful_offset_absolute_error_ms": float(
                        np.mean([abs(row["offset_error_ms"]) for row in successful])
                    )
                    if successful
                    else None,
                    "mean_successful_duration_absolute_error_ms": float(
                        np.mean([abs(row["duration_error_ms"]) for row in successful])
                    )
                    if successful
                    else None,
                }
            )
    with (output / "record_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record_rows[0]))
        writer.writeheader()
        writer.writerows(record_rows)

    aggregate = []
    for method in METHODS:
        group = [row for row in record_rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                "records": len(group),
                "beats": int(sum(row["beats"] for row in group)),
                "mean_record_joint_absolute_error_ms": float(
                    np.mean([row["mean_joint_absolute_error_ms"] for row in group])
                ),
                "median_record_joint_absolute_error_ms": float(
                    np.median([row["mean_joint_absolute_error_ms"] for row in group])
                ),
                "mean_record_failure_rate": float(
                    np.mean([row["failure_rate"] for row in group])
                ),
                "mean_record_successful_onset_absolute_error_ms": float(
                    np.mean(
                        [
                            row["mean_successful_onset_absolute_error_ms"]
                            for row in group
                            if row["mean_successful_onset_absolute_error_ms"] is not None
                        ]
                    )
                ),
                "mean_record_successful_offset_absolute_error_ms": float(
                    np.mean(
                        [
                            row["mean_successful_offset_absolute_error_ms"]
                            for row in group
                            if row["mean_successful_offset_absolute_error_ms"] is not None
                        ]
                    )
                ),
                "mean_record_successful_duration_absolute_error_ms": float(
                    np.mean(
                        [
                            row["mean_successful_duration_absolute_error_ms"]
                            for row in group
                            if row["mean_successful_duration_absolute_error_ms"] is not None
                        ]
                    )
                ),
            }
        )
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    target = sorted(
        [row for row in record_rows if row["method"] == "hcrd_quadratic"],
        key=lambda row: row["record"],
    )
    comparisons = []
    for method in METHODS[1:]:
        comparator = sorted(
            [row for row in record_rows if row["method"] == method],
            key=lambda row: row["record"],
        )
        differences = np.array(
            [
                first["mean_joint_absolute_error_ms"]
                - second["mean_joint_absolute_error_ms"]
                for first, second in zip(target, comparator)
            ]
        )
        lower, upper = paired_bootstrap_ci(
            differences, samples=20_000, seed=20261024
        )
        comparisons.append(
            {
                "comparison": f"hcrd_quadratic - {method}",
                "mean_record_difference_ms": float(np.mean(differences)),
                "bootstrap_ci_lower_ms": lower,
                "bootstrap_ci_upper_ms": upper,
                "exact_sign_p": exact_sign_test(differences),
                "win_rate": float(np.mean(differences < 0)),
                "ties": int(np.sum(differences == 0)),
            }
        )
    adjusted = _holm([row["exact_sign_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_sign_p"] = value
        row["superiority_supported"] = bool(
            row["bootstrap_ci_upper_ms"] < 0 and value < 0.05
        )
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
