"""Post-lock R3: off-the-shelf NeuroKit2 DWT QRS delineation on QTDB."""

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
import json
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import neurokit2
import numpy as np
import scipy
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "qtdb"
OUTPUT = PROJECT / "results" / "qtdb_modern_r3"
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.ecg import QRSBoundary, parse_qrs_boundaries  # noqa: E402
from hcrd.metrics import exact_sign_test, paired_bootstrap_ci  # noqa: E402

FAILURE_PENALTY_MS = 160.0
WORKERS = 8
METHODS = ("neurokit_dwt_ch0", "neurokit_dwt_ch1")


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
        raise RuntimeError("locked download manifest does not match R3 records")
    for row in manifest:
        path = DATA / f"{row['record']}.{row['extension']}"
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"QTDB integrity check failed for {path.name}")


def _failure_row(
    record: str,
    beat_index: int,
    method: str,
    target: QRSBoundary,
    sampling_frequency: float,
    reason: str,
) -> dict[str, object]:
    return {
        "record": record,
        "beat_index": beat_index,
        "method": method,
        "sampling_frequency": sampling_frequency,
        "target_onset": target.onset,
        "target_fiducial": target.fiducial,
        "target_offset": target.offset,
        "detected_r_peak": None,
        "predicted_onset": None,
        "predicted_offset": None,
        "success": 0,
        "failure_reason": reason,
        "onset_error_ms": None,
        "offset_error_ms": None,
        "duration_error_ms": None,
        "joint_absolute_error_ms": FAILURE_PENALTY_MS,
    }


def _success_row(
    record: str,
    beat_index: int,
    method: str,
    target: QRSBoundary,
    sampling_frequency: float,
    detected_r_peak: int,
    onset: int,
    offset: int,
) -> dict[str, object]:
    scale = 1000.0 / sampling_frequency
    onset_error = (onset - target.onset) * scale
    offset_error = (offset - target.offset) * scale
    duration_error = ((offset - onset) - (target.offset - target.onset)) * scale
    return {
        "record": record,
        "beat_index": beat_index,
        "method": method,
        "sampling_frequency": sampling_frequency,
        "target_onset": target.onset,
        "target_fiducial": target.fiducial,
        "target_offset": target.offset,
        "detected_r_peak": detected_r_peak,
        "predicted_onset": onset,
        "predicted_offset": offset,
        "success": 1,
        "failure_reason": "",
        "onset_error_ms": onset_error,
        "offset_error_ms": offset_error,
        "duration_error_ms": duration_error,
        "joint_absolute_error_ms": (abs(onset_error) + abs(offset_error)) / 2.0,
    }


def _process_channel(
    record: str,
    signal: np.ndarray,
    targets: list[QRSBoundary],
    sampling_frequency: float,
    channel: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import neurokit2 as nk

    method = f"neurokit_dwt_ch{channel}"
    try:
        cleaned = nk.ecg_clean(
            signal, sampling_rate=sampling_frequency, method="neurokit"
        )
        _, peak_info = nk.ecg_peaks(
            cleaned,
            sampling_rate=sampling_frequency,
            method="neurokit",
            correct_artifacts=False,
        )
        peaks = np.asarray(peak_info["ECG_R_Peaks"], dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(
                cleaned,
                peaks,
                sampling_rate=sampling_frequency,
                method="dwt",
                check=False,
            )
        onsets = np.asarray(waves["ECG_R_Onsets"], dtype=float)
        offsets = np.asarray(waves["ECG_R_Offsets"], dtype=float)
        if onsets.size != peaks.size or offsets.size != peaks.size:
            raise RuntimeError("DWT boundary arrays do not align with R peaks")
    except Exception as error:
        rows = [
            _failure_row(
                record,
                beat_index,
                method,
                target,
                sampling_frequency,
                f"pipeline:{type(error).__name__}",
            )
            for beat_index, target in enumerate(targets)
        ]
        return rows, {
            "record": record,
            "channel": channel,
            "detected_r_peaks": 0,
            "pipeline_error": f"{type(error).__name__}: {error}",
        }

    tolerance = max(1, int(round(0.045 * sampling_frequency)))
    rows: list[dict[str, object]] = []
    for beat_index, target in enumerate(targets):
        if peaks.size == 0:
            rows.append(
                _failure_row(
                    record,
                    beat_index,
                    method,
                    target,
                    sampling_frequency,
                    "no_r_peaks",
                )
            )
            continue
        match_index = int(np.argmin(np.abs(peaks - target.fiducial)))
        detected = int(peaks[match_index])
        if abs(detected - target.fiducial) > tolerance:
            rows.append(
                _failure_row(
                    record,
                    beat_index,
                    method,
                    target,
                    sampling_frequency,
                    "unmatched_r_peak",
                )
            )
            continue
        onset_value = onsets[match_index]
        offset_value = offsets[match_index]
        if not np.isfinite(onset_value) or not np.isfinite(offset_value):
            rows.append(
                _failure_row(
                    record,
                    beat_index,
                    method,
                    target,
                    sampling_frequency,
                    "missing_dwt_boundary",
                )
            )
            continue
        onset = int(round(float(onset_value)))
        offset = int(round(float(offset_value)))
        if not onset <= detected <= offset:
            rows.append(
                _failure_row(
                    record,
                    beat_index,
                    method,
                    target,
                    sampling_frequency,
                    "invalid_boundary_order",
                )
            )
            continue
        rows.append(
            _success_row(
                record,
                beat_index,
                method,
                target,
                sampling_frequency,
                detected,
                onset,
                offset,
            )
        )
    return rows, {
        "record": record,
        "channel": channel,
        "detected_r_peaks": int(peaks.size),
        "pipeline_error": "",
    }


def _process_record(record: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    annotation = wfdb.rdann(str(DATA / record), "q1c")
    targets = parse_qrs_boundaries(
        annotation.sample, annotation.symbol, annotation.num
    )
    waveform = wfdb.rdrecord(str(DATA / record))
    if waveform.p_signal is None or waveform.p_signal.shape[1] != 2:
        raise RuntimeError(f"{record} does not contain two physical channels")
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for channel in (0, 1):
        channel_rows, channel_diagnostics = _process_channel(
            record,
            np.asarray(waveform.p_signal[:, channel], dtype=float),
            targets,
            float(waveform.fs),
            channel,
        )
        rows.extend(channel_rows)
        diagnostics.append(channel_diagnostics)
    return rows, diagnostics


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    split = json.loads((DATA / "record_split.json").read_text(encoding="utf-8"))
    records = list(split["confirmation_locked"])
    if len(records) != 80:
        raise RuntimeError("R3 requires the exact 80 locked R2 records")
    _verify_manifest(records)

    started = time.perf_counter()
    trials: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for record, (record_rows, record_diagnostics) in zip(
            records, executor.map(_process_record, records, chunksize=1), strict=True
        ):
            trials.extend(record_rows)
            diagnostics.extend(record_diagnostics)
            print(json.dumps({"record": record, "rows": len(record_rows)}), flush=True)
    elapsed = time.perf_counter() - started
    trials.sort(key=lambda row: (str(row["record"]), str(row["method"]), int(row["beat_index"])))
    if len(trials) != 2 * 2785:
        raise RuntimeError(f"expected 5570 trial rows, found {len(trials)}")

    record_rows: list[dict[str, object]] = []
    for record in records:
        for method in METHODS:
            group = [
                row for row in trials if row["record"] == record and row["method"] == method
            ]
            successful = [row for row in group if row["success"] == 1]
            record_rows.append(
                {
                    "record": record,
                    "method": method,
                    "beats": len(group),
                    "failure_rate": 1.0 - len(successful) / len(group),
                    "mean_joint_absolute_error_ms": float(
                        np.mean([float(row["joint_absolute_error_ms"]) for row in group])
                    ),
                    "mean_successful_onset_absolute_error_ms": float(
                        np.mean([abs(float(row["onset_error_ms"])) for row in successful])
                    ) if successful else None,
                    "mean_successful_offset_absolute_error_ms": float(
                        np.mean([abs(float(row["offset_error_ms"])) for row in successful])
                    ) if successful else None,
                    "mean_successful_duration_absolute_error_ms": float(
                        np.mean([abs(float(row["duration_error_ms"])) for row in successful])
                    ) if successful else None,
                }
            )

    aggregate: list[dict[str, object]] = []
    for method in METHODS:
        group = [row for row in record_rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                "records": len(group),
                "beats": int(sum(int(row["beats"]) for row in group)),
                "mean_record_joint_absolute_error_ms": float(
                    np.mean([float(row["mean_joint_absolute_error_ms"]) for row in group])
                ),
                "median_record_joint_absolute_error_ms": float(
                    np.median([float(row["mean_joint_absolute_error_ms"]) for row in group])
                ),
                "mean_record_failure_rate": float(
                    np.mean([float(row["failure_rate"]) for row in group])
                ),
            }
        )

    with (
        PROJECT / "results" / "qtdb_confirmation_r2" / "record_summary.csv"
    ).open(encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    hcrd = {
        str(row["record"]): float(row["mean_joint_absolute_error_ms"])
        for row in original_rows
        if row["method"] == "hcrd_quadratic"
    }
    comparisons: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        comparator = {
            str(row["record"]): float(row["mean_joint_absolute_error_ms"])
            for row in record_rows
            if row["method"] == method
        }
        if set(comparator) != set(hcrd):
            raise RuntimeError("R3 and R2 record sets differ")
        differences = np.asarray(
            [hcrd[record] - comparator[record] for record in sorted(hcrd)]
        )
        lower, upper = paired_bootstrap_ci(
            differences, samples=20_000, seed=20261103 + method_index
        )
        comparisons.append(
            {
                "comparison": f"hcrd_quadratic - {method}",
                "post_lock_exploratory": True,
                "mean_record_difference_ms": float(np.mean(differences)),
                "bootstrap_ci_lower_ms": lower,
                "bootstrap_ci_upper_ms": upper,
                "exact_sign_p": exact_sign_test(differences),
                "hcrd_win_rate": float(np.mean(differences < 0)),
                "ties": int(np.sum(differences == 0)),
            }
        )
    adjusted = _holm([float(row["exact_sign_p"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_sign_p"] = value
        row["confirmatory_superiority_claim_permitted"] = False

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("trials.csv", trials),
        ("record_summary.csv", record_rows),
        ("diagnostics.csv", diagnostics),
    ):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUTPUT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (OUTPUT / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    metadata = {
        "protocol": str(PROJECT / "docs" / "qtdb_modern_baseline_protocol.md"),
        "post_lock_exploratory": True,
        "workers": WORKERS,
        "elapsed_seconds": elapsed,
        "records": records,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "wfdb": wfdb.__version__,
        "neurokit2": neurokit2.__version__,
    }
    (OUTPUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "comparisons": comparisons, "elapsed": elapsed}, indent=2))


if __name__ == "__main__":
    main()
