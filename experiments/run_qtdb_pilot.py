"""Exploratory QTDB pilot; never use its rows as confirmatory evidence."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "qtdb"
OUTPUT = PROJECT / "results" / "qtdb_pilot"
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.ecg import (  # noqa: E402
    DelineationResult,
    QRSBoundary,
    derivative_qrs_delineate,
    hcrd_qrs_delineate,
    parse_qrs_boundaries,
)

FAILURE_PENALTY_MS = 160.0


def _read_boundaries(record: str, annotator: str) -> list[QRSBoundary]:
    annotation = wfdb.rdann(str(DATA / record), annotator)
    return parse_qrs_boundaries(annotation.sample, annotation.symbol, annotation.num)


def _match_automatic(
    target: QRSBoundary,
    automatic: list[QRSBoundary],
    tolerance_samples: int,
) -> DelineationResult:
    if not automatic:
        return DelineationResult(None, None, 0.0, 0)
    candidate = min(automatic, key=lambda item: abs(item.fiducial - target.fiducial))
    if abs(candidate.fiducial - target.fiducial) > tolerance_samples:
        return DelineationResult(None, None, 0.0, 0)
    return DelineationResult(candidate.onset, candidate.offset, 0.0, 1)


def _row(
    record: str,
    beat_index: int,
    sampling_frequency: float,
    target: QRSBoundary,
    result: DelineationResult,
    **metadata: object,
) -> dict[str, object]:
    success = result.succeeded
    scale = 1000.0 / sampling_frequency
    onset_error = (
        (result.onset - target.onset) * scale if success and result.onset is not None else np.nan
    )
    offset_error = (
        (result.offset - target.offset) * scale
        if success and result.offset is not None
        else np.nan
    )
    duration_error = (
        ((result.offset - result.onset) - (target.offset - target.onset)) * scale
        if success and result.onset is not None and result.offset is not None
        else np.nan
    )
    return {
        "record": record,
        "beat_index": beat_index,
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
        "joint_absolute_error_ms": (
            (abs(onset_error) + abs(offset_error)) / 2.0
            if success
            else FAILURE_PENALTY_MS
        ),
        **metadata,
    }


def main() -> None:
    split = json.loads((DATA / "record_split.json").read_text())
    records = split["pilot"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "exploratory pilot only",
        "source_plan": str(PROJECT / "docs" / "qt_database_plan.md"),
        "records": records,
        "failure_penalty_ms": FAILURE_PENALTY_MS,
        "hcrd_amplitude_ratios": [0.1, 0.2, 0.3],
        "gaussian_sigmas": [1.0, 2.0, 4.0],
        "quadratic_regularizations": [1.0, 10.0, 100.0, 1000.0],
        "derivative_threshold_ratios": [0.05, 0.1, 0.2, 0.3],
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for record in records:
        waveform = wfdb.rdrecord(str(DATA / record))
        sampling_frequency = float(waveform.fs)
        manual = _read_boundaries(record, "q1c")
        automatic = {
            0: _read_boundaries(record, "pu0"),
            1: _read_boundaries(record, "pu1"),
        }
        for beat_index, target in enumerate(manual):
            tolerance = int(round(0.1 * sampling_frequency))
            for channel in (0, 1):
                signal = waveform.p_signal[:, channel]
                official = _match_automatic(target, automatic[channel], tolerance)
                rows.append(
                    _row(
                        record,
                        beat_index,
                        sampling_frequency,
                        target,
                        official,
                        family="official_ecgpuwave",
                        method=f"official_pu{channel}",
                        channel=channel,
                        guide_parameter=0.0,
                        selection_ratio=0.0,
                    )
                )
                for amplitude_ratio in (0.1, 0.2, 0.3):
                    raw = hcrd_qrs_delineate(
                        signal,
                        target.fiducial,
                        sampling_frequency,
                        guide="raw",
                        amplitude_ratio=amplitude_ratio,
                    )
                    rows.append(
                        _row(
                            record,
                            beat_index,
                            sampling_frequency,
                            target,
                            raw,
                            family="hcrd",
                            method="hcrd_raw",
                            channel=channel,
                            guide_parameter=0.0,
                            selection_ratio=amplitude_ratio,
                        )
                    )
                    for gaussian_sigma in (1.0, 2.0, 4.0):
                        result = hcrd_qrs_delineate(
                            signal,
                            target.fiducial,
                            sampling_frequency,
                            guide="gaussian",
                            gaussian_sigma=gaussian_sigma,
                            amplitude_ratio=amplitude_ratio,
                        )
                        rows.append(
                            _row(
                                record,
                                beat_index,
                                sampling_frequency,
                                target,
                                result,
                                family="hcrd",
                                method="hcrd_gaussian",
                                channel=channel,
                                guide_parameter=gaussian_sigma,
                                selection_ratio=amplitude_ratio,
                            )
                        )
                    for regularization in (1.0, 10.0, 100.0, 1000.0):
                        result = hcrd_qrs_delineate(
                            signal,
                            target.fiducial,
                            sampling_frequency,
                            guide="quadratic",
                            regularization=regularization,
                            amplitude_ratio=amplitude_ratio,
                        )
                        rows.append(
                            _row(
                                record,
                                beat_index,
                                sampling_frequency,
                                target,
                                result,
                                family="hcrd",
                                method="hcrd_quadratic",
                                channel=channel,
                                guide_parameter=regularization,
                                selection_ratio=amplitude_ratio,
                            )
                        )
                for gaussian_sigma in (1.0, 2.0, 4.0):
                    for threshold_ratio in (0.05, 0.1, 0.2, 0.3):
                        result = derivative_qrs_delineate(
                            signal,
                            target.fiducial,
                            sampling_frequency,
                            gaussian_sigma=gaussian_sigma,
                            threshold_ratio=threshold_ratio,
                        )
                        rows.append(
                            _row(
                                record,
                                beat_index,
                                sampling_frequency,
                                target,
                                result,
                                family="derivative",
                                method="derivative_threshold",
                                channel=channel,
                                guide_parameter=gaussian_sigma,
                                selection_ratio=threshold_ratio,
                            )
                        )

    fieldnames = list(rows[0])
    with (OUTPUT / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    keys = ("family", "method", "channel", "guide_parameter", "selection_ratio")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for key, group in grouped.items():
        record_means = []
        for record in records:
            values = [
                float(row["joint_absolute_error_ms"])
                for row in group
                if row["record"] == record
            ]
            if values:
                record_means.append(float(np.mean(values)))
        summaries.append(
            {
                **dict(zip(keys, key)),
                "records": len(record_means),
                "beats": len(group),
                "mean_record_joint_absolute_error_ms": float(np.mean(record_means)),
                "median_record_joint_absolute_error_ms": float(np.median(record_means)),
                "failure_rate": float(1.0 - np.mean([row["success"] for row in group])),
            }
        )
    summaries.sort(key=lambda row: row["mean_record_joint_absolute_error_ms"])
    (OUTPUT / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    selected = {}
    for family in ("hcrd", "derivative"):
        family_rows = [row for row in summaries if row["family"] == family]
        best_score = family_rows[0]["mean_record_joint_absolute_error_ms"]
        tied = [
            row
            for row in family_rows
            if row["mean_record_joint_absolute_error_ms"] <= best_score + 0.25
        ]
        tied.sort(
            key=lambda row: (
                row["guide_parameter"],
                -row["selection_ratio"],
                row["channel"],
                row["method"],
            )
        )
        selected[family] = tied[0]
    (OUTPUT / "selected.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
