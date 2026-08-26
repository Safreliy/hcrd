"""Run the fixed post-R2 QTDB multilevel HCRD fusion experiment M1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import scipy
import wfdb
from lightgbm import LGBMRegressor

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "qtdb"
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.ecg import (  # noqa: E402
    DelineationResult,
    QRSBoundary,
    hcrd_qrs_multilevel_candidates,
    parse_qrs_boundaries,
)
from hcrd.metrics import exact_sign_test, paired_bootstrap_ci  # noqa: E402


RATIOS = (0.1, 0.2, 0.3)
MAX_LEVELS = 4
FAILURE_PENALTY_MS = 160.0
METHODS = (
    "pilot_constant",
    "learned_level1",
    "learned_multilevel",
    "official_pu0",
    "official_pu1",
)


def _read_boundaries(record: str, annotator: str) -> list[QRSBoundary]:
    annotation = wfdb.rdann(str(DATA / record), annotator)
    return parse_qrs_boundaries(annotation.sample, annotation.symbol, annotation.num)


def _official(
    target: QRSBoundary, candidates: list[QRSBoundary], tolerance_samples: int
) -> DelineationResult:
    if not candidates:
        return DelineationResult(None, None, 0.0, 0)
    match = min(candidates, key=lambda item: abs(item.fiducial - target.fiducial))
    if abs(match.fiducial - target.fiducial) > tolerance_samples:
        return DelineationResult(None, None, 0.0, 0)
    return DelineationResult(match.onset, match.offset, 0.0, 1)


def _candidate_columns(channel: int, level: int, ratio: float) -> list[str]:
    prefix = f"c{channel}_l{level}_r{round(100 * ratio):02d}"
    return [
        f"{prefix}_onset_ms",
        f"{prefix}_offset_ms",
        f"{prefix}_width_ms",
        f"{prefix}_success",
        f"{prefix}_log1p_normalized_anchor",
        f"{prefix}_log1p_structure_count",
    ]


def _extract_record(record: str, partition: str) -> list[dict[str, object]]:
    waveform = wfdb.rdrecord(str(DATA / record))
    fs = float(waveform.fs)
    manual = _read_boundaries(record, "q1c")
    rows: list[dict[str, object]] = []
    for beat_index, target in enumerate(manual):
        scale = 1000.0 / fs
        row: dict[str, object] = {
            "partition": partition,
            "record": record,
            "beat_index": beat_index,
            "sampling_frequency": fs,
            "fiducial": target.fiducial,
            "target_onset": target.onset,
            "target_offset": target.offset,
            "target_onset_ms": (target.onset - target.fiducial) * scale,
            "target_offset_ms": (target.offset - target.fiducial) * scale,
        }
        for channel in (0, 1):
            candidates = hcrd_qrs_multilevel_candidates(
                waveform.p_signal[:, channel],
                target.fiducial,
                fs,
                guide="quadratic",
                regularization=10.0,
                amplitude_ratios=RATIOS,
                max_levels=MAX_LEVELS,
            )
            lookup = {
                (candidate.level, candidate.amplitude_ratio): candidate
                for candidate in candidates
            }
            for level in range(1, MAX_LEVELS + 1):
                for ratio in RATIOS:
                    columns = _candidate_columns(channel, level, ratio)
                    candidate = lookup.get((level, ratio))
                    if candidate is None or not candidate.succeeded:
                        values = [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan]
                    else:
                        assert candidate.onset is not None and candidate.offset is not None
                        onset_ms = (candidate.onset - target.fiducial) * scale
                        offset_ms = (candidate.offset - target.fiducial) * scale
                        values = [
                            onset_ms,
                            offset_ms,
                            offset_ms - onset_ms,
                            1.0,
                            np.log1p(candidate.normalized_anchor_amplitude),
                            np.log1p(candidate.structure_count),
                        ]
                    row.update(dict(zip(columns, values)))
        rows.append(row)
    return rows


def _model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=30,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=20260824,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
    )


def _feature_columns(frame: pd.DataFrame, *, multilevel: bool) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("c")]
    if multilevel:
        return columns
    return [column for column in columns if "_l1_" in column]


def _to_result(
    onset_ms: float, offset_ms: float, fiducial: int, fs: float
) -> DelineationResult:
    onset_ms = float(np.clip(onset_ms, -140.0, -4.0))
    offset_ms = float(np.clip(offset_ms, 4.0, 180.0))
    onset = fiducial + int(round(onset_ms * fs / 1000.0))
    offset = fiducial + int(round(offset_ms * fs / 1000.0))
    return DelineationResult(onset, offset, 0.0, 1)


def _trial_row(
    record: str,
    beat_index: int,
    method: str,
    target: QRSBoundary,
    result: DelineationResult,
    fs: float,
) -> dict[str, object]:
    success = result.succeeded
    scale = 1000.0 / fs
    if success and result.onset is not None and result.offset is not None:
        onset_error = (result.onset - target.onset) * scale
        offset_error = (result.offset - target.offset) * scale
        joint = (abs(onset_error) + abs(offset_error)) / 2.0
    else:
        onset_error = np.nan
        offset_error = np.nan
        joint = FAILURE_PENALTY_MS
    return {
        "record": record,
        "beat_index": beat_index,
        "method": method,
        "success": int(success),
        "onset_error_ms": onset_error,
        "offset_error_ms": offset_error,
        "joint_absolute_error_ms": joint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "qtdb_multilevel_m1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "record_split.json").read_text(encoding="utf-8"))
    train_records = split["pilot"]
    test_records = split["confirmation_locked"]
    if set(train_records) & set(test_records):
        raise RuntimeError("pilot and evaluation records overlap")

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for partition, records in (("pilot", train_records), ("evaluation", test_records)):
        for index, record in enumerate(records, start=1):
            rows.extend(_extract_record(record, partition))
            print(json.dumps({"partition": partition, "record": record, "done": index, "total": len(records)}), flush=True)
    features = pd.DataFrame(rows)
    features.to_csv(args.output / "features.csv.gz", index=False, compression="gzip")
    train = features[features["partition"] == "pilot"].copy()
    test = features[features["partition"] == "evaluation"].copy()

    prediction_columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prediction_columns["pilot_constant"] = (
        np.full(len(test), train["target_onset_ms"].median()),
        np.full(len(test), train["target_offset_ms"].median()),
    )
    for method, multilevel in (("learned_level1", False), ("learned_multilevel", True)):
        columns = _feature_columns(features, multilevel=multilevel)
        onset_model = _model().fit(train[columns], train["target_onset_ms"])
        offset_model = _model().fit(train[columns], train["target_offset_ms"])
        prediction_columns[method] = (
            onset_model.predict(test[columns]),
            offset_model.predict(test[columns]),
        )

    test_lookup = {
        (str(row.record), int(row.beat_index)): row
        for row in test.itertuples(index=False)
    }
    official = {
        (record, channel): _read_boundaries(record, f"pu{channel}")
        for record in test_records
        for channel in (0, 1)
    }
    trial_rows: list[dict[str, object]] = []
    for position, row in enumerate(test.itertuples(index=False)):
        target = QRSBoundary(int(row.target_onset), int(row.fiducial), int(row.target_offset))
        fs = float(row.sampling_frequency)
        for method, (onsets, offsets) in prediction_columns.items():
            result = _to_result(onsets[position], offsets[position], target.fiducial, fs)
            trial_rows.append(_trial_row(row.record, row.beat_index, method, target, result, fs))
        tolerance = int(round(0.1 * fs))
        for channel in (0, 1):
            result = _official(target, official[(row.record, channel)], tolerance)
            trial_rows.append(
                _trial_row(
                    row.record,
                    row.beat_index,
                    f"official_pu{channel}",
                    target,
                    result,
                    fs,
                )
            )
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(args.output / "trials.csv", index=False)

    record_summary = (
        trials.groupby(["record", "method"], as_index=False)
        .agg(
            beats=("joint_absolute_error_ms", "size"),
            failure_rate=("success", lambda values: 1.0 - float(np.mean(values))),
            mean_joint_absolute_error_ms=("joint_absolute_error_ms", "mean"),
            mean_onset_absolute_error_ms=("onset_error_ms", lambda values: float(np.nanmean(np.abs(values)))),
            mean_offset_absolute_error_ms=("offset_error_ms", lambda values: float(np.nanmean(np.abs(values)))),
        )
    )
    record_summary.to_csv(args.output / "record_summary.csv", index=False)
    aggregate = (
        record_summary.groupby("method", as_index=False)
        .agg(
            records=("record", "size"),
            macro_joint_absolute_error_ms=("mean_joint_absolute_error_ms", "mean"),
            median_record_joint_absolute_error_ms=("mean_joint_absolute_error_ms", "median"),
            macro_failure_rate=("failure_rate", "mean"),
        )
        .sort_values("macro_joint_absolute_error_ms")
    )

    comparisons = []
    primary = record_summary[record_summary["method"] == "learned_multilevel"].sort_values("record")
    for comparator_name in ("learned_level1", "pilot_constant", "official_pu0", "official_pu1"):
        comparator = record_summary[record_summary["method"] == comparator_name].sort_values("record")
        if primary["record"].tolist() != comparator["record"].tolist():
            raise RuntimeError("record alignment failure")
        differences = primary["mean_joint_absolute_error_ms"].to_numpy() - comparator["mean_joint_absolute_error_ms"].to_numpy()
        comparisons.append(
            {
                "comparison": f"learned_multilevel - {comparator_name}",
                "mean_difference_ms": float(np.mean(differences)),
                "bootstrap_95_ci_ms": list(paired_bootstrap_ci(differences, samples=100_000)),
                "records_improved": int(np.sum(differences < 0)),
                "records_total": int(len(differences)),
                "exact_sign_test_p_two_sided": exact_sign_test(differences),
            }
        )

    metadata = {
        "analysis_status": "post-R2 development; protocol fixed before M1 run",
        "protocol": "docs/qtdb_multilevel_fusion_protocol.md",
        "pilot_records": len(train_records),
        "evaluation_records": len(test_records),
        "pilot_beats": int(len(train)),
        "evaluation_beats": int(len(test)),
        "feature_count_level1": len(_feature_columns(features, multilevel=False)),
        "feature_count_multilevel": len(_feature_columns(features, multilevel=True)),
        "seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wfdb": wfdb.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "aggregate": aggregate.to_dict(orient="records"),
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
