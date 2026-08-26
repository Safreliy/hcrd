"""Run the fixed post-M1 ecgpuwave + multilevel HCRD hybrid experiment M2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "qtdb"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

from hcrd.ecg import DelineationResult, QRSBoundary  # noqa: E402
from hcrd.metrics import exact_sign_test, paired_bootstrap_ci  # noqa: E402
from run_qtdb_multilevel_fusion import (  # noqa: E402
    _feature_columns,
    _model,
    _official,
    _read_boundaries,
    _to_result,
    _trial_row,
)


def _add_ecgpuwave_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns = {
        f"pu{channel}_{suffix}": []
        for channel in (0, 1)
        for suffix in ("onset_ms", "offset_ms", "width_ms", "success")
    }
    by_record = {
        (record, channel): _read_boundaries(record, f"pu{channel}")
        for record in frame["record"].unique()
        for channel in (0, 1)
    }
    for row in frame.itertuples(index=False):
        fs = float(row.sampling_frequency)
        target = QRSBoundary(int(row.target_onset), int(row.fiducial), int(row.target_offset))
        tolerance = int(round(0.1 * fs))
        scale = 1000.0 / fs
        for channel in (0, 1):
            match = _official(target, by_record[(row.record, channel)], tolerance)
            if match.succeeded and match.onset is not None and match.offset is not None:
                onset_ms = (match.onset - target.fiducial) * scale
                offset_ms = (match.offset - target.fiducial) * scale
                values = (onset_ms, offset_ms, offset_ms - onset_ms, 1.0)
            else:
                values = (np.nan, np.nan, np.nan, 0.0)
            for suffix, value in zip(
                ("onset_ms", "offset_ms", "width_ms", "success"), values
            ):
                columns[f"pu{channel}_{suffix}"].append(value)
    for name, values in columns.items():
        result[name] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT / "results" / "qtdb_multilevel_m1" / "features.csv.gz",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "qtdb_hybrid_m2"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features = _add_ecgpuwave_features(pd.read_csv(args.features))
    train = features[features["partition"] == "pilot"].copy()
    test = features[features["partition"] == "evaluation"].copy()
    pu_columns = [column for column in features.columns if column.startswith("pu")]
    hcrd_columns = _feature_columns(features, multilevel=True)

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, columns in (
        ("learned_ecgpuwave", pu_columns),
        ("learned_ecgpuwave_hcrd", pu_columns + hcrd_columns),
    ):
        onset_model = _model().fit(train[columns], train["target_onset_ms"])
        offset_model = _model().fit(train[columns], train["target_offset_ms"])
        predictions[method] = (
            onset_model.predict(test[columns]),
            offset_model.predict(test[columns]),
        )

    trials: list[dict[str, object]] = []
    for position, row in enumerate(test.itertuples(index=False)):
        target = QRSBoundary(int(row.target_onset), int(row.fiducial), int(row.target_offset))
        fs = float(row.sampling_frequency)
        for method, (onsets, offsets) in predictions.items():
            estimate = _to_result(onsets[position], offsets[position], target.fiducial, fs)
            trials.append(_trial_row(row.record, row.beat_index, method, target, estimate, fs))
        for channel in (0, 1):
            onset = getattr(row, f"pu{channel}_onset_ms")
            offset = getattr(row, f"pu{channel}_offset_ms")
            if np.isfinite(onset) and np.isfinite(offset):
                estimate = DelineationResult(
                    target.fiducial + int(round(onset * fs / 1000.0)),
                    target.fiducial + int(round(offset * fs / 1000.0)),
                    0.0,
                    1,
                )
            else:
                estimate = DelineationResult(None, None, 0.0, 0)
            trials.append(
                _trial_row(
                    row.record,
                    row.beat_index,
                    f"official_pu{channel}",
                    target,
                    estimate,
                    fs,
                )
            )
    trial_frame = pd.DataFrame(trials)
    trial_frame.to_csv(args.output / "trials.csv", index=False)
    record_summary = (
        trial_frame.groupby(["record", "method"], as_index=False)
        .agg(
            beats=("joint_absolute_error_ms", "size"),
            failure_rate=("success", lambda values: 1.0 - float(np.mean(values))),
            mean_joint_absolute_error_ms=("joint_absolute_error_ms", "mean"),
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
    for primary_name, comparator_name in (
        ("learned_ecgpuwave_hcrd", "learned_ecgpuwave"),
        ("learned_ecgpuwave_hcrd", "official_pu0"),
        ("learned_ecgpuwave_hcrd", "official_pu1"),
        ("learned_ecgpuwave", "official_pu0"),
        ("learned_ecgpuwave", "official_pu1"),
    ):
        primary = record_summary[
            record_summary["method"] == primary_name
        ].sort_values("record")
        comparator = record_summary[
            record_summary["method"] == comparator_name
        ].sort_values("record")
        difference = (
            primary["mean_joint_absolute_error_ms"].to_numpy()
            - comparator["mean_joint_absolute_error_ms"].to_numpy()
        )
        comparisons.append(
            {
                "comparison": f"{primary_name} - {comparator_name}",
                "mean_difference_ms": float(np.mean(difference)),
                "bootstrap_95_ci_ms": list(
                    paired_bootstrap_ci(difference, samples=100_000)
                ),
                "records_improved": int(np.sum(difference < 0)),
                "records_total": int(len(difference)),
                "exact_sign_test_p_two_sided": exact_sign_test(difference),
            }
        )
    summary = {
        "analysis_status": "post-M1 development; protocol fixed before M2 run",
        "protocol": "docs/qtdb_hybrid_fusion_protocol.md",
        "pilot_records": int(train["record"].nunique()),
        "evaluation_records": int(test["record"].nunique()),
        "ecgpuwave_feature_count": len(pu_columns),
        "hcrd_feature_count": len(hcrd_columns),
        "aggregate": aggregate.to_dict(orient="records"),
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
