"""Exploratory accelerometer gate between P0 and geometry-only HCRD."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat

import run_ppg_dalia_benchmark as benchmark


PROJECT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT / "data" / "raw" / "ppg_dalia"
RESULT_ROOT = PROJECT / "results" / "ppg_dalia"
ACTIVITIES = benchmark.ACTIVITIES
HARD = benchmark.HARD_ACTIVITIES


def _motion_features() -> dict[str, dict[str, float]]:
    output = {}
    for path in sorted(RAW_ROOT.glob("ppg_dalia_*_data.mat")):
        activity = path.stem.removeprefix("ppg_dalia_").removesuffix("_data")
        records = np.atleast_1d(
            loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
        )
        for record in records:
            subject = str(record.fix.subj_name)
            acceleration = np.asarray(record.acc_ppg_site.v, dtype=float).reshape(-1)
            differences = np.abs(np.diff(acceleration))
            output[f"{subject}_{activity}"] = {
                "acceleration_std": float(np.std(acceleration)),
                "median_absolute_difference": float(np.median(differences)),
                "q95_absolute_difference": float(np.quantile(differences, 0.95)),
            }
    return output


def _combine_predictions(records, p0, geometry, motion, feature, threshold):
    return {
        record.key: (
            geometry[record.key]
            if motion[record.key][feature] >= threshold
            else p0[record.key]
        )
        for record in records
    }


def _pooled(folds):
    per_record = []
    for fold in folds:
        per_record.extend(fold["test_metrics"]["per_record"])
    report = benchmark._summarize(per_record)
    report["by_activity"] = {
        activity: benchmark._summarize(
            [item for item in per_record if item["activity"] == activity]
        )
        for activity in ACTIVITIES
        if any(item["activity"] == activity for item in per_record)
    }
    report["motion_intensive_macro_median_f1"] = float(
        np.mean(
            [
                report["by_activity"][activity]["median_record_f1"]
                for activity in HARD
            ]
        )
    )
    return report


def main() -> None:
    records, excluded, folds = benchmark.load_records()
    if excluded:
        raise RuntimeError("unexpected excluded PPG-DaLiA records")
    motion = _motion_features()
    features = (
        "acceleration_std",
        "median_absolute_difference",
        "q95_absolute_difference",
    )
    fold_results = []
    for fold in range(5):
        fold_root = RESULT_ROOT / "folds" / f"fold_{fold}"
        lock = json.loads((fold_root / "outer_test_lock.json").read_text(encoding="utf-8"))
        validation_subjects = set(lock["validation_subjects"])
        test_subjects = set(lock["test_subjects"])
        validation = [
            record for record in records if record.subject in validation_subjects
        ]
        test = [record for record in records if record.subject in test_subjects]
        geometry_spec = next(
            item for item in lock["models"] if item["name"] == "hgb_geometry"
        )
        model_path = PROJECT / str(geometry_spec["model_file"])
        if benchmark.sha256(model_path) != geometry_spec["model_sha256"]:
            raise RuntimeError(f"model hash mismatch: {model_path}")
        model = joblib.load(model_path)
        validation_signals = benchmark._conditioned(validation)
        validation_p0 = benchmark._p0_predictions(
            validation,
            validation_signals,
            int(lock["p0"]["maximum_bpm"]),
            float(lock["p0"]["prominence"]),
        )
        validation_geometry = benchmark._model_predictions(
            validation,
            model,
            "geometry",
            float(geometry_spec["threshold"]),
        )
        grid = []
        for feature in features:
            values = np.asarray([motion[record.key][feature] for record in validation])
            thresholds = np.unique(
                np.r_[
                    np.min(values) - max(1.0, 0.01 * np.ptp(values)),
                    np.quantile(values, np.linspace(0.05, 0.95, 19)),
                    np.max(values) + max(1.0, 0.01 * np.ptp(values)),
                ]
            )
            for threshold in thresholds:
                metrics = benchmark.score_predictions(
                    validation,
                    _combine_predictions(
                        validation,
                        validation_p0,
                        validation_geometry,
                        motion,
                        feature,
                        float(threshold),
                    ),
                    include_activities=False,
                )
                grid.append(
                    {
                        "feature": feature,
                        "threshold": float(threshold),
                        "geometry_records": int(
                            np.count_nonzero(values >= threshold)
                        ),
                        "metrics": benchmark._compact(metrics),
                    }
                )
        selected = max(
            grid,
            key=lambda item: benchmark._rank(
                dict(item["metrics"]), float(item["threshold"])
            ),
        )
        test_signals = benchmark._conditioned(test)
        test_p0 = benchmark._p0_predictions(
            test,
            test_signals,
            int(lock["p0"]["maximum_bpm"]),
            float(lock["p0"]["prominence"]),
        )
        test_geometry = benchmark._model_predictions(
            test,
            model,
            "geometry",
            float(geometry_spec["threshold"]),
        )
        test_predictions = _combine_predictions(
            test,
            test_p0,
            test_geometry,
            motion,
            str(selected["feature"]),
            float(selected["threshold"]),
        )
        test_metrics = benchmark.score_predictions(test, test_predictions)
        geometry_test_records = [
            record.key
            for record in test
            if motion[record.key][str(selected["feature"])]
            >= float(selected["threshold"])
        ]
        fold_result = {
            "fold": fold,
            "status": "exploratory_post_outer_test",
            "validation_grid": grid,
            "selected": selected,
            "test_geometry_records": geometry_test_records,
            "test_metrics": test_metrics,
        }
        fold_results.append(fold_result)
        print(
            f"fold {fold}: {selected['feature']} >= {selected['threshold']:.6g}; "
            f"geometry {len(geometry_test_records)}/{len(test)} test records; "
            f"median={test_metrics['median_record_f1']:.6f}, "
            f"micro={test_metrics['micro_f1']:.6f}",
            flush=True,
        )
    pooled = _pooled(fold_results)
    result = {
        "status": "exploratory_post_outer_test; requires independent confirmation",
        "rule": "Use geometry-only HCRD for records above a validation-selected wrist-acceleration threshold, otherwise P0.",
        "folds": fold_results,
        "pooled": pooled,
        "motion_features": motion,
    }
    destination = RESULT_ROOT / "motion_gate_exploratory.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "median_record_f1": pooled["median_record_f1"],
                "micro_f1": pooled["micro_f1"],
                "motion_macro_f1": pooled["motion_intensive_macro_median_f1"],
                "summary": str(destination.relative_to(PROJECT)).replace("\\", "/"),
                "sha256": benchmark.sha256(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
