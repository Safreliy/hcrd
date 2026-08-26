"""Post-test D3: augment full HCRD pulse features with local wrist motion."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import analyze_ppg_dalia_results as analysis
import run_ppg_dalia_benchmark as benchmark
from hcrd.ppg import suppress_events


PROJECT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT / "data" / "raw" / "ppg_dalia"
RESULT_ROOT = PROJECT / "results" / "ppg_dalia" / "d3_local_motion"
MODEL_ROOT = PROJECT / "results" / "ppg_dalia" / "models"
PROTOCOL = PROJECT / "docs" / "ppg_dalia_local_motion_development.md"
PROTOCOL_SHA256 = "38fd5b9053fe8024595c48156e26a14d16faed7d36019798beef490bc840a838"
WINDOW_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _clean_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if np.all(finite):
        return values
    if not np.any(finite):
        return np.zeros_like(values)
    indices = np.arange(values.size)
    return np.interp(indices, indices[finite], values[finite])


def _load_acceleration() -> dict[str, tuple[np.ndarray, float]]:
    output: dict[str, tuple[np.ndarray, float]] = {}
    paths = sorted(RAW_ROOT.glob("ppg_dalia_*_data.mat"))
    if not paths:
        raise FileNotFoundError(
            f"no PPG-DaLiA activity files under {RAW_ROOT}; run download_ppg_dalia.py"
        )
    for path in paths:
        activity = path.stem.removeprefix("ppg_dalia_").removesuffix("_data")
        records = np.atleast_1d(
            loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
        )
        for item in records:
            subject = str(item.fix.subj_name)
            key = f"{subject}_{activity}"
            output[key] = (
                _clean_signal(np.asarray(item.acc_ppg_site.v)),
                float(item.acc_ppg_site.fs),
            )
    return output


def local_motion_features(
    acceleration: np.ndarray,
    acceleration_fs: float,
    candidate_positions: np.ndarray,
    ppg_fs: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the fixed D3 local-motion coordinates at PPG candidate times."""

    acceleration = _clean_signal(acceleration)
    jerk = np.abs(np.diff(acceleration, prepend=acceleration[0]))
    columns = []
    names = []
    mean_8s = None
    for seconds in WINDOW_SECONDS:
        width = max(1, int(round(seconds * acceleration_fs)))
        mean = uniform_filter1d(acceleration, size=width, mode="nearest")
        mean_square = uniform_filter1d(
            acceleration * acceleration, size=width, mode="nearest"
        )
        standard_deviation = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
        mean_absolute_difference = uniform_filter1d(
            jerk, size=width, mode="nearest"
        )
        columns.extend(
            [np.log1p(standard_deviation), np.log1p(mean_absolute_difference)]
        )
        names.extend(
            [
                f"log1p_acc_std_{seconds:g}s",
                f"log1p_acc_mean_abs_diff_{seconds:g}s",
            ]
        )
        if seconds == 8.0:
            mean_8s = mean
    if mean_8s is None:  # pragma: no cover - fixed window tuple contains 8 s
        raise RuntimeError("8-second window missing")
    columns.append(np.log1p(np.abs(acceleration - mean_8s)))
    names.append("log1p_acc_abs_deviation_8s")
    sample_features = np.column_stack(columns)
    indices = np.rint(
        np.asarray(candidate_positions, dtype=np.float64)
        * acceleration_fs
        / float(ppg_fs)
    ).astype(np.int64)
    indices = np.clip(indices, 0, acceleration.size - 1)
    result = np.asarray(sample_features[indices], dtype=np.float32)
    if result.shape[1] != 11 or not np.all(np.isfinite(result)):
        raise RuntimeError("invalid D3 local-motion feature matrix")
    return result, tuple(names)


def _motion_matrices(records: list[benchmark.Record]):
    raw = _load_acceleration()
    matrices = {}
    expected_names = None
    for record in records:
        if record.key not in raw:
            raise KeyError(f"missing acceleration for {record.key}")
        acceleration, sampling_frequency = raw[record.key]
        matrix, names = local_motion_features(
            acceleration,
            sampling_frequency,
            record.positions,
            record.sampling_frequency,
        )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("motion feature names changed between records")
        matrices[record.key] = matrix
    return matrices, expected_names


def _features(record, motion, feature_set):
    if feature_set == "geometry_motion":
        return np.column_stack([record.geometry, motion[record.key]])
    if feature_set == "hybrid_motion":
        return np.column_stack(
            [record.geometry, record.morphology, motion[record.key]]
        )
    raise ValueError(feature_set)


def _make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_iter=200,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=1729,
    )


def _fit(records, motion, feature_set, fold):
    x = np.vstack([_features(record, motion, feature_set) for record in records])
    y = np.concatenate([record.labels for record in records])
    model = _make_model()
    model.fit(x, y)
    average_precision = float(
        average_precision_score(y, model.predict_proba(x)[:, 1])
    )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    path = MODEL_ROOT / f"d3_fold_{fold}_{feature_set}.joblib"
    joblib.dump(model, path)
    metadata = {
        "feature_set": feature_set,
        "input_features": int(x.shape[1]),
        "training_candidates": int(x.shape[0]),
        "training_positive_candidates": int(np.count_nonzero(y)),
        "descriptive_training_average_precision": average_precision,
        "model_file": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "model_sha256": benchmark.sha256(path),
    }
    return model, metadata


def _probabilities(records, model, motion, feature_set):
    return {
        record.key: model.predict_proba(
            _features(record, motion, feature_set)
        )[:, 1]
        for record in records
    }


def _predictions(records, probabilities, threshold):
    return {
        record.key: suppress_events(
            record.positions,
            probabilities[record.key],
            int(round(0.2 * record.sampling_frequency)),
            threshold=float(threshold),
        )
        for record in records
    }


def _select_threshold(records, probabilities):
    grid = []
    for threshold in benchmark.THRESHOLDS:
        metrics = benchmark.score_predictions(
            records,
            _predictions(records, probabilities, threshold),
            include_activities=False,
        )
        grid.append(
            {"threshold": threshold, "metrics": benchmark._compact(metrics)}
        )
    best = max(
        grid,
        key=lambda item: benchmark._rank(
            dict(item["metrics"]), float(item["threshold"])
        ),
    )
    return best, grid


def _pooled(per_record):
    report = benchmark._summarize(per_record)
    report["by_activity"] = {
        activity: benchmark._summarize(
            [item for item in per_record if item["activity"] == activity]
        )
        for activity in benchmark.ACTIVITIES
        if any(item["activity"] == activity for item in per_record)
    }
    report["motion_intensive_macro_median_f1"] = float(
        np.mean(
            [
                report["by_activity"][activity]["median_record_f1"]
                for activity in benchmark.HARD_ACTIVITIES
            ]
        )
    )
    return report


def _run_fold(records, folds, motion, fold):
    test_subjects = set(folds[fold])
    validation_subjects = set(folds[(fold + 1) % len(folds)])
    development = [
        record
        for record in records
        if record.subject not in test_subjects | validation_subjects
    ]
    validation = [record for record in records if record.subject in validation_subjects]
    test = [record for record in records if record.subject in test_subjects]
    original_lock = json.loads(
        (
            PROJECT
            / "results"
            / "ppg_dalia"
            / "folds"
            / f"fold_{fold}"
            / "outer_test_lock.json"
        ).read_text(encoding="utf-8")
    )
    fitted = {}
    validation_rows = []
    for feature_set in ("geometry_motion", "hybrid_motion"):
        model, metadata = _fit(development, motion, feature_set, fold)
        probabilities = _probabilities(validation, model, motion, feature_set)
        selected, grid = _select_threshold(validation, probabilities)
        selected_metrics = benchmark.score_predictions(
            validation,
            _predictions(validation, probabilities, float(selected["threshold"])),
        )
        row = {
            "metadata": metadata,
            "threshold_grid": grid,
            "selected": {
                "threshold": float(selected["threshold"]),
                "metrics": selected_metrics,
            },
        }
        validation_rows.append(row)
        fitted[feature_set] = model
        print(
            f"fold {fold} validation {feature_set}: "
            f"threshold={selected['threshold']:.2f}, "
            f"median={selected_metrics['median_record_f1']:.6f}, "
            f"micro={selected_metrics['micro_f1']:.6f}",
            flush=True,
        )
    primary = max(
        validation_rows,
        key=lambda row: (
            *benchmark._rank(dict(dict(row["selected"])["metrics"])),
            int(dict(row["metadata"])["feature_set"] == "geometry_motion"),
        ),
    )
    lock = {
        "status": "post-outer-test exploratory development",
        "fold": fold,
        "protocol_sha256": PROTOCOL_SHA256,
        "development_subjects": sorted({record.subject for record in development}),
        "validation_subjects": sorted(validation_subjects),
        "test_subjects": sorted(test_subjects),
        "p0": original_lock["p0"],
        "models": [
            {
                **dict(row["metadata"]),
                "threshold": float(dict(row["selected"])["threshold"]),
            }
            for row in validation_rows
        ],
        "primary_model": str(dict(primary["metadata"])["feature_set"]),
    }
    fold_root = RESULT_ROOT / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    lock_path = fold_root / "outer_test_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    test_signals = benchmark._conditioned(test)
    p0 = benchmark.score_predictions(
        test,
        benchmark._p0_predictions(
            test,
            test_signals,
            int(lock["p0"]["maximum_bpm"]),
            float(lock["p0"]["prominence"]),
        ),
    )
    test_models = {}
    for item in lock["models"]:
        feature_set = str(item["feature_set"])
        probabilities = _probabilities(
            test, fitted[feature_set], motion, feature_set
        )
        test_models[feature_set] = benchmark.score_predictions(
            test,
            _predictions(test, probabilities, float(item["threshold"])),
        )
        print(
            f"fold {fold} TEST {feature_set}: "
            f"median={test_models[feature_set]['median_record_f1']:.6f}, "
            f"micro={test_models[feature_set]['micro_f1']:.6f}",
            flush=True,
        )
    summary = {
        "fold": fold,
        "lock_sha256": benchmark.sha256(lock_path),
        "validation": validation_rows,
        "primary_model": lock["primary_model"],
        "test": {"p0": p0, "models": test_models},
    }
    summary_path = fold_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    if benchmark.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("D3 protocol changed after its development lock")
    if benchmark.sha256(benchmark.RECORD_MANIFEST) != benchmark.MANIFEST_SHA256:
        raise RuntimeError("PPG-DaLiA record manifest changed")
    records, excluded, folds = benchmark.load_records()
    if excluded:
        raise RuntimeError(f"unexpected excluded records: {excluded}")
    motion, motion_names = _motion_matrices(records)
    summaries = [_run_fold(records, folds, motion, fold) for fold in range(5)]

    p0_records = []
    model_records = {"geometry_motion": [], "hybrid_motion": []}
    primary_records = []
    selections = []
    for summary in summaries:
        p0_records.extend(summary["test"]["p0"]["per_record"])
        for feature_set in model_records:
            model_records[feature_set].extend(
                summary["test"]["models"][feature_set]["per_record"]
            )
        selected = str(summary["primary_model"])
        selections.append(selected)
        primary_records.extend(
            summary["test"]["models"][selected]["per_record"]
        )
    pooled_p0 = _pooled(p0_records)
    pooled_models = {
        name: _pooled(items) for name, items in model_records.items()
    }
    pooled_primary = _pooled(primary_records)
    pooled_primary["fold_model_selections"] = selections
    frozen = json.loads(
        (
            PROJECT
            / "results"
            / "ppg_dalia"
            / "nested_subjectwise_summary.json"
        ).read_text(encoding="utf-8")
    )
    comparisons = {
        "primary_vs_p0": analysis._bootstrap(
            pooled_primary, pooled_p0, seed=1740
        ),
        "primary_vs_frozen_geometry": analysis._bootstrap(
            pooled_primary, frozen["models"]["hgb_geometry"], seed=1741
        ),
        "primary_vs_frozen_hybrid": analysis._bootstrap(
            pooled_primary, frozen["models"]["hgb_hybrid"], seed=1742
        ),
        "primary_vs_mass_only": analysis._bootstrap(
            pooled_primary, frozen["models"]["hgb_mass_control"], seed=1743
        ),
    }
    result = {
        "status": "post-outer-test exploratory development; requires untouched-cohort confirmation",
        "protocol_sha256": PROTOCOL_SHA256,
        "motion_feature_names": list(motion_names or ()),
        "records": len(records),
        "folds": [
            {
                "fold": summary["fold"],
                "primary_model": summary["primary_model"],
                "lock_sha256": summary["lock_sha256"],
            }
            for summary in summaries
        ],
        "p0": pooled_p0,
        "models": pooled_models,
        "cross_fitted_primary": pooled_primary,
        "comparisons": comparisons,
    }
    destination = RESULT_ROOT / "local_motion_development.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_models": selections,
                "primary_median_f1": pooled_primary["median_record_f1"],
                "primary_micro_f1": pooled_primary["micro_f1"],
                "primary_motion_macro_f1": pooled_primary[
                    "motion_intensive_macro_median_f1"
                ],
                "p0_median_f1": pooled_p0["median_record_f1"],
                "p0_micro_f1": pooled_p0["micro_f1"],
                "result": str(destination.relative_to(PROJECT)).replace(
                    "\\", "/"
                ),
                "sha256": benchmark.sha256(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
