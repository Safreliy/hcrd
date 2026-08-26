"""Nested subject-wise PPG-DaLiA benchmark for multilevel HCRD event features."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hcrd.ppg import match_events, robust_bandpass, suppress_events
from hcrd.ppg_benchmark import align_ecg_to_ppg


PROJECT = Path(__file__).resolve().parents[1]
RECORD_MANIFEST = PROJECT / "data" / "manifests" / "ppg_dalia_records.json"
FEATURE_ROOT = PROJECT / "results" / "ppg_dalia" / "features"
RESULT_ROOT = PROJECT / "results" / "ppg_dalia"
MODEL_ROOT = RESULT_ROOT / "models"
PROTOCOL = PROJECT / "docs" / "ppg_dalia_hcrd_protocol.md"
PROTOCOL_SHA256 = "0d75839c5ea850795a29f4b0250c7297336c156b3cce3204586b208c847ed000"
MANIFEST_SHA256 = "95d18294f434159f3208c4c080a11993ada5d4745dcd40735067221958bc05c0"
MINIMUM_REFERENCE_BEATS = 3
TOLERANCE_SECONDS = 0.15
SCORING_WORKERS = max(1, min(8, os.cpu_count() or 1))
THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))
ACTIVITIES = (
    "car_driving",
    "cycling",
    "lunch_break",
    "sitting",
    "stair_climbing",
    "table_soccer",
    "walking",
    "working",
)
HARD_ACTIVITIES = ("stair_climbing", "table_soccer", "walking")


@dataclass
class Record:
    key: str
    subject: str
    activity: str
    outer_fold: int
    raw_file: Path
    sampling_frequency: float
    sample_count: int
    valid_sample_count: int
    positions: np.ndarray
    geometry: np.ndarray
    morphology: np.ndarray
    geometry_names: tuple[str, ...]
    labels: np.ndarray
    reference: np.ndarray
    provisional_reference_count: int
    invalid: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records() -> tuple[list[Record], list[dict[str, object]], list[list[str]]]:
    manifest = json.loads(RECORD_MANIFEST.read_text(encoding="utf-8"))
    records = []
    excluded = []
    for row in manifest["records"]:
        cache_path = FEATURE_ROOT / f"{row['key']}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing {cache_path}; run prepare_ppg_dalia_features.py"
            )
        with np.load(cache_path) as cached:
            provisional_count = int(cached["aligned_reference_p0"].size)
            if provisional_count < MINIMUM_REFERENCE_BEATS:
                excluded.append(
                    {
                        "key": row["key"],
                        "subject": row["subject"],
                        "activity": row["activity"],
                        "reference_beats": provisional_count,
                    }
                )
                continue
            invalid = np.asarray(cached["ppg_invalid_mask"], dtype=bool)
            records.append(
                Record(
                    key=str(row["key"]),
                    subject=str(row["subject"]),
                    activity=str(row["activity"]),
                    outer_fold=int(row["outer_fold"]),
                    raw_file=PROJECT / str(row["file"]),
                    sampling_frequency=float(cached["sampling_frequency"]),
                    sample_count=int(cached["sample_count"]),
                    valid_sample_count=int(np.count_nonzero(~invalid)),
                    positions=np.asarray(cached["positions"], dtype=np.int64),
                    geometry=np.asarray(cached["geometry"], dtype=np.float32),
                    morphology=np.asarray(cached["morphology"], dtype=np.float32),
                    geometry_names=tuple(cached["geometry_names"].astype(str)),
                    labels=np.asarray(cached["labels"], dtype=np.int8),
                    reference=np.asarray(
                        cached["unaligned_reference"], dtype=np.int64
                    ),
                    provisional_reference_count=provisional_count,
                    invalid=invalid,
                )
            )
    names = records[0].geometry_names
    if any(record.geometry_names != names for record in records):
        raise RuntimeError("geometry feature names differ between records")
    return records, excluded, manifest["folds"]


def _compatible_metrics(reference: np.ndarray, detected: np.ndarray, tolerance: int):
    if reference.size == 0:
        return 0.0, 0.0, 0.0, 0
    if detected.size == 0:
        return 0.0, 0.0, 0.0, 0
    insertions = np.searchsorted(detected, reference)
    left = np.clip(insertions - 1, 0, detected.size - 1)
    right = np.clip(insertions, 0, detected.size - 1)
    distances = np.minimum(
        np.abs(reference - detected[left]), np.abs(reference - detected[right])
    )
    correct = int(np.count_nonzero(distances < tolerance))
    sensitivity = correct / reference.size
    positive_predictive_value = correct / detected.size
    denominator = sensitivity + positive_predictive_value
    f1 = (
        2 * sensitivity * positive_predictive_value / denominator
        if denominator
        else 0.0
    )
    return sensitivity, positive_predictive_value, f1, correct


def _score_record(record: Record, detected_values: np.ndarray):
    detected = np.sort(np.unique(np.asarray(detected_values, dtype=np.int64)))
    detected = detected[
        (detected >= 0)
        & (detected < record.sample_count)
        & (~record.invalid[detected])
    ]
    reference, _ = align_ecg_to_ppg(
        record.reference, detected, record.sampling_frequency
    )
    keep = (reference >= 0) & (reference < record.sample_count)
    reference = reference[keep]
    reference = reference[~record.invalid[reference]]
    tolerance = int(round(TOLERANCE_SECONDS * record.sampling_frequency))
    exact = match_events(reference, detected, tolerance_samples=tolerance)
    comp_sens, comp_ppv, comp_f1, comp_correct = _compatible_metrics(
        reference, detected, tolerance
    )
    valid_minutes = record.valid_sample_count / record.sampling_frequency / 60.0
    errors_ms = exact.absolute_errors / record.sampling_frequency * 1000.0
    return {
        "key": record.key,
        "subject": record.subject,
        "activity": record.activity,
        "reference": int(reference.size),
        "detected": int(detected.size),
        "tp": int(exact.true_positive),
        "fp": int(exact.false_positive),
        "fn": int(exact.false_negative),
        "precision": float(exact.precision),
        "recall": float(exact.recall),
        "f1": float(exact.f1),
        "compatible_correct": comp_correct,
        "compatible_sensitivity": float(comp_sens),
        "compatible_ppv": float(comp_ppv),
        "compatible_f1": float(comp_f1),
        "absolute_error_sum_ms": float(np.sum(errors_ms)),
        "absolute_error_count": int(errors_ms.size),
        "median_absolute_error_ms": (
            float(np.median(errors_ms)) if errors_ms.size else None
        ),
        "valid_minutes": float(valid_minutes),
    }


def _summarize(per_record: list[dict[str, object]]) -> dict[str, object]:
    total_tp = sum(int(item["tp"]) for item in per_record)
    total_fp = sum(int(item["fp"]) for item in per_record)
    total_fn = sum(int(item["fn"]) for item in per_record)
    reference = sum(int(item["reference"]) for item in per_record)
    detected = sum(int(item["detected"]) for item in per_record)
    correct = sum(int(item["compatible_correct"]) for item in per_record)
    f1_values = np.asarray([float(item["f1"]) for item in per_record])
    compatible_values = np.asarray(
        [float(item["compatible_f1"]) for item in per_record]
    )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * total_tp / (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn
        else 0.0
    )
    compatible_sensitivity = correct / reference if reference else 0.0
    compatible_ppv = correct / detected if detected else 0.0
    compatible_denominator = compatible_sensitivity + compatible_ppv
    compatible_micro_f1 = (
        2 * compatible_sensitivity * compatible_ppv / compatible_denominator
        if compatible_denominator
        else 0.0
    )
    error_count = sum(int(item["absolute_error_count"]) for item in per_record)
    valid_minutes = sum(float(item["valid_minutes"]) for item in per_record)
    return {
        "records": len(per_record),
        "reference_beats": reference,
        "detected_beats": detected,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "median_record_f1": float(np.median(f1_values)),
        "record_f1_q1": float(np.quantile(f1_values, 0.25)),
        "record_f1_q3": float(np.quantile(f1_values, 0.75)),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(micro_f1),
        "compatible_median_record_f1": float(np.median(compatible_values)),
        "compatible_micro_sensitivity": float(compatible_sensitivity),
        "compatible_micro_ppv": float(compatible_ppv),
        "compatible_micro_f1": float(compatible_micro_f1),
        "mean_absolute_error_ms": (
            sum(float(item["absolute_error_sum_ms"]) for item in per_record)
            / error_count
            if error_count
            else None
        ),
        "false_positives_per_minute": (
            total_fp / valid_minutes if valid_minutes else None
        ),
        "valid_minutes": float(valid_minutes),
        "per_record": per_record,
    }


def score_predictions(
    records: list[Record],
    predictions: dict[str, np.ndarray],
    *,
    include_activities: bool = True,
) -> dict[str, object]:
    with ThreadPoolExecutor(max_workers=min(SCORING_WORKERS, len(records))) as pool:
        per_record = list(
            pool.map(
                _score_record,
                records,
                [predictions[record.key] for record in records],
            )
        )
    report = _summarize(per_record)
    if include_activities:
        report["by_activity"] = {
            activity: _summarize(
                [item for item in per_record if item["activity"] == activity]
            )
            for activity in ACTIVITIES
            if any(item["activity"] == activity for item in per_record)
        }
        hard = [
            float(report["by_activity"][activity]["median_record_f1"])
            for activity in HARD_ACTIVITIES
            if activity in report["by_activity"]
        ]
        report["motion_intensive_macro_median_f1"] = float(np.mean(hard))
    return report


def _compact(report: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"per_record", "by_activity"}
    }


def _rank(report: dict[str, object], tie: float = 0.0) -> tuple[float, ...]:
    return (
        float(report["median_record_f1"]),
        float(report["micro_f1"]),
        float(report["micro_precision"]),
        tie,
    )


def _conditioned(records: list[Record]) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        with np.load(record.raw_file) as raw:
            ppg = np.asarray(raw["ppg"], dtype=float)
        output[record.key] = robust_bandpass(ppg, record.sampling_frequency)
    return output


def _p0_predictions(records, signals, maximum_bpm, prominence):
    output = {}
    for record in records:
        peaks = find_peaks(
            signals[record.key],
            distance=int(round(record.sampling_frequency * 60.0 / maximum_bpm)),
            prominence=prominence,
        )[0]
        output[record.key] = peaks[~record.invalid[peaks]]
    return output


def _p1_predictions(records, minimum_persistence):
    return {
        record.key: suppress_events(
            record.positions,
            record.geometry[:, 120] + 1e-3 * record.geometry[:, 132],
            int(round(0.2 * record.sampling_frequency)),
            threshold=float(minimum_persistence),
        )
        for record in records
    }


def _heartpy_predictions(records, signals):
    import heartpy as hp

    output = {}
    for record in records:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                working, _ = hp.process(
                    signals[record.key],
                    sample_rate=record.sampling_frequency,
                    bpmmin=30,
                    bpmmax=240,
                    calc_freq=False,
                )
            peaks = np.asarray(working["peaklist"], dtype=np.int64)
            peaks = peaks[np.asarray(working["binary_peaklist"], dtype=bool)]
            peaks = peaks[~record.invalid[peaks]]
        except Exception as error:
            print(f"HeartPy failed on {record.key}: {type(error).__name__}: {error}")
            peaks = np.empty(0, dtype=np.int64)
        output[record.key] = peaks
    return output


def _mass_indices(names: tuple[str, ...]) -> np.ndarray:
    per_level = (
        "log1p_polygon_area",
        "signed_polygon_area",
        "log1p_quadratic_energy",
        "log1p_triangle_area",
    )
    cross_level = {
        "area_decay_ratio",
        "energy_decay_ratio",
        "log1p_total_polygon_area",
        "log1p_total_quadratic_energy",
        "log1p_total_triangle_area",
        "max_area_level",
        "max_energy_level",
    }
    return np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name in cross_level
            or any(name.endswith(f"_{suffix}") for suffix in per_level)
        ],
        dtype=np.int64,
    )


def _features(record: Record, feature_set: str) -> np.ndarray:
    if feature_set == "mass":
        return record.geometry[:, _mass_indices(record.geometry_names)]
    if feature_set == "geometry":
        return record.geometry
    if feature_set == "hybrid":
        return np.column_stack([record.geometry, record.morphology])
    raise ValueError(feature_set)


MODEL_SPECS = (
    {
        "name": "hgb_mass_control",
        "family": "hist_gradient_boosting",
        "feature_set": "mass",
        "eligible_primary": False,
        "params": {"learning_rate": 0.1, "max_leaf_nodes": 31},
    },
    {
        "name": "logistic_geometry",
        "family": "logistic",
        "feature_set": "geometry",
        "eligible_primary": True,
        "params": {"C": 0.1},
    },
    {
        "name": "hgb_geometry",
        "family": "hist_gradient_boosting",
        "feature_set": "geometry",
        "eligible_primary": True,
        "params": {"learning_rate": 0.1, "max_leaf_nodes": 31},
    },
    {
        "name": "hgb_hybrid",
        "family": "hist_gradient_boosting",
        "feature_set": "hybrid",
        "eligible_primary": True,
        "params": {"learning_rate": 0.05, "max_leaf_nodes": 15},
    },
)


def _make_model(spec):
    params = dict(spec["params"])
    if spec["family"] == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                class_weight="balanced",
                max_iter=2000,
                random_state=1729,
            ),
        )
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        max_iter=200,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=1729,
    )


def _fit_fold_models(records: list[Record], fold: int):
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    selections = []
    for original in MODEL_SPECS:
        spec = dict(original)
        feature_set = str(spec["feature_set"])
        x = np.vstack([_features(record, feature_set) for record in records])
        y = np.concatenate([record.labels for record in records])
        model = _make_model(spec)
        model.fit(x, y)
        ap = float(average_precision_score(y, model.predict_proba(x)[:, 1]))
        path = MODEL_ROOT / f"fold_{fold}_{spec['name']}.joblib"
        joblib.dump(model, path)
        selection = {
            **spec,
            "input_features": int(x.shape[1]),
            "training_candidates": int(x.shape[0]),
            "training_positive_candidates": int(np.count_nonzero(y)),
            "descriptive_training_average_precision": ap,
            "model_file": str(path.relative_to(PROJECT)).replace("\\", "/"),
            "model_sha256": sha256(path),
        }
        selections.append((selection, model))
        print(f"fold {fold} fit {spec['name']}: train AP={ap:.6f}", flush=True)
        del x, y
    return selections


def _model_predictions(records, model, feature_set, threshold):
    output = {}
    for record in records:
        probabilities = model.predict_proba(_features(record, feature_set))[:, 1]
        output[record.key] = suppress_events(
            record.positions,
            probabilities,
            int(round(0.2 * record.sampling_frequency)),
            threshold=float(threshold),
        )
    return output


def _select_p0(validation, signals):
    grid = []
    for maximum_bpm in (160, 180, 200, 240, 300):
        for prominence in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            metrics = score_predictions(
                validation,
                _p0_predictions(validation, signals, maximum_bpm, prominence),
                include_activities=False,
            )
            grid.append(
                {
                    "maximum_bpm": maximum_bpm,
                    "prominence": prominence,
                    "metrics": _compact(metrics),
                }
            )
    best = max(
        grid,
        key=lambda item: _rank(
            dict(item["metrics"]),
            float(item["prominence"]) - 1e-3 * float(item["maximum_bpm"]),
        ),
    )
    return best, grid


def _select_p1(validation):
    grid = []
    for persistence in range(1, 6):
        metrics = score_predictions(
            validation,
            _p1_predictions(validation, persistence),
            include_activities=False,
        )
        grid.append(
            {
                "minimum_persistence": persistence,
                "metrics": _compact(metrics),
            }
        )
    return max(
        grid,
        key=lambda item: _rank(
            dict(item["metrics"]), float(item["minimum_persistence"])
        ),
    ), grid


def _select_threshold(validation, model, feature_set):
    grid = []
    for threshold in THRESHOLDS:
        metrics = score_predictions(
            validation,
            _model_predictions(validation, model, feature_set, threshold),
            include_activities=False,
        )
        grid.append({"threshold": threshold, "metrics": _compact(metrics)})
    return max(
        grid,
        key=lambda item: _rank(dict(item["metrics"]), float(item["threshold"])),
    ), grid


def _run_fold(records, folds, fold):
    test_subjects = set(folds[fold])
    validation_subjects = set(folds[(fold + 1) % len(folds)])
    development = [
        record
        for record in records
        if record.subject not in test_subjects | validation_subjects
    ]
    validation = [record for record in records if record.subject in validation_subjects]
    test = [record for record in records if record.subject in test_subjects]
    validation_signals = _conditioned(validation)
    selected_p0, p0_grid = _select_p0(validation, validation_signals)
    selected_p1, p1_grid = _select_p1(validation)
    fitted = _fit_fold_models(development, fold)
    model_selections = []
    for selection, model in fitted:
        best, threshold_grid = _select_threshold(
            validation, model, str(selection["feature_set"])
        )
        selected_metrics = score_predictions(
            validation,
            _model_predictions(
                validation,
                model,
                str(selection["feature_set"]),
                float(best["threshold"]),
            ),
        )
        model_selections.append(
            {
                "selection": selection,
                "threshold_grid": threshold_grid,
                "selected": {
                    "threshold": best["threshold"],
                    "metrics": selected_metrics,
                },
            }
        )
        print(
            f"fold {fold} validation {selection['name']}: "
            f"threshold={best['threshold']:.2f}, "
            f"median={selected_metrics['median_record_f1']:.6f}",
            flush=True,
        )
    eligible = [
        item
        for item in model_selections
        if bool(dict(item["selection"])["eligible_primary"])
    ]
    primary = max(
        eligible,
        key=lambda item: (
            *_rank(dict(dict(item["selected"])["metrics"])),
            int(dict(item["selection"])["feature_set"] == "geometry"),
            int(dict(item["selection"])["family"] == "logistic"),
        ),
    )
    lock = {
        "fold": fold,
        "protocol_sha256": PROTOCOL_SHA256,
        "record_manifest_sha256": MANIFEST_SHA256,
        "benchmark_script_sha256": sha256(Path(__file__)),
        "development_subjects": sorted({record.subject for record in development}),
        "validation_subjects": sorted(validation_subjects),
        "test_subjects": sorted(test_subjects),
        "p0": {
            "maximum_bpm": selected_p0["maximum_bpm"],
            "prominence": selected_p0["prominence"],
        },
        "p1": {
            "minimum_persistence": selected_p1["minimum_persistence"]
        },
        "models": [
            {
                **dict(item["selection"]),
                "threshold": float(dict(item["selected"])["threshold"]),
            }
            for item in model_selections
        ],
        "primary_model": str(dict(primary["selection"])["name"]),
    }
    fold_root = RESULT_ROOT / "folds" / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    lock_path = fold_root / "outer_test_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    test_signals = _conditioned(test)
    p0_predictions = _p0_predictions(
        test,
        test_signals,
        int(lock["p0"]["maximum_bpm"]),
        float(lock["p0"]["prominence"]),
    )
    test_baselines = {
        "p0": score_predictions(test, p0_predictions),
        "p1": score_predictions(
            test,
            _p1_predictions(test, int(lock["p1"]["minimum_persistence"])),
        ),
        "heartpy": score_predictions(
            test, _heartpy_predictions(test, test_signals)
        ),
    }
    test_models = {}
    for item, (_, model) in zip(lock["models"], fitted, strict=True):
        metrics = score_predictions(
            test,
            _model_predictions(
                test,
                model,
                str(item["feature_set"]),
                float(item["threshold"]),
            ),
        )
        test_models[str(item["name"])] = metrics
        print(
            f"fold {fold} TEST {item['name']}: "
            f"median={metrics['median_record_f1']:.6f}, "
            f"micro={metrics['micro_f1']:.6f}",
            flush=True,
        )
    summary = {
        "fold": fold,
        "protocol_sha256": PROTOCOL_SHA256,
        "lock_sha256": sha256(lock_path),
        "development_records": len(development),
        "validation_records": len(validation),
        "test_records": len(test),
        "p0_grid": p0_grid,
        "selected_p0": selected_p0,
        "p1_grid": p1_grid,
        "selected_p1": selected_p1,
        "model_validation": model_selections,
        "primary_model": lock["primary_model"],
        "test": {"baselines": test_baselines, "models": test_models},
    }
    summary_path = fold_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"fold {fold} locked primary={lock['primary_model']}; "
        f"summary={sha256(summary_path)}",
        flush=True,
    )
    return summary


def _pooled(fold_summaries, method_type, method_name):
    per_record = []
    for fold in fold_summaries:
        per_record.extend(fold["test"][method_type][method_name]["per_record"])
    report = _summarize(per_record)
    report["by_activity"] = {
        activity: _summarize(
            [item for item in per_record if item["activity"] == activity]
        )
        for activity in ACTIVITIES
        if any(item["activity"] == activity for item in per_record)
    }
    report["motion_intensive_macro_median_f1"] = float(
        np.mean(
            [
                report["by_activity"][activity]["median_record_f1"]
                for activity in HARD_ACTIVITIES
            ]
        )
    )
    return report


def _pooled_primary(fold_summaries):
    per_record = []
    selections = []
    for fold in fold_summaries:
        name = str(fold["primary_model"])
        selections.append(name)
        per_record.extend(fold["test"]["models"][name]["per_record"])
    report = _summarize(per_record)
    report["by_activity"] = {
        activity: _summarize(
            [item for item in per_record if item["activity"] == activity]
        )
        for activity in ACTIVITIES
        if any(item["activity"] == activity for item in per_record)
    }
    report["motion_intensive_macro_median_f1"] = float(
        np.mean(
            [
                report["by_activity"][activity]["median_record_f1"]
                for activity in HARD_ACTIVITIES
            ]
        )
    )
    report["fold_model_selections"] = selections
    return report


def main() -> None:
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("protocol changed after it was frozen")
    if sha256(RECORD_MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("record manifest changed after it was frozen")
    records, excluded, folds = load_records()
    summaries = [_run_fold(records, folds, fold) for fold in range(5)]
    pooled_baselines = {
        name: _pooled(summaries, "baselines", name)
        for name in ("p0", "p1", "heartpy")
    }
    pooled_models = {
        name: _pooled(summaries, "models", name)
        for name in (
            "hgb_mass_control",
            "logistic_geometry",
            "hgb_geometry",
            "hgb_hybrid",
        )
    }
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "record_manifest_sha256": MANIFEST_SHA256,
        "records": len(records),
        "excluded_records": excluded,
        "folds": [
            {
                "fold": summary["fold"],
                "primary_model": summary["primary_model"],
                "lock_sha256": summary["lock_sha256"],
            }
            for summary in summaries
        ],
        "baselines": pooled_baselines,
        "models": pooled_models,
        "cross_fitted_primary": _pooled_primary(summaries),
        "published_context": {
            "status": "PPG-beats compatible metric is a sensitivity analysis; local exact matcher is primary",
            "activity_best_f1_from_2022_benchmark": {
                "sitting": 0.951,
                "working": 0.812,
                "cycling": 0.906,
                "walking": 0.769,
                "lunch_break": 0.668,
                "car_driving": 0.831,
                "stair_climbing": 0.719,
                "table_soccer": 0.653,
            },
        },
    }
    destination = RESULT_ROOT / "nested_subjectwise_summary.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    concise = {
        "records": len(records),
        "excluded_records": len(excluded),
        "primary_model_selections": result["cross_fitted_primary"][
            "fold_model_selections"
        ],
        "primary_median_f1": result["cross_fitted_primary"]["median_record_f1"],
        "primary_micro_f1": result["cross_fitted_primary"]["micro_f1"],
        "summary": str(destination.relative_to(PROJECT)).replace("\\", "/"),
        "sha256": sha256(destination),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
