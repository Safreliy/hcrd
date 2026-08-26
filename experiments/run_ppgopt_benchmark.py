"""Frozen subject-wise PPGopt benchmark for multilevel HCRD event features."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
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

from hcrd.ppg import (
    artifact_mask,
    iter_ppgopt_keys,
    load_ppgopt_recording,
    mask_events,
    match_event_cardinality,
    match_event_pairs,
    match_events,
    robust_bandpass,
    suppress_events,
)


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "raw" / "ppgopt"
FEATURE_ROOT = PROJECT / "results" / "ppgopt" / "features"
RESULT_ROOT = PROJECT / "results" / "ppgopt"
MODEL_ROOT = RESULT_ROOT / "models"
PROTOCOL = PROJECT / "docs" / "ppgopt_hcrd_protocol.md"
PROTOCOL_SHA256 = "1e33526deb7e3f24f426c8a2d92506c203e17bcbdee56200b2bde11388cbc782"
PHASE_SUBJECTS = {
    "development": (1, 2, 3),
    "validation": (4, 5),
    "confirmation": (6, 7),
}


@dataclass
class Record:
    key: str
    subject: int
    activity: str
    trial: int
    sampling_frequency: float
    sample_count: int
    valid_sample_count: int
    truth: np.ndarray
    artifacts: np.ndarray
    positions: np.ndarray
    geometry: np.ndarray
    morphology: np.ndarray
    labels: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_keep(positions: np.ndarray, artifacts: np.ndarray) -> np.ndarray:
    keep = np.ones(positions.size, dtype=bool)
    for left, right in np.asarray(artifacts, dtype=np.int64).reshape(-1, 2):
        keep &= (positions < left) | (positions > right)
    return keep


def load_records(subjects: tuple[int, ...]) -> list[Record]:
    records = []
    for subject, activity, trial in iter_ppgopt_keys():
        if subject not in subjects:
            continue
        recording = load_ppgopt_recording(DATA_ROOT, subject, activity, trial)
        truth = mask_events(recording.peaks, recording.artifacts)
        if truth.size == 0:
            continue
        cache_path = FEATURE_ROOT / f"{recording.key}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing {cache_path}; run prepare_ppgopt_features.py first"
            )
        with np.load(cache_path) as cached:
            positions = np.asarray(cached["positions"], dtype=np.int64)
            keep = _candidate_keep(positions, recording.artifacts)
            positions = positions[keep]
            geometry = np.asarray(cached["geometry"], dtype=float)[keep]
            morphology = np.asarray(cached["morphology"], dtype=float)[keep]
        pairs = match_event_pairs(
            truth,
            positions,
            tolerance_samples=int(round(0.250 * recording.sampling_frequency)),
        )
        labels = np.zeros(positions.size, dtype=np.int8)
        if pairs.size:
            labels[pairs[:, 1]] = 1
        invalid = artifact_mask(recording.signal.size, recording.artifacts)
        records.append(
            Record(
                key=recording.key,
                subject=subject,
                activity=activity,
                trial=trial,
                sampling_frequency=recording.sampling_frequency,
                sample_count=recording.signal.size,
                valid_sample_count=int(np.count_nonzero(~invalid)),
                truth=truth,
                artifacts=recording.artifacts,
                positions=positions,
                geometry=geometry,
                morphology=morphology,
                labels=labels,
            )
        )
    return records


def _score_group(
    records: list[Record],
    predictions: dict[str, np.ndarray],
    *,
    cardinality_only: bool = False,
) -> dict[str, object]:
    true_positive = false_positive = false_negative = 0
    errors: list[np.ndarray] = []
    per_record = []
    valid_minutes = 0.0
    for record in records:
        detected = np.asarray(predictions[record.key], dtype=np.int64)
        tolerance = int(round(0.250 * record.sampling_frequency))
        if cardinality_only:
            matched = match_event_cardinality(record.truth, detected, tolerance)
            record_tp = matched
            record_fp = int(detected.size - matched)
            record_fn = int(record.truth.size - matched)
            record_f1_denominator = 2 * record_tp + record_fp + record_fn
            record_f1 = (
                2 * record_tp / record_f1_denominator
                if record_f1_denominator
                else 0.0
            )
        else:
            result = match_events(record.truth, detected, tolerance_samples=tolerance)
            record_tp = result.true_positive
            record_fp = result.false_positive
            record_fn = result.false_negative
            record_f1 = result.f1
            errors.append(result.absolute_errors / record.sampling_frequency * 1000.0)
        true_positive += record_tp
        false_positive += record_fp
        false_negative += record_fn
        valid_minutes += record.valid_sample_count / record.sampling_frequency / 60.0
        per_record.append(
            {
                "key": record.key,
                "subject": record.subject,
                "activity": record.activity,
                "reference": int(record.truth.size),
                "detected": int(detected.size),
                "tp": record_tp,
                "fp": record_fp,
                "fn": record_fn,
                "f1": record_f1,
            }
        )
    denominator_precision = true_positive + false_positive
    denominator_recall = true_positive + false_negative
    precision = true_positive / denominator_precision if denominator_precision else 0.0
    recall = true_positive / denominator_recall if denominator_recall else 0.0
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
    all_errors = np.concatenate(errors) if errors else np.empty(0)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_error_ms": float(np.mean(all_errors)) if all_errors.size else None,
        "median_absolute_error_ms": float(np.median(all_errors)) if all_errors.size else None,
        "false_positives_per_minute": false_positive / valid_minutes if valid_minutes else None,
        "valid_minutes": valid_minutes,
        "per_record": per_record,
    }


def score_predictions(
    records: list[Record],
    predictions: dict[str, np.ndarray],
    *,
    cardinality_only: bool = False,
) -> dict[str, object]:
    report = _score_group(records, predictions, cardinality_only=cardinality_only)
    report["by_subject"] = {
        str(subject): _score_group(
            [record for record in records if record.subject == subject],
            predictions,
            cardinality_only=cardinality_only,
        )
        for subject in sorted({record.subject for record in records})
    }
    report["by_activity"] = {
        activity: _score_group(
            [record for record in records if record.activity == activity],
            predictions,
            cardinality_only=cardinality_only,
        )
        for activity in ("rest", "squat", "step")
        if any(record.activity == activity for record in records)
    }
    return report


def _conditioned_signals(records: list[Record]) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        raw = load_ppgopt_recording(
            DATA_ROOT,
            record.subject,
            record.activity,
            record.trial,
            load_annotations=False,
        )
        output[record.key] = robust_bandpass(raw.signal, record.sampling_frequency)
    return output


def _baseline_predictions(
    records: list[Record],
    distance_bpm: int,
    prominence: float,
    conditioned_signals: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        conditioned = conditioned_signals[record.key]
        distance = int(round(record.sampling_frequency * 60.0 / distance_bpm))
        peaks = find_peaks(conditioned, distance=distance, prominence=prominence)[0]
        output[record.key] = mask_events(peaks, record.artifacts)
    return output


def _heartpy_predictions(
    records: list[Record], conditioned_signals: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    import heartpy as hp

    output = {}
    for record in records:
        conditioned = conditioned_signals[record.key]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                working, _ = hp.process(
                    conditioned,
                    sample_rate=record.sampling_frequency,
                    bpmmin=30,
                    bpmmax=200,
                    calc_freq=False,
                )
            peaks = np.asarray(working["peaklist"], dtype=np.int64)
            accepted = np.asarray(working["binary_peaklist"], dtype=bool)
            peaks = peaks[accepted]
        except Exception as error:  # HeartPy explicitly fails on some noisy traces.
            print(f"HeartPy failed on {record.key}: {type(error).__name__}: {error}")
            peaks = np.empty(0, dtype=np.int64)
        output[record.key] = mask_events(peaks, record.artifacts)
    return output


def _persistence_predictions(
    records: list[Record], minimum_persistence: int
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        support = record.geometry[:, 120]
        total_amplitude = record.geometry[:, 132]
        score = support + 1e-3 * total_amplitude
        output[record.key] = suppress_events(
            record.positions,
            score,
            int(round(0.300 * record.sampling_frequency)),
            threshold=float(minimum_persistence),
        )
    return output


def _features(record: Record, feature_set: str) -> np.ndarray:
    if feature_set == "geometry":
        return record.geometry
    if feature_set == "hybrid":
        return np.column_stack([record.geometry, record.morphology])
    raise ValueError(feature_set)


def _make_model(family: str, params: dict[str, object]):
    if family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                class_weight="balanced",
                max_iter=2000,
                random_state=1729,
            ),
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            max_iter=200,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=1729,
        )
    raise ValueError(family)


def _candidate_arrays(records: list[Record], feature_set: str):
    return (
        np.vstack([_features(record, feature_set) for record in records]),
        np.concatenate([record.labels for record in records]),
    )


def _grid_specs() -> dict[str, list[dict[str, object]]]:
    return {
        "logistic_geometry": [
            {
                "name": "logistic_geometry",
                "family": "logistic",
                "feature_set": "geometry",
                "params": {"C": value},
            }
            for value in (0.1, 1.0, 10.0)
        ],
        "hgb_geometry": [
            {
                "name": "hgb_geometry",
                "family": "hist_gradient_boosting",
                "feature_set": "geometry",
                "params": {"learning_rate": rate, "max_leaf_nodes": leaves},
            }
            for rate in (0.05, 0.1)
            for leaves in (15, 31)
        ],
        "hgb_hybrid": [
            {
                "name": "hgb_hybrid",
                "family": "hist_gradient_boosting",
                "feature_set": "hybrid",
                "params": {"learning_rate": rate, "max_leaf_nodes": leaves},
            }
            for rate in (0.05, 0.1)
            for leaves in (15, 31)
        ],
    }


def _select_models(records: list[Record]) -> tuple[list[dict[str, object]], dict[str, object]]:
    selections = []
    cv_report: dict[str, object] = {}
    subjects = sorted({record.subject for record in records})
    for model_name, specs in _grid_specs().items():
        candidates = []
        for spec in specs:
            out_of_fold_probabilities = []
            out_of_fold_labels = []
            fold_scores = []
            for held_out in subjects:
                training = [record for record in records if record.subject != held_out]
                testing = [record for record in records if record.subject == held_out]
                x_train, y_train = _candidate_arrays(training, str(spec["feature_set"]))
                x_test, y_test = _candidate_arrays(testing, str(spec["feature_set"]))
                model = _make_model(str(spec["family"]), dict(spec["params"]))
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_test)[:, 1]
                out_of_fold_probabilities.append(probabilities)
                out_of_fold_labels.append(y_test)
                fold_scores.append(float(average_precision_score(y_test, probabilities)))
            pooled_labels = np.concatenate(out_of_fold_labels)
            pooled_probabilities = np.concatenate(out_of_fold_probabilities)
            candidate = {
                **spec,
                "pooled_average_precision": float(
                    average_precision_score(pooled_labels, pooled_probabilities)
                ),
                "fold_average_precision": fold_scores,
            }
            candidates.append(candidate)
            print(
                f"CV {model_name} {spec['params']}: "
                f"AP={candidate['pooled_average_precision']:.6f}",
                flush=True,
            )
        best = max(
            candidates,
            key=lambda item: (
                float(item["pooled_average_precision"]),
                -float(dict(item["params"]).get("C", 0.0)),
                -float(dict(item["params"]).get("learning_rate", 0.0)),
                -int(dict(item["params"]).get("max_leaf_nodes", 0)),
            ),
        )
        selections.append(best)
        cv_report[model_name] = candidates
        x_all, y_all = _candidate_arrays(records, str(best["feature_set"]))
        model = _make_model(str(best["family"]), dict(best["params"]))
        model.fit(x_all, y_all)
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        destination = MODEL_ROOT / f"{model_name}.joblib"
        joblib.dump(model, destination)
        best["model_file"] = str(destination.relative_to(PROJECT)).replace("\\", "/")
        best["model_sha256"] = sha256(destination)
    return selections, cv_report


def _model_predictions(
    records: list[Record],
    model,
    feature_set: str,
    threshold: float,
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        probabilities = model.predict_proba(_features(record, feature_set))[:, 1]
        output[record.key] = suppress_events(
            record.positions,
            probabilities,
            int(round(0.300 * record.sampling_frequency)),
            threshold=threshold,
        )
    return output


def _run_development(records: list[Record]) -> dict[str, object]:
    conditioned_signals = _conditioned_signals(records)
    p0_grid = []
    for distance_bpm in (200, 180, 160, 140):
        for prominence in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            report = score_predictions(
                records,
                _baseline_predictions(
                    records, distance_bpm, prominence, conditioned_signals
                ),
                cardinality_only=True,
            )
            p0_grid.append(
                {
                    "distance_bpm": distance_bpm,
                    "prominence": prominence,
                    "metrics": report,
                }
            )
            print(
                f"P0 bpm={distance_bpm} prominence={prominence}: F1={report['f1']:.6f}",
                flush=True,
            )
    best_p0 = max(
        p0_grid,
        key=lambda item: (
            float(dict(item["metrics"])["f1"]),
            float(dict(item["metrics"])["precision"]),
            float(item["prominence"]),
            -int(item["distance_bpm"]),
        ),
    )
    best_p0["selection_metrics"] = best_p0["metrics"]
    best_p0["metrics"] = score_predictions(
        records,
        _baseline_predictions(
            records,
            int(best_p0["distance_bpm"]),
            float(best_p0["prominence"]),
            conditioned_signals,
        ),
    )
    p1_grid = []
    for persistence in range(1, 6):
        report = score_predictions(
            records,
            _persistence_predictions(records, persistence),
            cardinality_only=True,
        )
        p1_grid.append({"minimum_persistence": persistence, "metrics": report})
        print(f"P1 persistence={persistence}: F1={report['f1']:.6f}", flush=True)
    best_p1 = max(
        p1_grid,
        key=lambda item: (
            float(dict(item["metrics"])["f1"]),
            float(dict(item["metrics"])["precision"]),
            int(item["minimum_persistence"]),
        ),
    )
    best_p1["selection_metrics"] = best_p1["metrics"]
    best_p1["metrics"] = score_predictions(
        records,
        _persistence_predictions(records, int(best_p1["minimum_persistence"])),
    )
    all_candidates = score_predictions(
        records, {record.key: record.positions for record in records}
    )
    heartpy_report = score_predictions(
        records, _heartpy_predictions(records, conditioned_signals)
    )
    models, cv_report = _select_models(records)
    return {
        "phase": "development",
        "protocol_sha256": PROTOCOL_SHA256,
        "subjects": PHASE_SUBJECTS["development"],
        "candidate_upper_bound": all_candidates,
        "p0_grid": p0_grid,
        "selected_p0": best_p0,
        "p1_grid": p1_grid,
        "selected_p1": best_p1,
        "heartpy": heartpy_report,
        "selected_models": models,
        "model_cv": cv_report,
    }


def _load_development() -> dict[str, object]:
    path = RESULT_ROOT / "development" / "summary.json"
    if not path.exists():
        raise FileNotFoundError("development summary is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_baselines(
    records: list[Record], development: dict[str, object]
) -> dict[str, object]:
    p0 = dict(development["selected_p0"])
    p1 = dict(development["selected_p1"])
    conditioned_signals = _conditioned_signals(records)
    return {
        "p0": {
            "parameters": {
                "distance_bpm": p0["distance_bpm"],
                "prominence": p0["prominence"],
            },
            "metrics": score_predictions(
                records,
                _baseline_predictions(
                    records,
                    int(p0["distance_bpm"]),
                    float(p0["prominence"]),
                    conditioned_signals,
                ),
            ),
        },
        "p1": {
            "parameters": {"minimum_persistence": p1["minimum_persistence"]},
            "metrics": score_predictions(
                records,
                _persistence_predictions(records, int(p1["minimum_persistence"])),
            ),
        },
        "heartpy": {
            "metrics": score_predictions(
                records, _heartpy_predictions(records, conditioned_signals)
            )
        },
    }


def _run_validation(records: list[Record]) -> dict[str, object]:
    development = _load_development()
    baselines = _fixed_baselines(records, development)
    selected_models = list(development["selected_models"])
    model_reports = []
    thresholds = np.arange(0.05, 1.0, 0.05)
    for selection in selected_models:
        model_path = PROJECT / str(selection["model_file"])
        if sha256(model_path) != selection["model_sha256"]:
            raise RuntimeError(f"model hash mismatch: {model_path}")
        model = joblib.load(model_path)
        grid = []
        for threshold in thresholds:
            report = score_predictions(
                records,
                _model_predictions(
                    records, model, str(selection["feature_set"]), float(threshold)
                ),
                cardinality_only=True,
            )
            grid.append({"threshold": float(threshold), "metrics": report})
        best = max(
            grid,
            key=lambda item: (
                float(dict(item["metrics"])["f1"]),
                float(dict(item["metrics"])["precision"]),
                float(item["threshold"]),
            ),
        )
        best["selection_metrics"] = best["metrics"]
        best["metrics"] = score_predictions(
            records,
            _model_predictions(
                records,
                model,
                str(selection["feature_set"]),
                float(best["threshold"]),
            ),
        )
        print(
            f"validation {selection['name']}: threshold={best['threshold']:.2f}, "
            f"F1={dict(best['metrics'])['f1']:.6f}",
            flush=True,
        )
        model_reports.append({"selection": selection, "threshold_grid": grid, "selected": best})
    primary = max(
        model_reports,
        key=lambda item: (
            float(dict(dict(item["selected"])["metrics"])["f1"]),
            int(str(dict(item["selection"])["feature_set"]) == "geometry"),
            int(str(dict(item["selection"])["family"]) == "logistic"),
        ),
    )
    frozen = {
        "protocol_sha256": PROTOCOL_SHA256,
        "created_after_validation_subjects": PHASE_SUBJECTS["validation"],
        "confirmation_subjects": PHASE_SUBJECTS["confirmation"],
        "p0": baselines["p0"]["parameters"],
        "p1": baselines["p1"]["parameters"],
        "models": [
            {
                **dict(item["selection"]),
                "threshold": float(dict(item["selected"])["threshold"]),
            }
            for item in model_reports
        ],
        "primary_model": str(dict(primary["selection"])["name"]),
        "benchmark_script_sha256": sha256(Path(__file__)),
    }
    frozen_path = RESULT_ROOT / "frozen_confirmation_rule.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return {
        "phase": "validation",
        "protocol_sha256": PROTOCOL_SHA256,
        "subjects": PHASE_SUBJECTS["validation"],
        "baselines": baselines,
        "models": model_reports,
        "primary_model": frozen["primary_model"],
        "frozen_confirmation_rule": str(frozen_path.relative_to(PROJECT)).replace("\\", "/"),
        "frozen_confirmation_rule_sha256": sha256(frozen_path),
    }


def _run_confirmation(records: list[Record]) -> dict[str, object]:
    development = _load_development()
    frozen_path = RESULT_ROOT / "frozen_confirmation_rule.json"
    if not frozen_path.exists():
        raise FileNotFoundError("confirmation is locked until validation freezes a rule")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("protocol hash mismatch")
    baselines = _fixed_baselines(records, development)
    model_reports = []
    for selection in frozen["models"]:
        model_path = PROJECT / str(selection["model_file"])
        if sha256(model_path) != selection["model_sha256"]:
            raise RuntimeError(f"model hash mismatch: {model_path}")
        model = joblib.load(model_path)
        report = score_predictions(
            records,
            _model_predictions(
                records,
                model,
                str(selection["feature_set"]),
                float(selection["threshold"]),
            ),
        )
        model_reports.append(
            {
                "name": selection["name"],
                "feature_set": selection["feature_set"],
                "threshold": selection["threshold"],
                "metrics": report,
            }
        )
        print(f"confirmation {selection['name']}: F1={report['f1']:.6f}", flush=True)
    return {
        "phase": "confirmation",
        "protocol_sha256": PROTOCOL_SHA256,
        "subjects": PHASE_SUBJECTS["confirmation"],
        "frozen_confirmation_rule_sha256": sha256(frozen_path),
        "baselines": baselines,
        "models": model_reports,
        "primary_model": frozen["primary_model"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASE_SUBJECTS, required=True)
    args = parser.parse_args()
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("protocol changed after the benchmark was frozen")
    if args.phase == "confirmation" and not (
        RESULT_ROOT / "frozen_confirmation_rule.json"
    ).exists():
        raise RuntimeError("confirmation labels remain locked until validation is complete")
    records = load_records(PHASE_SUBJECTS[args.phase])
    if args.phase == "development":
        summary = _run_development(records)
    elif args.phase == "validation":
        summary = _run_validation(records)
    else:
        summary = _run_confirmation(records)
    destination = RESULT_ROOT / args.phase
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    concise = {
        "phase": args.phase,
        "subjects": PHASE_SUBJECTS[args.phase],
        "recordings": len(records),
        "expert_events": sum(record.truth.size for record in records),
        "candidates": sum(record.positions.size for record in records),
        "summary": str(summary_path.relative_to(PROJECT)).replace("\\", "/"),
        "sha256": sha256(summary_path),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
