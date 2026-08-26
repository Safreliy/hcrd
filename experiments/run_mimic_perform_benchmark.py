"""Locked MIMIC PERform benchmark for the full multilevel HCRD representation."""

from __future__ import annotations

import argparse
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
from hcrd.ppg_benchmark import align_ecg_to_ppg, flatline_mask


PROJECT = Path(__file__).resolve().parents[1]
RECORD_MANIFEST = PROJECT / "data" / "manifests" / "mimic_perform_records.json"
FEATURE_ROOT = PROJECT / "results" / "mimic_perform" / "features"
RESULT_ROOT = PROJECT / "results" / "mimic_perform"
MODEL_ROOT = RESULT_ROOT / "models"
PROTOCOL = PROJECT / "docs" / "mimic_perform_hcrd_protocol.md"
PROTOCOL_SHA256 = "510e47b365ef2e284ebd7adb8d5ecec4bd41ed2b4492a6c7ca6531e4cb76a511"
MANIFEST_SHA256 = "8d2511c3ce018ed5094b8a03fcaf95b652cb5cec051822e0cea68a17a675f517"
TOLERANCE_SECONDS = 0.15
MINIMUM_REFERENCE_BEATS = 3
THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))
SCORING_WORKERS = max(1, min(8, os.cpu_count() or 1))


@dataclass
class Record:
    key: str
    record_id: str
    subject_id: str
    group: str
    raw_file: Path
    sampling_frequency: float
    sample_count: int
    valid_sample_count: int
    detector_agreement: float
    positions: np.ndarray
    geometry: np.ndarray
    morphology: np.ndarray
    geometry_names: tuple[str, ...]
    labels: np.ndarray
    unaligned_reference: np.ndarray
    aligned_reference_p0_count: int
    ppg_invalid: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(row: dict[str, object]) -> str:
    return f"{row['source_split']}_{Path(str(row['file'])).stem}"


def load_records(phase: str) -> tuple[list[Record], list[dict[str, object]]]:
    manifest = json.loads(RECORD_MANIFEST.read_text(encoding="utf-8"))
    records: list[Record] = []
    excluded: list[dict[str, object]] = []
    for row in manifest["records"]:
        if row["phase"] != phase:
            continue
        key = _cache_key(row)
        cache_path = FEATURE_ROOT / f"{key}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing {cache_path}; run prepare_mimic_perform_features.py"
            )
        with np.load(cache_path) as cached:
            aligned_count = int(cached["aligned_reference_p0"].size)
            if aligned_count < MINIMUM_REFERENCE_BEATS:
                excluded.append(
                    {
                        "key": key,
                        "record_id": str(row["record_id"]),
                        "group": str(row["group"]),
                        "aligned_reference_p0": aligned_count,
                    }
                )
                continue
            invalid = np.asarray(cached["ppg_invalid_mask"], dtype=bool)
            records.append(
                Record(
                    key=key,
                    record_id=str(row["record_id"]),
                    subject_id=str(row["subject_id"]),
                    group=str(row["group"]),
                    raw_file=PROJECT / str(row["file"]),
                    sampling_frequency=float(cached["sampling_frequency"]),
                    sample_count=int(cached["sample_count"]),
                    valid_sample_count=int(np.count_nonzero(~invalid)),
                    detector_agreement=float(cached["agreement_fraction"]),
                    positions=np.asarray(cached["positions"], dtype=np.int64),
                    geometry=np.asarray(cached["geometry"], dtype=np.float32),
                    morphology=np.asarray(cached["morphology"], dtype=np.float32),
                    geometry_names=tuple(cached["geometry_names"].astype(str)),
                    labels=np.asarray(cached["labels"], dtype=np.int8),
                    unaligned_reference=np.asarray(
                        cached["unaligned_reference"], dtype=np.int64
                    ),
                    aligned_reference_p0_count=aligned_count,
                    ppg_invalid=invalid,
                )
            )
    if not records:
        raise RuntimeError(f"no evaluable records for phase {phase}")
    expected_names = records[0].geometry_names
    if any(record.geometry_names != expected_names for record in records):
        raise RuntimeError("geometry feature names differ between records")
    return records, excluded


def _aligned_reference(record: Record, detected: np.ndarray) -> np.ndarray:
    aligned, _ = align_ecg_to_ppg(
        record.unaligned_reference, detected, record.sampling_frequency
    )
    keep = (aligned >= 0) & (aligned < record.sample_count)
    aligned = aligned[keep]
    return aligned[~record.ppg_invalid[aligned]]


def _score_record(
    record: Record, detected_values: np.ndarray
) -> tuple[dict[str, object], np.ndarray, float]:
    detected = np.asarray(detected_values, dtype=np.int64)
    detected = detected[
        (detected >= 0)
        & (detected < record.sample_count)
        & (~record.ppg_invalid[detected])
    ]
    reference = _aligned_reference(record, detected)
    result = match_events(
        reference,
        detected,
        tolerance_samples=int(round(TOLERANCE_SECONDS * record.sampling_frequency)),
    )
    item = {
        "key": record.key,
        "record_id": record.record_id,
        "subject_id": record.subject_id,
        "group": record.group,
        "reference": int(reference.size),
        "detected": int(detected.size),
        "tp": int(result.true_positive),
        "fp": int(result.false_positive),
        "fn": int(result.false_negative),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
        "detector_agreement": record.detector_agreement,
    }
    return (
        item,
        result.absolute_errors / record.sampling_frequency * 1000.0,
        record.valid_sample_count / record.sampling_frequency / 60.0,
    )


def _score_group(
    records: list[Record], predictions: dict[str, np.ndarray]
) -> dict[str, object]:
    with ThreadPoolExecutor(max_workers=min(SCORING_WORKERS, len(records))) as pool:
        outcomes = list(
            pool.map(
                _score_record,
                records,
                [predictions[record.key] for record in records],
            )
        )
    per_record = [item for item, _, _ in outcomes]
    errors = [error for _, error, _ in outcomes]
    valid_minutes = float(sum(minutes for _, _, minutes in outcomes))
    total_tp = sum(int(item["tp"]) for item in per_record)
    total_fp = sum(int(item["fp"]) for item in per_record)
    total_fn = sum(int(item["fn"]) for item in per_record)
    precision_denominator = total_tp + total_fp
    recall_denominator = total_tp + total_fn
    f1_denominator = 2 * total_tp + total_fp + total_fn
    precision = total_tp / precision_denominator if precision_denominator else 0.0
    recall = total_tp / recall_denominator if recall_denominator else 0.0
    micro_f1 = 2 * total_tp / f1_denominator if f1_denominator else 0.0
    f1_values = np.asarray([float(item["f1"]) for item in per_record])
    all_errors = np.concatenate(errors) if errors else np.empty(0)
    return {
        "records": len(records),
        "reference_beats": int(
            sum(int(item["reference"]) for item in per_record)
        ),
        "detected_beats": int(sum(int(item["detected"]) for item in per_record)),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "median_record_f1": float(np.median(f1_values)),
        "record_f1_q1": float(np.quantile(f1_values, 0.25)),
        "record_f1_q3": float(np.quantile(f1_values, 0.75)),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(micro_f1),
        "mean_absolute_error_ms": (
            float(np.mean(all_errors)) if all_errors.size else None
        ),
        "median_absolute_error_ms": (
            float(np.median(all_errors)) if all_errors.size else None
        ),
        "false_positives_per_minute": (
            float(total_fp / valid_minutes) if valid_minutes else None
        ),
        "valid_minutes": float(valid_minutes),
        "median_detector_agreement": float(
            np.median([record.detector_agreement for record in records])
        ),
        "per_record": per_record,
    }


def score_predictions(
    records: list[Record],
    predictions: dict[str, np.ndarray],
    *,
    include_groups: bool = True,
) -> dict[str, object]:
    report = _score_group(records, predictions)
    if include_groups:
        report["by_group"] = {
            group: _score_group(
                [record for record in records if record.group == group], predictions
            )
            for group in ("a", "n")
            if any(record.group == group for record in records)
        }
    return report


def _rank(report: dict[str, object], final_tie: float = 0.0) -> tuple[float, ...]:
    return (
        float(report["median_record_f1"]),
        float(report["micro_f1"]),
        float(report["micro_precision"]),
        final_tie,
    )


def _conditioned_signals(records: list[Record]) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        with np.load(record.raw_file) as raw:
            ppg = np.asarray(raw["ppg"], dtype=float)
        output[record.key] = robust_bandpass(ppg, record.sampling_frequency)
    return output


def _p0_predictions(
    records: list[Record],
    conditioned: dict[str, np.ndarray],
    maximum_bpm: int,
    prominence: float,
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        peaks = find_peaks(
            conditioned[record.key],
            distance=int(round(record.sampling_frequency * 60.0 / maximum_bpm)),
            prominence=prominence,
        )[0]
        output[record.key] = peaks[~record.ppg_invalid[peaks]]
    return output


def _heartpy_predictions(
    records: list[Record], conditioned: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    import heartpy as hp

    output = {}
    for record in records:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                working, _ = hp.process(
                    conditioned[record.key],
                    sample_rate=record.sampling_frequency,
                    bpmmin=30,
                    bpmmax=300,
                    calc_freq=False,
                )
            peaks = np.asarray(working["peaklist"], dtype=np.int64)
            peaks = peaks[np.asarray(working["binary_peaklist"], dtype=bool)]
            peaks = peaks[~record.ppg_invalid[peaks]]
        except Exception as error:
            print(f"HeartPy failed on {record.key}: {type(error).__name__}: {error}")
            peaks = np.empty(0, dtype=np.int64)
        output[record.key] = peaks
    return output


def _p1_predictions(
    records: list[Record], minimum_persistence: int
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        support = record.geometry[:, 120]
        total_amplitude = record.geometry[:, 132]
        output[record.key] = suppress_events(
            record.positions,
            support + 1e-3 * total_amplitude,
            int(round(0.2 * record.sampling_frequency)),
            threshold=float(minimum_persistence),
        )
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
    indices = [
        index
        for index, name in enumerate(names)
        if name in cross_level or any(name.endswith(f"_{suffix}") for suffix in per_level)
    ]
    return np.asarray(indices, dtype=np.int64)


def _features(record: Record, feature_set: str) -> np.ndarray:
    if feature_set == "mass":
        return record.geometry[:, _mass_indices(record.geometry_names)]
    if feature_set == "geometry":
        return record.geometry
    if feature_set == "hybrid":
        return np.column_stack([record.geometry, record.morphology])
    raise ValueError(feature_set)


def _candidate_arrays(
    records: list[Record], feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([_features(record, feature_set) for record in records]),
        np.concatenate([record.labels for record in records]),
    )


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


def _make_model(spec: dict[str, object]):
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


def _fit_models(records: list[Record]) -> list[dict[str, object]]:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    selections = []
    for original in MODEL_SPECS:
        spec = dict(original)
        x_train, y_train = _candidate_arrays(records, str(spec["feature_set"]))
        model = _make_model(spec)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_train)[:, 1]
        candidate_ap = float(average_precision_score(y_train, probabilities))
        destination = MODEL_ROOT / f"mimic_{spec['name']}.joblib"
        joblib.dump(model, destination)
        selection = {
            **spec,
            "input_features": int(x_train.shape[1]),
            "training_candidates": int(x_train.shape[0]),
            "training_positive_candidates": int(np.count_nonzero(y_train)),
            "descriptive_training_average_precision": candidate_ap,
            "model_file": str(destination.relative_to(PROJECT)).replace("\\", "/"),
            "model_sha256": sha256(destination),
        }
        selections.append(selection)
        print(
            f"fit {spec['name']}: {x_train.shape[1]} features, "
            f"descriptive train AP={candidate_ap:.6f}",
            flush=True,
        )
        del x_train, y_train, probabilities, model
    return selections


def _model_predictions(
    records: list[Record], model, feature_set: str, threshold: float
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        probabilities = model.predict_proba(_features(record, feature_set))[:, 1]
        output[record.key] = suppress_events(
            record.positions,
            probabilities,
            int(round(0.2 * record.sampling_frequency)),
            threshold=threshold,
        )
    return output


def _run_development(records: list[Record]) -> dict[str, object]:
    conditioned = _conditioned_signals(records)
    p0_grid = []
    for maximum_bpm in (180, 200, 240, 300):
        for prominence in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            report = score_predictions(
                records,
                _p0_predictions(
                    records, conditioned, maximum_bpm, float(prominence)
                ),
                include_groups=False,
            )
            item = {
                "maximum_bpm": maximum_bpm,
                "prominence": prominence,
                "metrics": report,
            }
            p0_grid.append(item)
            print(
                f"P0 bpm={maximum_bpm} prominence={prominence}: "
                f"median={report['median_record_f1']:.6f} "
                f"micro={report['micro_f1']:.6f}",
                flush=True,
            )
    best_p0 = max(
        p0_grid,
        key=lambda item: _rank(
            dict(item["metrics"]),
            float(item["prominence"]) - 1e-3 * float(item["maximum_bpm"]),
        ),
    )
    best_p0["selection_metrics"] = best_p0["metrics"]
    best_p0["metrics"] = score_predictions(
        records,
        _p0_predictions(
            records,
            conditioned,
            int(best_p0["maximum_bpm"]),
            float(best_p0["prominence"]),
        ),
    )
    p1_grid = []
    for minimum_persistence in range(1, 6):
        report = score_predictions(
            records,
            _p1_predictions(records, minimum_persistence),
            include_groups=False,
        )
        item = {
            "minimum_persistence": minimum_persistence,
            "metrics": report,
        }
        p1_grid.append(item)
        print(
            f"P1 persistence={minimum_persistence}: "
            f"median={report['median_record_f1']:.6f} "
            f"micro={report['micro_f1']:.6f}",
            flush=True,
        )
    best_p1 = max(
        p1_grid,
        key=lambda item: _rank(
            dict(item["metrics"]), float(item["minimum_persistence"])
        ),
    )
    best_p1["selection_metrics"] = best_p1["metrics"]
    best_p1["metrics"] = score_predictions(
        records,
        _p1_predictions(records, int(best_p1["minimum_persistence"])),
    )
    heartpy = score_predictions(records, _heartpy_predictions(records, conditioned))
    del conditioned
    models = _fit_models(records)
    return {
        "phase": "development",
        "protocol_sha256": PROTOCOL_SHA256,
        "records": len(records),
        "p0_grid": p0_grid,
        "selected_p0": best_p0,
        "p1_grid": p1_grid,
        "selected_p1": best_p1,
        "heartpy": heartpy,
        "models": models,
        "mass_feature_names": [
            records[0].geometry_names[index]
            for index in _mass_indices(records[0].geometry_names)
        ],
    }


def _load_development() -> dict[str, object]:
    path = RESULT_ROOT / "development" / "summary.json"
    if not path.exists():
        raise FileNotFoundError("development summary is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_baselines(
    records: list[Record], development: dict[str, object]
) -> dict[str, object]:
    selected_p0 = dict(development["selected_p0"])
    selected_p1 = dict(development["selected_p1"])
    conditioned = _conditioned_signals(records)
    reports = {
        "p0": {
            "parameters": {
                "maximum_bpm": int(selected_p0["maximum_bpm"]),
                "prominence": float(selected_p0["prominence"]),
            },
            "metrics": score_predictions(
                records,
                _p0_predictions(
                    records,
                    conditioned,
                    int(selected_p0["maximum_bpm"]),
                    float(selected_p0["prominence"]),
                ),
            ),
        },
        "p1": {
            "parameters": {
                "minimum_persistence": int(selected_p1["minimum_persistence"])
            },
            "metrics": score_predictions(
                records,
                _p1_predictions(
                    records, int(selected_p1["minimum_persistence"])
                ),
            ),
        },
        "heartpy": {
            "metrics": score_predictions(
                records, _heartpy_predictions(records, conditioned)
            )
        },
    }
    return reports


def _run_validation(records: list[Record]) -> dict[str, object]:
    development = _load_development()
    baselines = _fixed_baselines(records, development)
    model_reports = []
    for selection_object in development["models"]:
        selection = dict(selection_object)
        model_path = PROJECT / str(selection["model_file"])
        if sha256(model_path) != selection["model_sha256"]:
            raise RuntimeError(f"model hash mismatch: {model_path}")
        model = joblib.load(model_path)
        grid = []
        for threshold in THRESHOLDS:
            report = score_predictions(
                records,
                _model_predictions(
                    records, model, str(selection["feature_set"]), threshold
                ),
                include_groups=False,
            )
            grid.append({"threshold": threshold, "metrics": report})
        best = max(
            grid,
            key=lambda item: _rank(
                dict(item["metrics"]), float(item["threshold"])
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
            f"median={dict(best['metrics'])['median_record_f1']:.6f}, "
            f"micro={dict(best['metrics'])['micro_f1']:.6f}",
            flush=True,
        )
        model_reports.append(
            {"selection": selection, "threshold_grid": grid, "selected": best}
        )
    eligible = [
        item for item in model_reports if bool(dict(item["selection"])["eligible_primary"])
    ]
    primary = max(
        eligible,
        key=lambda item: (
            *_rank(dict(dict(item["selected"])["metrics"])),
            int(dict(item["selection"])["feature_set"] == "geometry"),
            int(dict(item["selection"])["family"] == "logistic"),
        ),
    )
    frozen = {
        "protocol_sha256": PROTOCOL_SHA256,
        "record_manifest_sha256": MANIFEST_SHA256,
        "development_summary_sha256": sha256(
            RESULT_ROOT / "development" / "summary.json"
        ),
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
        "confirmation_split": "official MIMIC PERform Testing (100 adult, 100 neonate)",
    }
    frozen_path = RESULT_ROOT / "frozen_confirmation_rule.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return {
        "phase": "validation",
        "protocol_sha256": PROTOCOL_SHA256,
        "records": len(records),
        "baselines": baselines,
        "models": model_reports,
        "primary_model": frozen["primary_model"],
        "frozen_confirmation_rule": str(
            frozen_path.relative_to(PROJECT)
        ).replace("\\", "/"),
        "frozen_confirmation_rule_sha256": sha256(frozen_path),
    }


def _run_confirmation(records: list[Record]) -> dict[str, object]:
    frozen_path = RESULT_ROOT / "frozen_confirmation_rule.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("frozen rule uses a different protocol")
    if frozen["record_manifest_sha256"] != MANIFEST_SHA256:
        raise RuntimeError("frozen rule uses a different split manifest")
    if frozen["benchmark_script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("benchmark script changed after validation")
    development = _load_development()
    baselines = _fixed_baselines(records, development)
    reports = []
    for selection_object in frozen["models"]:
        selection = dict(selection_object)
        model_path = PROJECT / str(selection["model_file"])
        if sha256(model_path) != selection["model_sha256"]:
            raise RuntimeError(f"model hash mismatch: {model_path}")
        model = joblib.load(model_path)
        metrics = score_predictions(
            records,
            _model_predictions(
                records,
                model,
                str(selection["feature_set"]),
                float(selection["threshold"]),
            ),
        )
        reports.append(
            {
                "name": selection["name"],
                "feature_set": selection["feature_set"],
                "eligible_primary": selection["eligible_primary"],
                "threshold": selection["threshold"],
                "metrics": metrics,
            }
        )
        print(
            f"confirmation {selection['name']}: "
            f"median={metrics['median_record_f1']:.6f}, "
            f"micro={metrics['micro_f1']:.6f}",
            flush=True,
        )
    return {
        "phase": "confirmation",
        "protocol_sha256": PROTOCOL_SHA256,
        "records": len(records),
        "frozen_confirmation_rule_sha256": sha256(frozen_path),
        "baselines": baselines,
        "models": reports,
        "primary_model": frozen["primary_model"],
        "published_context": {
            "method": "MSPTDfast v2",
            "dataset": "MIMIC PERform Testing",
            "median_record_f1": 0.968,
            "status": "contextual; different locally generated ECG reference",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("development", "validation", "confirmation"), required=True
    )
    args = parser.parse_args()
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("protocol changed after it was frozen")
    if sha256(RECORD_MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("record split manifest changed after it was frozen")
    frozen_path = RESULT_ROOT / "frozen_confirmation_rule.json"
    if args.phase == "confirmation" and not frozen_path.exists():
        raise RuntimeError("official Testing data remain locked until validation")
    records, excluded = load_records(args.phase)
    if args.phase == "development":
        summary = _run_development(records)
    elif args.phase == "validation":
        summary = _run_validation(records)
    else:
        summary = _run_confirmation(records)
    summary["minimum_reference_beats"] = MINIMUM_REFERENCE_BEATS
    summary["excluded_records"] = excluded
    destination = RESULT_ROOT / args.phase
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    concise = {
        "phase": args.phase,
        "evaluable_records": len(records),
        "excluded_records": len(excluded),
        "candidates": int(sum(record.positions.size for record in records)),
        "cached_alignment_reference_beats": int(
            sum(record.aligned_reference_p0_count for record in records)
        ),
        "summary": str(summary_path.relative_to(PROJECT)).replace("\\", "/"),
        "sha256": sha256(summary_path),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
