#!/usr/bin/env python3
"""Staged E1 benchmark for expert-labelled LC--MS EIC peak shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from hcrd.lcms import eic_pair_feature_bank, eic_partition
from hcrd.lcms_data import EICFlatCache


SEED = 20260825
REPRESENTATIONS = (
    "raw64",
    "domain",
    "hcrd_1",
    "hcrd_8",
    "hcrd_geometry",
    "area_only",
)
FEATURE_WIDTHS = {
    "raw64": 150,
    "domain": 222,
    "hcrd_1": 354,
    "hcrd_8": 1896,
    "hcrd_geometry": 594,
    "area_only": 96,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_rows(
    labels_path: Path,
    cache: EICFlatCache,
    partition: str,
) -> tuple[NDArray[np.int16], NDArray[np.int32], NDArray[np.uint8], dict[str, int]]:
    header = pd.read_csv(labels_path, nrows=0).columns.tolist()
    if header[0] != "row ID":
        raise ValueError("unexpected primary classification schema")
    sample_columns = [
        sample for sample in header[1:] if eic_partition("sample", sample) == partition
    ]
    # This is the leakage guard: development never reads confirmation sample
    # columns, and confirmation is inaccessible before selection is frozen.
    frame = pd.read_csv(labels_path, usecols=["row ID", *sample_columns])
    if frame.shape[0] != cache.peak_count:
        raise ValueError("classification/RData peak counts differ")
    sample_lookup = {name: index for index, name in enumerate(cache.sample_names)}
    if set(sample_columns) - set(sample_lookup):
        raise ValueError("classification sample absent from EIC cache")
    allowed_peaks = np.asarray(
        [
            index
            for index, peak_id in enumerate(frame["row ID"].astype(str))
            if eic_partition("peak", peak_id) == partition
        ],
        dtype=np.int32,
    )
    sample_rows: list[int] = []
    peak_rows: list[int] = []
    y_rows: list[int] = []
    counts = {"definite_peak": 0, "definite_nonpeak": 0, "inconclusive": 0}
    for sample in sample_columns:
        values = pd.to_numeric(frame[sample], errors="coerce").to_numpy()[allowed_peaks]
        for peak_index, value in zip(allowed_peaks, values, strict=True):
            if np.isnan(value):
                label = 1
                counts["definite_peak"] += 1
            elif value == 1.0:
                label = 0
                counts["definite_nonpeak"] += 1
            elif value == 0.0:
                counts["inconclusive"] += 1
                continue
            else:
                raise ValueError(f"unexpected expert code {value!r}")
            sample_rows.append(sample_lookup[sample])
            peak_rows.append(int(peak_index))
            y_rows.append(label)
    return (
        np.asarray(sample_rows, dtype=np.int16),
        np.asarray(peak_rows, dtype=np.int32),
        np.asarray(y_rows, dtype=np.uint8),
        counts,
    )


def _extract_chunk(
    task: tuple[str, list[tuple[int, int]]]
) -> dict[str, NDArray[np.float32]]:
    cache_directory, rows = task
    cache = EICFlatCache(cache_directory)
    output = {
        name: np.empty((len(rows), width), dtype=np.float32)
        for name, width in FEATURE_WIDTHS.items()
    }
    for row, (sample_index, peak_index) in enumerate(rows):
        pair = cache.pair(sample_index, peak_index)
        bank = eic_pair_feature_bank(
            pair.short_intensity,
            pair.short_retention_time,
            pair.long_intensity,
            pair.long_retention_time,
        )
        for name in REPRESENTATIONS:
            output[name][row] = getattr(bank, name)
    return output


def _chunks(rows: list[tuple[int, int]], size: int) -> Iterable[list[tuple[int, int]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def build_features(
    *,
    source_dir: Path,
    flat_cache: Path,
    output_dir: Path,
    partition: str,
    workers: int,
    chunk_size: int,
    selection_path: Path | None = None,
) -> None:
    if partition not in {"train", "validation", "confirmation"}:
        raise ValueError("unsupported partition")
    if partition == "confirmation":
        if selection_path is None or not selection_path.exists():
            raise RuntimeError("confirmation is locked until selection_frozen.json exists")
    cache = EICFlatCache(flat_cache)
    labels_path = source_dir / "Classification_before_cleanup.csv"
    sample_indices, peak_indices, labels, label_counts = _label_rows(
        labels_path, cache, partition
    )
    if np.unique(labels).size != 2:
        raise RuntimeError(f"partition {partition} lacks one binary class")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "sample_index.npy", sample_indices)
    np.save(output_dir / "peak_index.npy", peak_indices)
    np.save(output_dir / "labels.npy", labels)
    arrays = {
        name: np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(labels.size, width),
        )
        for name, width in FEATURE_WIDTHS.items()
    }
    rows = list(zip(sample_indices.tolist(), peak_indices.tolist(), strict=True))
    tasks = ((str(flat_cache), chunk) for chunk in _chunks(rows, chunk_size))
    position = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_index, result in enumerate(executor.map(_extract_chunk, tasks), start=1):
            count = next(iter(result.values())).shape[0]
            for name, values in result.items():
                arrays[name][position : position + count] = values
            position += count
            if chunk_index % 10 == 0 or position == labels.size:
                print(f"{partition}: {position}/{labels.size} EIC pairs", flush=True)
    if position != labels.size:
        raise RuntimeError("feature extraction ended early")
    for array in arrays.values():
        array.flush()
    metadata = {
        "protocol": "hcrd-e1-v1",
        "partition": partition,
        "case_count": int(labels.size),
        "positive_count": int(np.sum(labels)),
        "negative_count": int(labels.size - np.sum(labels)),
        "expert_code_counts_before_exclusion": label_counts,
        "sample_group_count": int(np.unique(sample_indices).size),
        "peak_group_count": int(np.unique(peak_indices).size),
        "feature_widths": FEATURE_WIDTHS,
        "source_sha256": {
            "EIC_data.RData": _sha256(source_dir / "EIC_data.RData"),
            "Classification_before_cleanup.csv": _sha256(labels_path),
        },
        "selection_sha256": (
            _sha256(selection_path) if partition == "confirmation" else None
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _load_partition(directory: Path, name: str) -> NDArray[np.float32]:
    return np.load(directory / f"{name}.npy", mmap_mode="r")


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED,
        early_stopping=False,
    )


def _best_mcc_threshold(labels: NDArray[np.uint8], scores: NDArray[np.float64]) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_y = labels[order].astype(np.int64)
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    positives = int(tp[-1])
    negatives = labels.size - positives
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), labels.size - 1]
    tp_e, fp_e = tp[ends], fp[ends]
    fn_e, tn_e = positives - tp_e, negatives - fp_e
    denominator = np.sqrt(
        (tp_e + fp_e) * (tp_e + fn_e) * (tn_e + fp_e) * (tn_e + fn_e)
    )
    mcc = np.divide(
        tp_e * tn_e - fp_e * fn_e,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    best = float(np.max(mcc))
    candidates = sorted_scores[ends[np.isclose(mcc, best, rtol=0.0, atol=1e-15)]]
    return float(np.max(candidates))


def _metrics(
    labels: NDArray[np.uint8], scores: NDArray[np.float64], threshold: float
) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "threshold": float(threshold),
    }


def fit_development(train_dir: Path, validation_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_y = np.load(train_dir / "labels.npy")
    validation_y = np.load(validation_dir / "labels.npy")
    metrics: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}
    for name in REPRESENTATIONS:
        print(f"fit {name}", flush=True)
        model = _model()
        model.fit(_load_partition(train_dir, name), train_y)
        scores = model.predict_proba(_load_partition(validation_dir, name))[:, 1]
        threshold = _best_mcc_threshold(validation_y, scores)
        thresholds[name] = threshold
        metrics[name] = _metrics(validation_y, scores, threshold)
        joblib.dump(model, output_dir / f"model_{name}.joblib", compress=3)
        np.save(output_dir / f"validation_scores_{name}.npy", scores)
        print(json.dumps({name: metrics[name]}), flush=True)
    comparator = sorted(
        ("raw64", "domain"),
        key=lambda name: (
            -metrics[name]["average_precision"],
            -metrics[name]["roc_auc"],
            name,
        ),
    )[0]
    selection = {
        "protocol": "hcrd-e1-v1",
        "frozen_before_confirmation": True,
        "seed": SEED,
        "comparator": comparator,
        "thresholds": thresholds,
        "validation_metrics": metrics,
        "feature_widths": FEATURE_WIDTHS,
        "learner": {
            "class": "HistGradientBoostingClassifier",
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
            "random_state": SEED,
            "early_stopping": False,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "train_metadata_sha256": _sha256(train_dir / "metadata.json"),
        "validation_metadata_sha256": _sha256(validation_dir / "metadata.json"),
    }
    (output_dir / "selection_frozen.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )


def _weighted_ap_preparation(scores: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]) + 1]
    return order, starts


def _weighted_ap(
    labels: NDArray[np.uint8],
    weights: NDArray[np.float64],
    order: NDArray[np.int64],
    starts: NDArray[np.int64],
) -> float:
    sorted_weight = weights[order]
    sorted_positive = sorted_weight * labels[order]
    group_weight = np.add.reduceat(sorted_weight, starts)
    group_positive = np.add.reduceat(sorted_positive, starts)
    total_positive = float(np.sum(group_positive))
    if total_positive <= 0.0:
        return np.nan
    cumulative_weight = np.cumsum(group_weight)
    cumulative_positive = np.cumsum(group_positive)
    precision = np.divide(
        cumulative_positive,
        cumulative_weight,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_weight > 0.0,
    )
    return float(np.sum(group_positive * precision) / total_positive)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def evaluate_confirmation(
    confirmation_dir: Path, model_dir: Path, output_dir: Path, bootstrap: int
) -> None:
    selection_path = model_dir / "selection_frozen.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("frozen_before_confirmation"):
        raise RuntimeError("selection file is not frozen")
    confirmation_metadata = json.loads(
        (confirmation_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if confirmation_metadata.get("selection_sha256") != _sha256(selection_path):
        raise RuntimeError("confirmation features were not unlocked by this selection")
    labels = np.load(confirmation_dir / "labels.npy")
    sample_index = np.load(confirmation_dir / "sample_index.npy")
    peak_index = np.load(confirmation_dir / "peak_index.npy")
    scores: dict[str, NDArray[np.float64]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name in REPRESENTATIONS:
        model = joblib.load(model_dir / f"model_{name}.joblib")
        scores[name] = model.predict_proba(_load_partition(confirmation_dir, name))[:, 1]
        metrics[name] = _metrics(labels, scores[name], selection["thresholds"][name])
    comparator = selection["comparator"]

    sample_values, sample_inverse = np.unique(sample_index, return_inverse=True)
    peak_values, peak_inverse = np.unique(peak_index, return_inverse=True)
    preparations = {
        name: _weighted_ap_preparation(values) for name, values in scores.items()
    }
    rng = np.random.default_rng(SEED)
    bootstrap_ap = {
        name: np.empty(bootstrap, dtype=np.float64) for name in REPRESENTATIONS
    }
    for replicate in range(bootstrap):
        sample_counts = rng.multinomial(
            sample_values.size, np.full(sample_values.size, 1.0 / sample_values.size)
        )
        peak_counts = rng.multinomial(
            peak_values.size, np.full(peak_values.size, 1.0 / peak_values.size)
        )
        weights = sample_counts[sample_inverse] * peak_counts[peak_inverse]
        weights = weights.astype(float, copy=False)
        for name in REPRESENTATIONS:
            order, starts = preparations[name]
            bootstrap_ap[name][replicate] = _weighted_ap(
                labels, weights, order, starts
            )
        if (replicate + 1) % 1000 == 0:
            print(f"bootstrap {replicate + 1}/{bootstrap}", flush=True)

    comparisons = {}
    raw_p_values = {}
    for name in REPRESENTATIONS:
        if name == comparator:
            continue
        difference = bootstrap_ap[name] - bootstrap_ap[comparator]
        finite = difference[np.isfinite(difference)]
        point = metrics[name]["average_precision"] - metrics[comparator][
            "average_precision"
        ]
        p_value = min(
            1.0,
            2.0
            * min(
                (np.sum(finite <= 0.0) + 1.0) / (finite.size + 1.0),
                (np.sum(finite >= 0.0) + 1.0) / (finite.size + 1.0),
            ),
        )
        raw_p_values[name] = float(p_value)
        comparisons[name] = {
            "versus": comparator,
            "ap_difference": float(point),
            "cluster_bootstrap_95_ci": np.quantile(finite, [0.025, 0.975]).tolist(),
            "two_sided_bootstrap_p": float(p_value),
        }
    adjusted = _holm(raw_p_values)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    main_ci = comparisons["hcrd_8"]["cluster_bootstrap_95_ci"]
    success = bool(
        comparisons["hcrd_8"]["ap_difference"] > 0.0
        and main_ci[0] > 0.0
        and metrics["hcrd_8"]["average_precision"]
        > metrics["hcrd_1"]["average_precision"]
    )
    result = {
        "protocol": "hcrd-e1-v1",
        "selection_sha256": _sha256(selection_path),
        "comparator": comparator,
        "confirmation_metadata": confirmation_metadata,
        "metrics": metrics,
        "ap_comparisons": comparisons,
        "prospective_success": success,
        "success_components": {
            "positive_hcrd8_comparator_ap": comparisons["hcrd_8"][
                "ap_difference"
            ]
            > 0.0,
            "positive_cluster_ci_lower": main_ci[0] > 0.0,
            "hcrd8_exceeds_hcrd1": metrics["hcrd_8"]["average_precision"]
            > metrics["hcrd_1"]["average_precision"],
        },
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": SEED,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confirmation_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    prediction_frame = pd.DataFrame(
        {
            "sample_index": sample_index,
            "peak_index": peak_index,
            "label": labels,
            **{f"score_{name}": value for name, value in scores.items()},
        }
    )
    prediction_frame.to_csv(output_dir / "confirmation_predictions.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-features")
    build.add_argument("--partition", choices=("train", "validation", "confirmation"), required=True)
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--flat-cache", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--selection", type=Path)
    build.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    build.add_argument("--chunk-size", type=int, default=64)

    fit = subparsers.add_parser("fit-development")
    fit.add_argument("--train-dir", type=Path, required=True)
    fit.add_argument("--validation-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-confirmation")
    evaluate.add_argument("--confirmation-dir", type=Path, required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=10000)

    args = parser.parse_args()
    if args.command == "build-features":
        if args.workers < 1 or args.chunk_size < 1:
            parser.error("workers and chunk-size must be positive")
        build_features(
            source_dir=args.source_dir,
            flat_cache=args.flat_cache,
            output_dir=args.output_dir,
            partition=args.partition,
            workers=args.workers,
            chunk_size=args.chunk_size,
            selection_path=args.selection,
        )
    elif args.command == "fit-development":
        fit_development(args.train_dir, args.validation_dir, args.output_dir)
    elif args.command == "evaluate-confirmation":
        evaluate_confirmation(
            args.confirmation_dir, args.model_dir, args.output_dir, args.bootstrap
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
