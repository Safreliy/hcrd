"""Discovery candidate U1-D2: preserve HCRD level identity in the learner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))

from hcrd.structure_features import (  # noqa: E402
    hcrd_representation_batch,
    raw_channel_batch,
    wavelet_channel_batch,
)
from run_ucr_u1_benchmark import ALPHAS, _fit_head  # noqa: E402


METHODS = (
    "raw_minirocket",
    "wavelet_componentwise",
    "hcrd_componentwise",
    "hcrd_componentwise_hybrid",
    "raw_hcrd_fusion",
    "hcrd_structure_extratrees",
    "hcrd_structure_lightgbm",
)


def _scores(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
    }


def _componentwise_scale(collection: np.ndarray) -> np.ndarray:
    centered = collection - np.median(collection, axis=2, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=2, keepdims=True))
    return np.divide(
        centered,
        rms,
        out=centered.copy(),
        where=rms > 1e-3,
    )


def _componentwise_transform(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    total_kernels: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    channel_count = train_x.shape[1]
    kernels_per_channel = int(
        84 * np.ceil(total_kernels / (84 * channel_count))
    )
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    started = time.perf_counter()
    for channel in range(channel_count):
        transform = MiniRocket(
            n_kernels=kernels_per_channel,
            random_state=channel,
            n_jobs=workers,
        )
        train_channel = train_x[:, channel : channel + 1, :]
        test_channel = test_x[:, channel : channel + 1, :]
        train_blocks.append(transform.fit_transform(train_channel, train_y))
        test_blocks.append(transform.transform(test_channel))
    return (
        np.concatenate(train_blocks, axis=1),
        np.concatenate(test_blocks, axis=1),
        time.perf_counter() - started,
        kernels_per_channel,
    )


def _minirocket_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    kernels: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    transform = MiniRocket(n_kernels=kernels, random_state=0, n_jobs=workers)
    started = time.perf_counter()
    train_features = transform.fit_transform(train_x, train_y)
    test_features = transform.transform(test_x)
    return train_features, test_features, time.perf_counter() - started


def _ridge_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    transform_seconds: float,
) -> tuple[dict[str, float], np.ndarray]:
    prediction, fit_seconds, predict_seconds = _fit_head(
        train_x, train_y, test_x
    )
    return (
        {
            **_scores(test_y, prediction),
            "feature_count": int(train_x.shape[1]),
            "transform_seconds": transform_seconds,
            "head_fit_seconds": fit_seconds,
            "head_predict_seconds": predict_seconds,
            "ridge_alphas": ALPHAS.tolist(),
        },
        prediction,
    )


def run_dataset(
    record: dict[str, object],
    *,
    data: Path,
    output: Path,
    workers: int,
    kernels: int,
    max_levels: int,
    spatial_bins: int,
    top_k: int,
    manifest_sha256: str,
) -> dict[str, object]:
    name = str(record["name"])
    started = time.perf_counter()
    train_collection, train_labels = load_classification(
        name=name, split="train", extract_path=data
    )
    test_collection, test_labels = load_classification(
        name=name, split="test", extract_path=data
    )
    encoder = LabelEncoder().fit(train_labels)
    train_y = encoder.transform(train_labels)
    test_y = encoder.transform(test_labels)
    train_signals = np.asarray(train_collection[:, 0, :], dtype=float)
    test_signals = np.asarray(test_collection[:, 0, :], dtype=float)
    all_signals = np.concatenate([train_signals, test_signals])
    train_size = train_signals.shape[0]

    extraction_started = time.perf_counter()
    raw = raw_channel_batch(all_signals)
    wavelet = _componentwise_scale(
        wavelet_channel_batch(all_signals, max_levels=max_levels)
    )
    hcrd, structure, _ = hcrd_representation_batch(
        all_signals,
        max_levels=max_levels,
        spatial_bins=spatial_bins,
        top_k=top_k,
        workers=workers,
    )
    hcrd = _componentwise_scale(hcrd)
    extraction_seconds = time.perf_counter() - extraction_started

    metrics: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    raw_train, raw_test, raw_seconds = _minirocket_features(
        raw[:train_size], train_y, raw[train_size:], kernels=kernels, workers=workers
    )
    metrics["raw_minirocket"], predictions["raw_minirocket"] = _ridge_metrics(
        raw_train, train_y, raw_test, test_y, transform_seconds=raw_seconds
    )
    del raw

    wavelet_train, wavelet_test, wavelet_seconds, kernels_per_channel = (
        _componentwise_transform(
            wavelet[:train_size],
            train_y,
            wavelet[train_size:],
            total_kernels=kernels,
            workers=workers,
        )
    )
    metrics["wavelet_componentwise"], predictions["wavelet_componentwise"] = (
        _ridge_metrics(
            wavelet_train,
            train_y,
            wavelet_test,
            test_y,
            transform_seconds=wavelet_seconds,
        )
    )
    del wavelet, wavelet_train, wavelet_test

    hcrd_train, hcrd_test, hcrd_seconds, _ = _componentwise_transform(
        hcrd[:train_size],
        train_y,
        hcrd[train_size:],
        total_kernels=kernels,
        workers=workers,
    )
    metrics["hcrd_componentwise"], predictions["hcrd_componentwise"] = (
        _ridge_metrics(
            hcrd_train,
            train_y,
            hcrd_test,
            test_y,
            transform_seconds=hcrd_seconds,
        )
    )
    del hcrd

    train_structure = structure[:train_size]
    test_structure = structure[train_size:]
    hcrd_hybrid_train = np.concatenate([hcrd_train, train_structure], axis=1)
    hcrd_hybrid_test = np.concatenate([hcrd_test, test_structure], axis=1)
    metrics["hcrd_componentwise_hybrid"], predictions["hcrd_componentwise_hybrid"] = (
        _ridge_metrics(
            hcrd_hybrid_train,
            train_y,
            hcrd_hybrid_test,
            test_y,
            transform_seconds=hcrd_seconds,
        )
    )
    raw_hcrd_train = np.concatenate(
        [raw_train, hcrd_train, train_structure], axis=1
    )
    raw_hcrd_test = np.concatenate([raw_test, hcrd_test, test_structure], axis=1)
    metrics["raw_hcrd_fusion"], predictions["raw_hcrd_fusion"] = _ridge_metrics(
        raw_hcrd_train,
        train_y,
        raw_hcrd_test,
        test_y,
        transform_seconds=raw_seconds + hcrd_seconds,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
        n_jobs=workers,
    )
    model_started = time.perf_counter()
    extra_trees.fit(train_structure, train_y)
    extra_prediction = extra_trees.predict(test_structure)
    metrics["hcrd_structure_extratrees"] = {
        **_scores(test_y, extra_prediction),
        "feature_count": int(train_structure.shape[1]),
        "transform_seconds": 0.0,
        "head_fit_seconds": time.perf_counter() - model_started,
    }
    predictions["hcrd_structure_extratrees"] = extra_prediction

    minimum_leaf = max(5, min(20, train_size // 20))
    lightgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=minimum_leaf,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=workers,
        verbosity=-1,
    )
    model_started = time.perf_counter()
    lightgbm.fit(train_structure, train_y)
    lightgbm_prediction = lightgbm.predict(test_structure)
    metrics["hcrd_structure_lightgbm"] = {
        **_scores(test_y, lightgbm_prediction),
        "feature_count": int(train_structure.shape[1]),
        "transform_seconds": 0.0,
        "head_fit_seconds": time.perf_counter() - model_started,
        "min_child_samples": minimum_leaf,
    }
    predictions["hcrd_structure_lightgbm"] = lightgbm_prediction

    output = output.resolve()
    prediction_path = output / "predictions" / f"{name}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, target=test_y, **predictions)
    result = {
        "protocol": "U1-D2 exploratory",
        "dataset": name,
        "assignment": record["assignment"],
        "manifest_sha256": manifest_sha256,
        "train_size": int(train_size),
        "test_size": int(test_y.size),
        "length": int(train_signals.shape[1]),
        "class_count": int(np.unique(train_y).size),
        "workers": workers,
        "requested_total_kernels": kernels,
        "kernels_per_component": kernels_per_channel,
        "representation_seconds": extraction_seconds,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    result_path = output / "per_dataset" / f"{name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(output: Path, results: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        for method in METHODS:
            rows.append(
                {
                    "dataset": result["dataset"],
                    "method": method,
                    **result["metrics"][method],
                }
            )
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        values = np.asarray([float(row["accuracy"]) for row in selected])
        summaries.append(
            {
                "method": method,
                "dataset_count": len(values),
                "mean_accuracy": float(np.mean(values)),
                "median_accuracy": float(np.median(values)),
            }
        )
    (output / "summary.json").write_text(
        json.dumps({"datasets": len(results), "summary": summaries}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("discovery", "confirmation"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "data" / "manifests" / "ucr_u1_manifest.json",
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT / "data" / "raw" / "ucr_2018"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "ucr_u1_d2_componentwise"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kernels", type=int, default=10_000)
    parser.add_argument("--max-levels", type=int, default=5)
    parser.add_argument("--spatial-bins", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "confirmation" and not (
        PROJECT / "results" / "ucr_u1" / "frozen_subgroup_rule.json"
    ).exists():
        raise RuntimeError("confirmation remains locked until a candidate rule is frozen")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    records = [
        item
        for item in manifest["records"]
        if item["eligible"] and item["assignment"] == args.stage
    ]
    if args.datasets:
        selected = set(args.datasets)
        records = [item for item in records if item["name"] in selected]
        missing = selected - {str(item["name"]) for item in records}
        if missing:
            raise ValueError(f"not eligible in {args.stage}: {sorted(missing)}")
    output = args.output / args.stage
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in enumerate(records, start=1):
        result_path = output / "per_dataset" / f"{record['name']}.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(json.dumps({"dataset": record["name"], "status": "cached"}), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "dataset": record["name"],
                        "index": index,
                        "total": len(records),
                        "status": "running",
                    }
                ),
                flush=True,
            )
            result = run_dataset(
                record,
                data=args.data,
                output=output,
                workers=args.workers,
                kernels=args.kernels,
                max_levels=args.max_levels,
                spatial_bins=args.spatial_bins,
                top_k=args.top_k,
                manifest_sha256=manifest_sha256,
            )
            print(
                json.dumps(
                    {
                        "dataset": record["name"],
                        "status": "complete",
                        "wall_seconds": result["wall_seconds"],
                        "accuracies": {
                            method: result["metrics"][method]["accuracy"]
                            for method in METHODS
                        },
                    }
                ),
                flush=True,
            )
        results.append(result)
        write_summary(output, results)


if __name__ == "__main__":
    main()

