"""Run the frozen discovery or confirmation stage of UCR protocol U1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import aeon
import numpy as np
import sklearn
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.structure_features import (  # noqa: E402
    geometry_feature_names,
    hcrd_representation_batch,
    raw_channel_batch,
    structure_feature_names,
    wavelet_channel_batch,
)


METHODS = (
    "raw_minirocket",
    "wavelet_minirocket",
    "hcrd_minirocket",
    "hcrd_energy",
    "hcrd_structure",
    "hcrd_hybrid",
)
ALPHAS = np.logspace(-3, 3, 10)


def _fit_head(
    train_features: np.ndarray,
    train_y: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=ALPHAS),
    )
    started = time.perf_counter()
    model.fit(train_features, train_y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(test_features)
    predict_seconds = time.perf_counter() - started
    return np.asarray(prediction), fit_seconds, predict_seconds


def _scores(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
    }


def _evaluate_minirocket(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    kernels: int,
    workers: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    transform = MiniRocket(
        n_kernels=kernels,
        random_state=0,
        n_jobs=workers,
    )
    started = time.perf_counter()
    train_features = transform.fit_transform(train_x, train_y)
    test_features = transform.transform(test_x)
    transform_seconds = time.perf_counter() - started
    prediction, fit_seconds, predict_seconds = _fit_head(
        train_features, train_y, test_features
    )
    metrics = {
        **_scores(test_y, prediction),
        "transform_seconds": transform_seconds,
        "head_fit_seconds": fit_seconds,
        "head_predict_seconds": predict_seconds,
        "feature_count": int(train_features.shape[1]),
    }
    return metrics, prediction, train_features, test_features


def _evaluate_tabular(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    prediction, fit_seconds, predict_seconds = _fit_head(train_x, train_y, test_x)
    return (
        {
            **_scores(test_y, prediction),
            "transform_seconds": 0.0,
            "head_fit_seconds": fit_seconds,
            "head_predict_seconds": predict_seconds,
            "feature_count": int(train_x.shape[1]),
        },
        prediction,
    )


def _load_manifest(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def run_dataset(
    record: dict[str, object],
    *,
    data_path: Path,
    output_path: Path,
    workers: int,
    kernels: int,
    max_levels: int,
    spatial_bins: int,
    top_k: int,
    manifest_sha256: str,
) -> dict[str, object]:
    output_path = output_path.resolve()
    name = str(record["name"])
    dataset_started = time.perf_counter()
    train_collection, train_labels = load_classification(
        name=name, split="train", extract_path=data_path
    )
    test_collection, test_labels = load_classification(
        name=name, split="test", extract_path=data_path
    )
    encoder = LabelEncoder().fit(train_labels)
    train_y = encoder.transform(train_labels)
    test_y = encoder.transform(test_labels)
    train_signals = np.asarray(train_collection[:, 0, :], dtype=float)
    test_signals = np.asarray(test_collection[:, 0, :], dtype=float)
    all_signals = np.concatenate([train_signals, test_signals], axis=0)
    train_size = train_signals.shape[0]

    extraction_started = time.perf_counter()
    raw_channels = raw_channel_batch(all_signals)
    raw_extraction_seconds = time.perf_counter() - extraction_started

    extraction_started = time.perf_counter()
    wavelet_channels = wavelet_channel_batch(all_signals, max_levels=max_levels)
    wavelet_extraction_seconds = time.perf_counter() - extraction_started

    extraction_started = time.perf_counter()
    hcrd_channels, structure_features, geometry_features = hcrd_representation_batch(
        all_signals,
        max_levels=max_levels,
        spatial_bins=spatial_bins,
        top_k=top_k,
        workers=workers,
    )
    hcrd_extraction_seconds = time.perf_counter() - extraction_started

    train_structure = structure_features[:train_size]
    test_structure = structure_features[train_size:]
    feature_names = structure_feature_names(
        max_levels=max_levels, spatial_bins=spatial_bins, top_k=top_k
    )
    energy_indices = np.asarray(
        [index for index, feature_name in enumerate(feature_names) if "_energy_" in feature_name]
    )

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}

    raw_metrics, raw_prediction, _, _ = _evaluate_minirocket(
        raw_channels[:train_size],
        train_y,
        raw_channels[train_size:],
        test_y,
        kernels=kernels,
        workers=workers,
    )
    raw_metrics["representation_seconds"] = raw_extraction_seconds
    metrics["raw_minirocket"] = raw_metrics
    predictions["raw_minirocket"] = raw_prediction
    del raw_channels

    wavelet_metrics, wavelet_prediction, _, _ = _evaluate_minirocket(
        wavelet_channels[:train_size],
        train_y,
        wavelet_channels[train_size:],
        test_y,
        kernels=kernels,
        workers=workers,
    )
    wavelet_metrics["representation_seconds"] = wavelet_extraction_seconds
    metrics["wavelet_minirocket"] = wavelet_metrics
    predictions["wavelet_minirocket"] = wavelet_prediction
    del wavelet_channels

    hcrd_metrics, hcrd_prediction, hcrd_train_transform, hcrd_test_transform = (
        _evaluate_minirocket(
            hcrd_channels[:train_size],
            train_y,
            hcrd_channels[train_size:],
            test_y,
            kernels=kernels,
            workers=workers,
        )
    )
    hcrd_metrics["representation_seconds"] = hcrd_extraction_seconds
    metrics["hcrd_minirocket"] = hcrd_metrics
    predictions["hcrd_minirocket"] = hcrd_prediction
    del hcrd_channels

    energy_metrics, energy_prediction = _evaluate_tabular(
        train_structure[:, energy_indices],
        train_y,
        test_structure[:, energy_indices],
        test_y,
    )
    energy_metrics["representation_seconds"] = hcrd_extraction_seconds
    metrics["hcrd_energy"] = energy_metrics
    predictions["hcrd_energy"] = energy_prediction

    structure_metrics, structure_prediction = _evaluate_tabular(
        train_structure,
        train_y,
        test_structure,
        test_y,
    )
    structure_metrics["representation_seconds"] = hcrd_extraction_seconds
    metrics["hcrd_structure"] = structure_metrics
    predictions["hcrd_structure"] = structure_prediction

    hybrid_train = np.concatenate([hcrd_train_transform, train_structure], axis=1)
    hybrid_test = np.concatenate([hcrd_test_transform, test_structure], axis=1)
    hybrid_metrics, hybrid_prediction = _evaluate_tabular(
        hybrid_train, train_y, hybrid_test, test_y
    )
    hybrid_metrics["representation_seconds"] = hcrd_extraction_seconds
    hybrid_metrics["transform_seconds"] = hcrd_metrics["transform_seconds"]
    metrics["hcrd_hybrid"] = hybrid_metrics
    predictions["hcrd_hybrid"] = hybrid_prediction

    geometry_names = geometry_feature_names(max_levels=max_levels)
    train_geometry = geometry_features[:train_size]
    geometry_summary = {
        feature_name: {
            "mean": float(np.mean(train_geometry[:, index])),
            "median": float(np.median(train_geometry[:, index])),
            "standard_deviation": float(np.std(train_geometry[:, index])),
        }
        for index, feature_name in enumerate(geometry_names)
    }

    output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / "predictions" / f"{name}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        target=test_y,
        **predictions,
    )
    result = {
        "protocol": "U1",
        "manifest_sha256": manifest_sha256,
        "dataset": name,
        "assignment": record["assignment"],
        "train_size": int(train_size),
        "test_size": int(test_signals.shape[0]),
        "length": int(train_signals.shape[1]),
        "class_count": int(np.unique(train_y).size),
        "max_levels": max_levels,
        "spatial_bins": spatial_bins,
        "top_k": top_k,
        "requested_kernels": kernels,
        "workers": workers,
        "metrics": metrics,
        "training_geometry": geometry_summary,
        "prediction_file": str(prediction_path.relative_to(PROJECT)),
        "wall_seconds": time.perf_counter() - dataset_started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "aeon": aeon.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    result_path = output_path / "per_dataset" / f"{name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_collection_summary(output: Path, results: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        for method in METHODS:
            rows.append(
                {
                    "dataset": result["dataset"],
                    "assignment": result["assignment"],
                    "train_size": result["train_size"],
                    "test_size": result["test_size"],
                    "length": result["length"],
                    "class_count": result["class_count"],
                    "method": method,
                    **result["metrics"][method],
                }
            )
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        accuracies = np.asarray([float(row["accuracy"]) for row in selected])
        summary.append(
            {
                "method": method,
                "dataset_count": len(selected),
                "mean_accuracy": float(np.mean(accuracies)),
                "median_accuracy": float(np.median(accuracies)),
                "mean_balanced_accuracy": float(
                    np.mean([float(row["balanced_accuracy"]) for row in selected])
                ),
                "mean_macro_f1": float(
                    np.mean([float(row["macro_f1"]) for row in selected])
                ),
            }
        )
    (output / "summary.json").write_text(
        json.dumps({"datasets": len(results), "summary": summary}, indent=2),
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
        "--output", type=Path, default=PROJECT / "results" / "ucr_u1"
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
        args.output / "frozen_subgroup_rule.json"
    ).exists():
        raise RuntimeError(
            "confirmation is locked until discovery writes frozen_subgroup_rule.json"
        )

    manifest, manifest_digest = _load_manifest(args.manifest)
    records = [
        record
        for record in manifest["records"]
        if record["eligible"] and record["assignment"] == args.stage
    ]
    if args.datasets:
        requested = set(args.datasets)
        records = [record for record in records if record["name"] in requested]
        missing = requested - {str(record["name"]) for record in records}
        if missing:
            raise ValueError(f"not eligible in {args.stage}: {sorted(missing)}")

    stage_output = args.output / args.stage
    stage_output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        name = str(record["name"])
        result_path = stage_output / "per_dataset" / f"{name}.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(json.dumps({"dataset": name, "status": "cached"}), flush=True)
        else:
            print(
                json.dumps(
                    {"dataset": name, "index": index, "total": len(records), "status": "running"}
                ),
                flush=True,
            )
            result = run_dataset(
                record,
                data_path=args.data,
                output_path=stage_output,
                workers=args.workers,
                kernels=args.kernels,
                max_levels=args.max_levels,
                spatial_bins=args.spatial_bins,
                top_k=args.top_k,
                manifest_sha256=manifest_digest,
            )
            print(
                json.dumps(
                    {
                        "dataset": name,
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
        write_collection_summary(stage_output, results)


if __name__ == "__main__":
    main()
