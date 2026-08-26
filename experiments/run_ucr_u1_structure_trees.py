"""U1-D3: nonlinear learning on the complete HCRD structure collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from aeon.datasets import load_classification
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
)

from hcrd.structure_features import hcrd_representation_batch  # noqa: E402


METHODS = ("hcrd_structure_extratrees_cv", "hcrd_structure_lightgbm")


def scores(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
    }


def run_dataset(
    record: dict[str, object],
    *,
    data: Path,
    output: Path,
    workers: int,
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
    train_size = train_signals.shape[0]
    all_signals = np.concatenate([train_signals, test_signals])

    extraction_started = time.perf_counter()
    _, structure, _ = hcrd_representation_batch(
        all_signals, max_levels=5, spatial_bins=8, top_k=8, workers=workers
    )
    extraction_seconds = time.perf_counter() - extraction_started
    train_x = structure[:train_size]
    test_x = structure[train_size:]

    class_counts = np.bincount(train_y)
    folds = int(min(5, np.min(class_counts)))
    base_forest = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )
    forest_started = time.perf_counter()
    if folds >= 2:
        search = GridSearchCV(
            base_forest,
            param_grid={
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5],
            },
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=0),
            n_jobs=workers,
            refit=True,
        )
        search.fit(train_x, train_y)
        forest = search.best_estimator_
        best_parameters = search.best_params_
        best_cv_accuracy = float(search.best_score_)
    else:
        forest = base_forest.set_params(min_samples_leaf=2, max_features="sqrt")
        forest.fit(train_x, train_y)
        best_parameters = {"min_samples_leaf": 2, "max_features": "sqrt"}
        best_cv_accuracy = None
    forest_prediction = forest.predict(test_x)
    forest_seconds = time.perf_counter() - forest_started

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
    lightgbm_started = time.perf_counter()
    lightgbm.fit(train_x, train_y)
    lightgbm_prediction = lightgbm.predict(test_x)
    lightgbm_seconds = time.perf_counter() - lightgbm_started

    metrics = {
        "hcrd_structure_extratrees_cv": {
            **scores(test_y, forest_prediction),
            "feature_count": int(train_x.shape[1]),
            "fit_seconds": forest_seconds,
            "cv_folds": folds,
            "best_cv_accuracy": best_cv_accuracy,
            "best_parameters": best_parameters,
        },
        "hcrd_structure_lightgbm": {
            **scores(test_y, lightgbm_prediction),
            "feature_count": int(train_x.shape[1]),
            "fit_seconds": lightgbm_seconds,
            "min_child_samples": minimum_leaf,
        },
    }
    output = output.resolve()
    prediction_path = output / "predictions" / f"{name}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        target=test_y,
        hcrd_structure_extratrees_cv=forest_prediction,
        hcrd_structure_lightgbm=lightgbm_prediction,
    )
    result = {
        "protocol": "U1-D3 exploratory",
        "dataset": name,
        "assignment": record["assignment"],
        "manifest_sha256": manifest_sha256,
        "train_size": int(train_size),
        "test_size": int(test_y.size),
        "length": int(train_signals.shape[1]),
        "class_count": int(np.unique(train_y).size),
        "workers": workers,
        "representation_seconds": extraction_seconds,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    result_path = output / "per_dataset" / f"{name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(output: Path, results: list[dict[str, object]]) -> None:
    rows = [
        {
            "dataset": result["dataset"],
            "method": method,
            **result["metrics"][method],
        }
        for result in results
        for method in METHODS
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in METHODS:
        values = np.asarray(
            [float(row["accuracy"]) for row in rows if row["method"] == method]
        )
        summary.append(
            {
                "method": method,
                "dataset_count": int(values.size),
                "mean_accuracy": float(np.mean(values)),
                "median_accuracy": float(np.median(values)),
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
        "--output", type=Path, default=PROJECT / "results" / "ucr_u1_d3_structure"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "confirmation" and not (
        PROJECT / "results" / "ucr_u1" / "frozen_subgroup_rule.json"
    ).exists():
        raise RuntimeError("confirmation remains locked until a candidate rule is frozen")

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
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
                manifest_sha256=digest,
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
