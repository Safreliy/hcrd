"""Compute training-only CV evidence for the U1 representation gate."""

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
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.structure_features import raw_channel_batch, wavelet_channel_batch  # noqa: E402


ALPHAS = np.logspace(-3, 3, 10)


def cv_accuracy(features: np.ndarray, labels: np.ndarray, *, workers: int) -> tuple[float | None, int]:
    counts = np.bincount(labels)
    folds = int(min(5, np.min(counts)))
    if folds < 2:
        return None, folds
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    model = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=ALPHAS),
    )
    values = cross_val_score(
        model, features, labels, cv=splitter, scoring="accuracy", n_jobs=workers
    )
    return float(np.mean(values)), folds


def run_dataset(
    record: dict[str, object],
    *,
    data: Path,
    workers: int,
    kernels: int,
) -> dict[str, object]:
    name = str(record["name"])
    started = time.perf_counter()
    collection, labels = load_classification(
        name=name, split="train", extract_path=data
    )
    encoded = LabelEncoder().fit_transform(labels)
    signals = np.asarray(collection[:, 0, :], dtype=float)
    raw = raw_channel_batch(signals)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Level value of .* is too high.*")
        wavelet = wavelet_channel_batch(signals, max_levels=5)

    transform = MiniRocket(n_kernels=kernels, random_state=0, n_jobs=workers)
    raw_started = time.perf_counter()
    raw_features = transform.fit_transform(raw, encoded)
    raw_transform_seconds = time.perf_counter() - raw_started
    raw_cv, folds = cv_accuracy(raw_features, encoded, workers=workers)
    del raw, raw_features, transform

    transform = MiniRocket(n_kernels=kernels, random_state=0, n_jobs=workers)
    wavelet_started = time.perf_counter()
    wavelet_features = transform.fit_transform(wavelet, encoded)
    wavelet_transform_seconds = time.perf_counter() - wavelet_started
    wavelet_cv, wavelet_folds = cv_accuracy(
        wavelet_features, encoded, workers=workers
    )
    if wavelet_folds != folds:
        raise RuntimeError("raw and wavelet CV fold counts differ")
    return {
        "dataset": name,
        "assignment": record["assignment"],
        "train_size": int(signals.shape[0]),
        "length": int(signals.shape[1]),
        "class_count": int(np.unique(encoded).size),
        "cv_folds": folds,
        "raw_cv_accuracy": raw_cv,
        "wavelet_cv_accuracy": wavelet_cv,
        "raw_transform_seconds": raw_transform_seconds,
        "wavelet_transform_seconds": wavelet_transform_seconds,
        "wall_seconds": time.perf_counter() - started,
    }


def write_scores(output: Path, results: list[dict[str, object]]) -> None:
    with (output / "training_cv.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


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
        "--output", type=Path, default=PROJECT / "results" / "ucr_u1_gate"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kernels", type=int, default=10_000)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "confirmation" and not (
        PROJECT / "results" / "ucr_u1" / "frozen_subgroup_rule.json"
    ).exists():
        raise RuntimeError("confirmation remains locked until the gate is frozen")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
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
                record, data=args.data, workers=args.workers, kernels=args.kernels
            )
            result["manifest_sha256"] = manifest_sha256
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "dataset": record["name"],
                        "status": "complete",
                        "raw_cv": result["raw_cv_accuracy"],
                        "wavelet_cv": result["wavelet_cv_accuracy"],
                        "wall_seconds": result["wall_seconds"],
                    }
                ),
                flush=True,
            )
        results.append(result)
        write_scores(output, results)


if __name__ == "__main__":
    main()

