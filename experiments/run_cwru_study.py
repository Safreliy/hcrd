"""Leakage-resistant exploratory CWRU representation benchmark (protocol R1)."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.features import REPRESENTATIONS, representation_features_batch  # noqa: E402
from hcrd.datasets import load_cwru_drive_end  # noqa: E402


def deterministic_windows(
    signal: np.ndarray, *, length: int, count: int
) -> list[tuple[int, np.ndarray]]:
    available = signal.size // length
    if available < count:
        raise ValueError(f"only {available} nonoverlapping windows available; {count} requested")
    block_indices = np.rint(np.linspace(0, available - 1, count)).astype(int)
    if np.unique(block_indices).size != count:
        raise RuntimeError("window selection unexpectedly produced duplicates")
    return [
        (int(block * length), signal[block * length : (block + 1) * length].copy())
        for block in block_indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=PROJECT / "data" / "raw" / "cwru"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "cwru_r1"
    )
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--windows-per-record", type=int, default=24)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend", choices=("serial", "thread", "process"), default="serial"
    )
    parser.add_argument(
        "--representations",
        nargs="*",
        default=list(REPRESENTATIONS),
        choices=sorted(REPRESENTATIONS),
    )
    args = parser.parse_args()

    if not (args.data / "manifest.json").exists():
        raise FileNotFoundError(
            f"{args.data / 'manifest.json'} is missing; run experiments/download_cwru.py"
        )
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    labels = sorted({str(item["label"]) for item in manifest})

    samples: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    record_metadata: list[dict[str, object]] = []
    for item in manifest:
        record_id = int(item["record_id"])
        signal, rpm = load_cwru_drive_end(args.data / f"{record_id}.mat")
        record_metadata.append(
            {
                "record_id": record_id,
                "label": item["label"],
                "load_hp": int(item["load_hp"]),
                "rpm": rpm,
                "samples": int(signal.size),
                "sha256": item["sha256"],
            }
        )
        for window_index, (start, window) in enumerate(
            deterministic_windows(
                signal, length=args.window_length, count=args.windows_per_record
            )
        ):
            samples.append(
                {
                    "sample_index": len(samples),
                    "record_id": record_id,
                    "label": str(item["label"]),
                    "load_hp": int(item["load_hp"]),
                    "window_index": window_index,
                    "start": start,
                }
            )
            windows.append(window)

    args.output.mkdir(parents=True, exist_ok=True)
    feature_times: dict[str, float] = {}
    features_by_representation: dict[str, np.ndarray] = {}
    for representation in args.representations:
        started = time.perf_counter()
        features = representation_features_batch(
            windows,
            representation,
            n_components=args.components,
            workers=args.workers,
            backend=args.backend,
        )
        feature_times[representation] = time.perf_counter() - started
        features_by_representation[representation] = features
        np.save(args.output / f"features_{representation}.npy", features)
        print(
            json.dumps(
                {
                    "representation": representation,
                    "feature_seconds": feature_times[representation],
                }
            ),
            flush=True,
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    targets = np.asarray([labels.index(str(sample["label"])) for sample in samples])
    loads = np.asarray([int(sample["load_hp"]) for sample in samples])
    predictions_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for representation, feature_matrix in features_by_representation.items():
        for held_out_load in sorted(np.unique(loads)):
            train = loads != held_out_load
            test = ~train
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=0),
            )
            started = time.perf_counter()
            model.fit(feature_matrix[train], targets[train])
            predicted = model.predict(feature_matrix[test])
            model_seconds = time.perf_counter() - started
            matrix = confusion_matrix(targets[test], predicted, labels=np.arange(len(labels)))
            balanced_accuracy = float(balanced_accuracy_score(targets[test], predicted))
            macro_f1 = float(f1_score(targets[test], predicted, average="macro"))
            fold_rows.append(
                {
                    "representation": representation,
                    "held_out_load_hp": int(held_out_load),
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": macro_f1,
                    "model_seconds": model_seconds,
                    "confusion_matrix": json.dumps(matrix.tolist()),
                }
            )
            for sample_index, predicted_index in zip(
                np.flatnonzero(test), predicted, strict=True
            ):
                predictions_rows.append(
                    {
                        **samples[int(sample_index)],
                        "representation": representation,
                        "predicted_label": labels[int(predicted_index)],
                        "correct": int(predicted_index == targets[sample_index]),
                    }
                )

    with (args.output / "fold_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions_rows[0]))
        writer.writeheader()
        writer.writerows(predictions_rows)

    summary: list[dict[str, object]] = []
    raw_folds = {
        int(row["held_out_load_hp"]): float(row["balanced_accuracy"])
        for row in fold_rows
        if row["representation"] == "raw"
    }
    for representation in args.representations:
        selected = [row for row in fold_rows if row["representation"] == representation]
        scores = np.asarray([float(row["balanced_accuracy"]) for row in selected])
        f1_scores = np.asarray([float(row["macro_f1"]) for row in selected])
        deltas = np.asarray(
            [
                float(row["balanced_accuracy"])
                - raw_folds[int(row["held_out_load_hp"])]
                for row in selected
            ]
        )
        summary.append(
            {
                "representation": representation,
                "mean_balanced_accuracy": float(np.mean(scores)),
                "min_balanced_accuracy": float(np.min(scores)),
                "max_balanced_accuracy": float(np.max(scores)),
                "mean_macro_f1": float(np.mean(f1_scores)),
                "mean_delta_vs_raw": float(np.mean(deltas)),
                "feature_seconds": feature_times[representation],
                "milliseconds_per_window": 1000.0
                * feature_times[representation]
                / len(windows),
            }
        )
    metadata = {
        "protocol": "R1 / version 0.4",
        "exploratory": True,
        "window_length": args.window_length,
        "windows_per_record": args.windows_per_record,
        "components": args.components,
        "workers": args.workers,
        "parallel_backend": args.backend,
        "sample_count": len(samples),
        "labels": labels,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "records": record_metadata,
        "summary": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
