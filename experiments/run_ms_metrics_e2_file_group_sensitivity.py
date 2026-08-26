#!/usr/bin/env python3
"""Acquisition-file delete-group sensitivity for E2 HCRD transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260826
DATASETS = ("falkor", "mesoscope")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
    )


def _nan_summary(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized median, linear 0.9 quantile, and maximum over axis zero."""

    ordered = np.sort(block, axis=0)
    flat = ordered.reshape(ordered.shape[0], -1)
    counts = np.sum(~np.isnan(flat), axis=0)
    median = np.full(flat.shape[1], np.nan, dtype=ordered.dtype)
    q90 = np.full(flat.shape[1], np.nan, dtype=ordered.dtype)
    maximum = np.full(flat.shape[1], np.nan, dtype=ordered.dtype)
    for count in np.unique(counts[counts > 0]):
        selected = counts == count
        values = flat[: int(count), selected]
        median[selected] = np.median(values, axis=0)
        q90[selected] = np.quantile(values, 0.9, axis=0)
        maximum[selected] = values[-1]
    shape = ordered.shape[1:]
    return median.reshape(shape), q90.reshape(shape), maximum.reshape(shape)


def _file_folds(file_count: int, fold_count: int, dataset_index: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED + 10_000 * dataset_index)
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(rng.permutation(file_count), fold_count)]


def _aggregate(
    dataset: str, retained_files: np.ndarray, *, block_size: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    directory = PROJECT / "results" / "ms_metrics_e2" / dataset
    labels_all = np.asarray(np.load(directory / "labels.npy"), dtype=np.int8)
    keep = labels_all >= 0
    qscore_cube = np.load(directory / "per_file_qscore.npy", mmap_mode="r")
    hcrd_cube = np.load(directory / "per_file_hcrd_8.npy", mmap_mode="r")
    file_count, feature_count, width = hcrd_cube.shape
    if qscore_cube.shape != (file_count, feature_count, 2) or width != 948:
        raise ValueError(f"{dataset}: per-file cache schema mismatch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        qscore = np.nanmedian(np.asarray(qscore_cube[retained_files]), axis=0)
    qscore = np.nan_to_num(qscore, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hcrd = np.empty((feature_count, 3 * width + 3), dtype=np.float32)
    for start in range(0, feature_count, block_size):
        stop = min(feature_count, start + block_size)
        block = np.asarray(hcrd_cube[retained_files, start:stop, :])
        available = np.mean(np.isfinite(block[:, :, 0]), axis=0)[:, None]
        median, q90, maximum = _nan_summary(block)
        hcrd[start:stop] = np.nan_to_num(
            np.concatenate(
                [median, q90, maximum, available, qscore[start:stop]], axis=1
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    print(
        f"{dataset}: aggregated {retained_files.size}/{file_count} files",
        flush=True,
    )
    return labels_all[keep], {"qscore": qscore[keep], "hcrd_8_q": hcrd[keep]}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def _evaluate_fold(
    fold: int,
    folds: dict[str, list[np.ndarray]],
    file_counts: dict[str, int],
    block_size: int,
) -> list[dict[str, object]]:
    datasets = {}
    removed_counts = {}
    for dataset in DATASETS:
        removed = folds[dataset][fold]
        retained = np.setdiff1d(
            np.arange(file_counts[dataset], dtype=np.int64),
            removed,
            assume_unique=True,
        )
        datasets[dataset] = _aggregate(dataset, retained, block_size=block_size)
        removed_counts[dataset] = int(removed.size)
    rows: list[dict[str, object]] = []
    for source, target in (("falkor", "mesoscope"), ("mesoscope", "falkor")):
        source_y, source_x = datasets[source]
        target_y, target_x = datasets[target]
        ap = {}
        for representation in ("qscore", "hcrd_8_q"):
            model = _model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                model.fit(source_x[representation], source_y)
            scores = model.predict_proba(target_x[representation])[:, 1]
            ap[representation] = float(average_precision_score(target_y, scores))
        rows.append(
            {
                "fold": fold,
                "direction": f"{source}_to_{target}",
                "source_removed_files": removed_counts[source],
                "target_removed_files": removed_counts[target],
                "ap_qscore": ap["qscore"],
                "ap_hcrd_8_q": ap["hcrd_8_q"],
                "delta_hcrd_qscore": ap["hcrd_8_q"] - ap["qscore"],
            }
        )
    print(f"completed fold {fold}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "ms_metrics_e2_file_group_sensitivity",
    )
    args = parser.parse_args()
    if args.fold_count < 2 or args.block_size < 1 or args.jobs < 1:
        raise SystemExit("fold-count must be at least two; block-size and jobs positive")
    selected_folds = tuple(args.folds or range(args.fold_count))
    if any(fold < 0 or fold >= args.fold_count for fold in selected_folds):
        raise SystemExit("fold index outside configured fold count")

    file_counts = {
        dataset: int(
            np.load(
                PROJECT
                / "results"
                / "ms_metrics_e2"
                / dataset
                / "per_file_hcrd_8.npy",
                mmap_mode="r",
            ).shape[0]
        )
        for dataset in DATASETS
    }
    folds = {
        dataset: _file_folds(file_counts[dataset], args.fold_count, index)
        for index, dataset in enumerate(DATASETS)
    }
    nested = joblib.Parallel(n_jobs=min(args.jobs, len(selected_folds)), verbose=10)(
        joblib.delayed(_evaluate_fold)(
            fold, folds, file_counts, args.block_size
        )
        for fold in selected_folds
    )
    rows = [row for fold_rows in nested for row in fold_rows]
    rows.sort(key=lambda row: (int(row["fold"]), str(row["direction"])))

    summary: dict[str, object] = {
        "analysis": "acquisition-file delete-group representation and model refit",
        "fold_count": args.fold_count,
        "evaluated_folds": list(selected_folds),
        "directions": {},
    }
    for direction in ("falkor_to_mesoscope", "mesoscope_to_falkor"):
        direction_rows = [row for row in rows if row["direction"] == direction]
        values = np.asarray([float(row["delta_hcrd_qscore"]) for row in direction_rows])
        summary["directions"][direction] = {
            "delta_mean": float(np.mean(values)),
            "delta_min": float(np.min(values)),
            "delta_max": float(np.max(values)),
            "positive_folds": int(np.sum(values > 0.0)),
            "evaluated_folds": int(values.size),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "folds.csv", rows)
    _write_json(args.output / "summary.json", summary)
    protocol = PROJECT / "docs" / "ms_metrics_e2_refit_sensitivity_protocol.md"
    cube_paths = {
        dataset: {
            name: PROJECT
            / "results"
            / "ms_metrics_e2"
            / dataset
            / f"per_file_{name}.npy"
            for name in ("qscore", "hcrd_8")
        }
        for dataset in DATASETS
    }
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": protocol.relative_to(PROJECT).as_posix(),
        "protocol_sha256": _sha256(protocol),
        "runner_sha256": _sha256(Path(__file__)),
        "cube_sha256": {
            dataset: {name: _sha256(path) for name, path in paths.items()}
            for dataset, paths in cube_paths.items()
        },
    }
    _write_json(args.output / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
