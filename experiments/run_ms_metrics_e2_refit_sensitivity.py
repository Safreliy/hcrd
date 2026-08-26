#!/usr/bin/env python3
"""Source-refit RT-block bootstrap for the E2 LC-MS transfer result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260826
DATASETS = {
    "falkor": "made_data_FT2040",
    "mesoscope": "made_data_MS3000",
}
REPRESENTATIONS = ("qscore", "hcrd_8_q")
_CACHE: dict[str, tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]] = {}


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


def _matrix_path(dataset: str, representation: str) -> Path:
    return PROJECT / "results" / "ms_metrics_e2" / dataset / f"{representation}.npy"


def _load(dataset: str, block_seconds: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    key = f"{dataset}:{block_seconds}"
    if key in _CACHE:
        return _CACHE[key]
    directory = PROJECT / "results" / "ms_metrics_e2" / dataset
    encoded = np.asarray(np.load(directory / "labels.npy"), dtype=np.int8)
    keep = encoded >= 0
    label_path = (
        PROJECT
        / "third_party"
        / "MS_metrics"
        / DATASETS[dataset]
        / "classified_feats.csv"
    )
    table = pd.read_csv(label_path)
    if len(table) != encoded.size:
        raise ValueError(f"{dataset}: label-table alignment failure")
    expected = table["feat_class"].map({"Good": 1, "Bad": 0}).fillna(-1).to_numpy(np.int8)
    if not np.array_equal(encoded, expected):
        raise ValueError(f"{dataset}: encoded labels do not match source table")
    midpoint = 0.5 * (table["min_rt"].to_numpy(float) + table["max_rt"].to_numpy(float))
    clusters = np.floor(midpoint[keep] / block_seconds).astype(np.int64)
    matrices = {
        name: np.asarray(np.load(_matrix_path(dataset, name), mmap_mode="r")[keep])
        for name in REPRESENTATIONS
    }
    value = encoded[keep], matrices, clusters
    _CACHE[key] = value
    return value


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


def _cluster_indices(
    clusters: np.ndarray, rng: np.random.Generator, labels: np.ndarray
) -> np.ndarray:
    groups = np.unique(clusters)
    for _ in range(100):
        chosen = rng.choice(groups, size=groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == group) for group in chosen])
        if np.unique(labels[indices]).size == 2:
            return indices
    raise RuntimeError("could not draw a two-class cluster resample")


def _replicate(block_seconds: int, replicate: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED + 1_000_000 * block_seconds + replicate)
    output: list[dict[str, object]] = []
    for direction_index, (source, target) in enumerate(
        (("falkor", "mesoscope"), ("mesoscope", "falkor"))
    ):
        source_y, source_x, source_clusters = _load(source, block_seconds)
        target_y, target_x, target_clusters = _load(target, block_seconds)
        source_indices = _cluster_indices(source_clusters, rng, source_y)
        target_indices = _cluster_indices(target_clusters, rng, target_y)
        scores: dict[str, np.ndarray] = {}
        for name in REPRESENTATIONS:
            model = _model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                model.fit(source_x[name][source_indices], source_y[source_indices])
            scores[name] = model.predict_proba(target_x[name])[:, 1]
        ap = {
            name: float(
                average_precision_score(
                    target_y[target_indices], scores[name][target_indices]
                )
            )
            for name in REPRESENTATIONS
        }
        output.append(
            {
                "block_seconds": block_seconds,
                "replicate": replicate,
                "direction": f"{source}_to_{target}",
                "source_rows": int(source_indices.size),
                "target_rows": int(target_indices.size),
                "source_positive": int(np.sum(source_y[source_indices] == 1)),
                "target_positive": int(np.sum(target_y[target_indices] == 1)),
                "ap_qscore": ap["qscore"],
                "ap_hcrd_8_q": ap["hcrd_8_q"],
                "delta_hcrd_qscore": ap["hcrd_8_q"] - ap["qscore"],
                "direction_index": direction_index,
            }
        )
    return output


def _interval(values: np.ndarray) -> dict[str, object]:
    probability = float(np.mean(values > 0.0))
    two_sided = float(min(1.0, 2.0 * min(probability, 1.0 - probability)))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "percentile_95_ci": np.quantile(values, [0.025, 0.975]).tolist(),
        "positive_fraction": probability,
        "two_sided_sign_probability": two_sided,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--primary-replicates", type=int, default=1000)
    parser.add_argument("--sensitivity-replicates", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "ms_metrics_e2_refit_sensitivity",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.primary_replicates < 20 or args.sensitivity_replicates < 20:
        raise SystemExit("positive jobs and at least 20 replicates are required")
    designs = ((30, args.sensitivity_replicates), (60, args.primary_replicates), (120, args.sensitivity_replicates))
    tasks = [
        (block_seconds, replicate)
        for block_seconds, replicates in designs
        for replicate in range(replicates)
    ]
    # Load each block design once in the parent process.  The thread backend
    # then shares the read-only matrices instead of materialising one filtered
    # 2,847-column bank in every worker process.
    for block_seconds, _ in designs:
        for dataset in DATASETS:
            _load(dataset, block_seconds)
    nested = joblib.Parallel(
        n_jobs=args.jobs,
        verbose=10,
        prefer="threads",
    )(
        joblib.delayed(_replicate)(block_seconds, replicate)
        for block_seconds, replicate in tasks
    )
    rows = [row for pair in nested for row in pair]
    rows.sort(key=lambda row: (int(row["block_seconds"]), int(row["replicate"]), int(row["direction_index"])))
    summary: dict[str, object] = {
        "estimand": "source-refit and paired target RT-block AP difference",
        "primary_block_seconds": 60,
        "designs": {str(block): replicates for block, replicates in designs},
        "directions": {},
    }
    for block_seconds, _ in designs:
        block_rows = [row for row in rows if int(row["block_seconds"]) == block_seconds]
        block_output: dict[str, object] = {}
        for direction in ("falkor_to_mesoscope", "mesoscope_to_falkor"):
            direction_rows = [row for row in block_rows if row["direction"] == direction]
            block_output[direction] = {
                "hcrd_8_q_minus_qscore": _interval(
                    np.asarray([float(row["delta_hcrd_qscore"]) for row in direction_rows])
                ),
            }
        summary["directions"][str(block_seconds)] = block_output

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "replicates.csv", rows)
    _write_json(args.output / "summary.json", summary)
    protocol = PROJECT / "docs" / "ms_metrics_e2_refit_sensitivity_protocol.md"
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": __import__("sys").argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "protocol": protocol.relative_to(PROJECT).as_posix(),
        "protocol_sha256": _sha256(protocol),
        "runner_sha256": _sha256(Path(__file__)),
        "matrix_sha256": {
            dataset: {
                name: _sha256(_matrix_path(dataset, name)) for name in REPRESENTATIONS
            }
            for dataset in DATASETS
        },
    }
    _write_json(args.output / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
