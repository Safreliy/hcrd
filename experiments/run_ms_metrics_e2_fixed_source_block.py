#!/usr/bin/env python3
"""Fixed-source target RT-block bootstrap for the E2 transfer result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    from experiments.run_lcms_eic_e1 import _weighted_ap, _weighted_ap_preparation
except ModuleNotFoundError:  # Direct ``python experiments/script.py`` execution.
    from run_lcms_eic_e1 import _weighted_ap, _weighted_ap_preparation

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260826
DATASETS = {
    "falkor": "made_data_FT2040",
    "mesoscope": "made_data_MS3000",
}
REPRESENTATIONS = ("qscore", "hcrd_8_q")
DIRECTIONS = (("falkor", "mesoscope"), ("mesoscope", "falkor"))


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


def _load_target(
    dataset: str,
    block_seconds: int,
    data_root: Path,
    third_party_root: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    directory = data_root / dataset
    encoded = np.asarray(np.load(directory / "labels.npy"), dtype=np.int8)
    keep = encoded >= 0
    table = pd.read_csv(
        third_party_root
        / DATASETS[dataset]
        / "classified_feats.csv"
    )
    if len(table) != encoded.size:
        raise ValueError(f"{dataset}: label-table alignment failure")
    expected = (
        table["feat_class"].map({"Good": 1, "Bad": 0}).fillna(-1).to_numpy(np.int8)
    )
    if not np.array_equal(encoded, expected):
        raise ValueError(f"{dataset}: encoded labels do not match source table")
    midpoint = 0.5 * (
        table["min_rt"].to_numpy(float) + table["max_rt"].to_numpy(float)
    )
    blocks = np.floor(midpoint[keep] / block_seconds).astype(np.int64)
    matrices = {
        name: np.asarray(np.load(directory / f"{name}.npy", mmap_mode="r")[keep])
        for name in REPRESENTATIONS
    }
    return encoded[keep], matrices, blocks


def _scores(
    source: str,
    target: str,
    matrices: dict[str, np.ndarray],
    model_dir: Path,
) -> dict[str, np.ndarray]:
    return {
        name: joblib.load(
            model_dir / f"model_{source}_to_{target}_{name}.joblib"
        ).predict_proba(matrices[name])[:, 1]
        for name in REPRESENTATIONS
    }


def _block_index(blocks: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(blocks, return_inverse=True)
    return inverse, int(np.max(inverse) + 1)


def _draw_weights(
    inverse: np.ndarray,
    block_count: int,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.full(block_count, 1.0 / block_count)
    for _ in range(100):
        multiplicity = rng.multinomial(block_count, probability).astype(float)
        weights = multiplicity[inverse]
        if np.sum(weights[labels == 0]) > 0 and np.sum(weights[labels == 1]) > 0:
            return weights
    raise RuntimeError("could not draw a two-class target block resample")


def _run_cell(
    source: str,
    target: str,
    block_seconds: int,
    replicates: int,
    direction_index: int,
    data_root: Path,
    third_party_root: Path,
    model_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    labels, matrices, blocks = _load_target(
        target, block_seconds, data_root, third_party_root
    )
    scores = _scores(source, target, matrices, model_dir)
    preparations = {
        name: _weighted_ap_preparation(value) for name, value in scores.items()
    }
    inverse, block_count = _block_index(blocks)
    rng = np.random.default_rng(
        SEED + 1_000_000 * block_seconds + 10_000 * direction_index
    )
    rows: list[dict[str, object]] = []
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        weights = _draw_weights(inverse, block_count, labels, rng)
        ap = {}
        for name in REPRESENTATIONS:
            order, starts = preparations[name]
            ap[name] = _weighted_ap(labels, weights, order, starts)
        difference = float(ap["hcrd_8_q"] - ap["qscore"])
        differences[replicate] = difference
        rows.append(
            {
                "block_seconds": block_seconds,
                "replicate": replicate,
                "direction": f"{source}_to_{target}",
                "sampled_feature_multiplicity": int(np.sum(weights)),
                "sampled_positive_multiplicity": int(np.sum(weights[labels == 1])),
                "ap_qscore": float(ap["qscore"]),
                "ap_hcrd_8_q": float(ap["hcrd_8_q"]),
                "delta_hcrd_qscore": difference,
            }
        )
    positive_fraction = float(np.mean(differences > 0.0))
    two_sided = float(
        min(1.0, 2.0 * min(positive_fraction, 1.0 - positive_fraction))
    )
    point_difference = float(
        average_precision_score(labels, scores["hcrd_8_q"])
        - average_precision_score(labels, scores["qscore"])
    )
    return rows, {
        "target_block_count": block_count,
        "target_feature_count": int(labels.size),
        "point_ap_difference": point_difference,
        "bootstrap_mean": float(np.mean(differences)),
        "bootstrap_median": float(np.median(differences)),
        "percentile_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "positive_fraction": positive_fraction,
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
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT / "results" / "ms_metrics_e2",
    )
    parser.add_argument(
        "--third-party-root",
        type=Path,
        default=PROJECT / "third_party" / "MS_metrics",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT / "results" / "ms_metrics_e2" / "evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "ms_metrics_e2_fixed_source_block",
    )
    args = parser.parse_args()
    if args.replicates < 20:
        raise SystemExit("at least 20 replicates are required")

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "estimand": "fixed-source paired target RT-block AP difference",
        "primary_block_seconds": 60,
        "replicates_per_cell": args.replicates,
        "directions": {},
    }
    for block_seconds in (30, 60, 120):
        block_output: dict[str, object] = {}
        for direction_index, (source, target) in enumerate(DIRECTIONS):
            cell_rows, cell_summary = _run_cell(
                source,
                target,
                block_seconds,
                args.replicates,
                direction_index,
                args.data_root,
                args.third_party_root,
                args.model_dir,
            )
            rows.extend(cell_rows)
            block_output[f"{source}_to_{target}"] = {
                "hcrd_8_q_minus_qscore": cell_summary
            }
        summary["directions"][str(block_seconds)] = block_output

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "replicates.csv", rows)
    _write_json(args.output / "summary.json", summary)
    protocol = PROJECT / "docs" / "ms_metrics_e2_fixed_source_block_protocol.md"
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": [
            "python",
            "experiments/run_ms_metrics_e2_fixed_source_block.py",
            "--replicates",
            str(args.replicates),
        ],
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "protocol": protocol.relative_to(PROJECT).as_posix(),
        "protocol_sha256": _sha256(protocol),
        "runner_sha256": _sha256(Path(__file__)),
        "input_sha256": {
            dataset: {
                "labels": _sha256(args.data_root / dataset / "labels.npy"),
                **{
                    name: _sha256(args.data_root / dataset / f"{name}.npy")
                    for name in REPRESENTATIONS
                },
                "classified_features": _sha256(
                    args.third_party_root
                    / DATASETS[dataset]
                    / "classified_feats.csv"
                ),
            }
            for dataset in DATASETS
        },
        "model_sha256": {
            f"{source}_to_{target}": {
                name: _sha256(
                    args.model_dir / f"model_{source}_to_{target}_{name}.joblib"
                )
                for name in REPRESENTATIONS
            }
            for source, target in DIRECTIONS
        },
    }
    _write_json(args.output / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
