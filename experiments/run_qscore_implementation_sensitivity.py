#!/usr/bin/env python3
"""Extract and evaluate qscore implementation sensitivities for E2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from experiments.run_lcms_eic_e1 import (  # noqa: E402
    _weighted_ap,
    _weighted_ap_preparation,
)
from experiments.run_ms_metrics_e2 import (  # noqa: E402
    _retention_time_seconds,
    global_window_qscore,
)

SEED = 20260826
DATASETS = ("falkor", "mesoscope")
_BOUNDS: pd.DataFrame | None = None


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


def _initialise(labels_path: str) -> None:
    global _BOUNDS
    _BOUNDS = pd.read_csv(labels_path)


def _process_file(path_string: str) -> np.ndarray:
    from pyteomics import mzml

    if _BOUNDS is None:
        raise RuntimeError("worker bounds not initialized")
    bounds = _BOUNDS
    count = len(bounds)
    min_mz = bounds["min_mz"].to_numpy(float)
    max_mz = bounds["max_mz"].to_numpy(float)
    min_rt = bounds["min_rt"].to_numpy(float)
    max_rt = bounds["max_rt"].to_numpy(float)
    times: list[list[float]] = [[] for _ in range(count)]
    intensities: list[list[float]] = [[] for _ in range(count)]
    with mzml.MzML(path_string, use_index=False) as reader:
        for spectrum in reader:
            if int(spectrum.get("ms level", 1)) != 1:
                continue
            retention_time = _retention_time_seconds(spectrum)
            if retention_time is None:
                continue
            active = np.flatnonzero((min_rt <= retention_time) & (retention_time <= max_rt))
            if active.size == 0:
                continue
            mz = np.asarray(spectrum.get("m/z array", []), dtype=float)
            intensity = np.asarray(spectrum.get("intensity array", []), dtype=float)
            if mz.size == 0 or mz.size != intensity.size:
                continue
            left = np.searchsorted(mz, min_mz[active], side="left")
            right = np.searchsorted(mz, max_mz[active], side="right")
            for feature, start, stop in zip(active, left, right, strict=True):
                if stop <= start:
                    continue
                value = float(np.sum(intensity[start:stop]))
                if np.isfinite(value):
                    times[int(feature)].append(retention_time)
                    intensities[int(feature)].append(value)
    output = np.full((count, 2), np.nan, dtype=np.float32)
    for index, (x, y) in enumerate(zip(times, intensities, strict=True)):
        if len(set(x)) >= 5:
            output[index] = global_window_qscore(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            )
    return output


def extract(dataset: str, mzml_dir: Path, labels: Path, output: Path, workers: int) -> None:
    files = sorted(mzml_dir.glob("*.mzML"))
    if not files:
        raise RuntimeError(f"no mzML files in {mzml_dir}")
    bounds = pd.read_csv(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    cube = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(len(files), len(bounds), 2),
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise,
        initargs=(str(labels),),
    ) as executor:
        for index, values in enumerate(executor.map(_process_file, map(str, files))):
            cube[index] = values
            print(f"{dataset}: {index + 1}/{len(files)}", flush=True)
    cube.flush()
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "minimum_points": 5,
        "mzml_files": [path.name for path in files],
        "labels_sha256": _sha256(labels),
        "runner_sha256": _sha256(Path(__file__)),
    }
    _write_json(output.with_suffix(".metadata.json"), metadata)


def _summaries(cube: np.ndarray) -> dict[str, np.ndarray]:
    finite = np.isfinite(cube[:, :, 0])
    with np.errstate(all="ignore"):
        median = np.nanmedian(cube, axis=0)
        q90 = np.nanquantile(cube, 0.9, axis=0)
        maximum = np.nanmax(cube, axis=0)
    correlation = np.where(np.isfinite(cube[:, :, 1]), cube[:, :, 1], -np.inf)
    ordered = np.sort(correlation, axis=0)
    second_correlation = ordered[-2] if ordered.shape[0] >= 2 else ordered[-1]
    second_correlation[~np.isfinite(second_correlation)] = np.nan
    availability = np.mean(finite, axis=0)[:, None]
    values = {
        "q2": median,
        "q5": np.column_stack(
            [median[:, 0], median[:, 1], maximum[:, 0], maximum[:, 1], second_correlation]
        ),
        "q7": np.column_stack(
            [median, q90, maximum, availability]
        ),
    }
    return {
        key: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for key, value in values.items()
    }


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


def _paired_bootstrap(labels: np.ndarray, left: np.ndarray, right: np.ndarray, replicates: int) -> dict[str, object]:
    left_preparation = _weighted_ap_preparation(left)
    right_preparation = _weighted_ap_preparation(right)
    rng = np.random.default_rng(SEED)
    probability = np.full(labels.size, 1.0 / labels.size)
    differences = np.empty(replicates)
    for replicate in range(replicates):
        weights = rng.multinomial(labels.size, probability).astype(float)
        differences[replicate] = _weighted_ap(labels, weights, *left_preparation) - _weighted_ap(
            labels, weights, *right_preparation
        )
    tail = min(
        (np.sum(differences <= 0.0) + 1.0) / (replicates + 1.0),
        (np.sum(differences >= 0.0) + 1.0) / (replicates + 1.0),
    )
    return {
        "difference": float(
            average_precision_score(labels, left) - average_precision_score(labels, right)
        ),
        "bootstrap_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(min(1.0, 2.0 * tail)),
    }


def _fidelity(dataset_values: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, object]:
    author_path = (
        PROJECT
        / "third_party"
        / "MS_metrics"
        / "made_data_FT2040"
        / "features_extracted.csv"
    )
    author = pd.read_csv(author_path).set_index("feature")
    names = (
        PROJECT / "results" / "ms_metrics_e2" / "falkor" / "feature_names.txt"
    ).read_text(encoding="utf-8").splitlines()
    kept_names = np.asarray(names)[keep]
    common_mask = np.asarray([name in author.index for name in kept_names])
    common = kept_names[common_mask]
    mapping = {
        "q_current_med_snr": (dataset_values["q_current"][common_mask, 0], "med_SNR"),
        "q_current_med_cor": (dataset_values["q_current"][common_mask, 1], "med_cor"),
        "q_min5_med_snr": (dataset_values["q_min5"][common_mask, 0], "med_SNR"),
        "q_min5_med_cor": (dataset_values["q_min5"][common_mask, 1], "med_cor"),
        "q_author5_max_snr": (dataset_values["q_author5"][common_mask, 2], "max_SNR"),
        "q_author5_max_cor": (dataset_values["q_author5"][common_mask, 3], "max_cor"),
        "q_author5_second_cor": (dataset_values["q_author5"][common_mask, 4], "medtop3_cor"),
    }
    output = {}
    for name, (computed, column) in mapping.items():
        observed = author.loc[common, column].to_numpy(float)
        finite = np.isfinite(computed) & np.isfinite(observed)
        output[name] = {
            "author_column": column,
            "n": int(np.sum(finite)),
            "pearson": float(pearsonr(computed[finite], observed[finite]).statistic),
            "spearman": float(spearmanr(computed[finite], observed[finite]).statistic),
        }
    return output


def evaluate(output: Path, bootstrap: int) -> None:
    values = {}
    labels = {}
    keep_masks = {}
    for dataset in DATASETS:
        directory = PROJECT / "results" / "ms_metrics_e2" / dataset
        encoded = np.asarray(np.load(directory / "labels.npy"), dtype=np.int8)
        keep = encoded >= 0
        keep_masks[dataset] = keep
        labels[dataset] = encoded[keep]
        current_cube = np.asarray(np.load(directory / "per_file_qscore.npy", mmap_mode="r"))
        min5_cube = np.asarray(
            np.load(
                PROJECT
                / "results"
                / "qscore_implementation_sensitivity"
                / dataset
                / "per_file_qscore_min5.npy",
                mmap_mode="r",
            )
        )
        current = _summaries(current_cube)
        min5 = _summaries(min5_cube)
        stored = np.asarray(np.load(directory / "qscore.npy", mmap_mode="r"))
        if not np.allclose(current["q2"], stored, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{dataset}: current qscore aggregation mismatch")
        hcrd = np.asarray(np.load(directory / "hcrd_8_q.npy", mmap_mode="r"))
        hcrd_base = hcrd[:, :-2]
        variants = {
            "q_current": current["q2"],
            "q_min5": min5["q2"],
            "q_author5": min5["q5"],
            "q_multi7": current["q7"],
        }
        values[dataset] = {
            name: {
                "qscore": matrix[keep],
                "hcrd_plus_qscore": np.concatenate([hcrd_base, matrix], axis=1)[keep],
            }
            for name, matrix in variants.items()
        }

    rows = []
    results = {}
    for source, target in (("falkor", "mesoscope"), ("mesoscope", "falkor")):
        direction = f"{source}_to_{target}"
        results[direction] = {}
        for variant in ("q_current", "q_min5", "q_author5", "q_multi7"):
            scores = {}
            for representation in ("qscore", "hcrd_plus_qscore"):
                model = _model()
                model.fit(values[source][variant][representation], labels[source])
                scores[representation] = model.predict_proba(
                    values[target][variant][representation]
                )[:, 1]
            comparison = _paired_bootstrap(
                labels[target], scores["hcrd_plus_qscore"], scores["qscore"], bootstrap
            )
            results[direction][variant] = {
                "qscore_width": int(values[source][variant]["qscore"].shape[1]),
                "hcrd_plus_qscore_width": int(
                    values[source][variant]["hcrd_plus_qscore"].shape[1]
                ),
                "qscore_ap": float(average_precision_score(labels[target], scores["qscore"])),
                "hcrd_plus_qscore_ap": float(
                    average_precision_score(labels[target], scores["hcrd_plus_qscore"])
                ),
                "qscore_roc_auc": float(roc_auc_score(labels[target], scores["qscore"])),
                "hcrd_plus_qscore_roc_auc": float(
                    roc_auc_score(labels[target], scores["hcrd_plus_qscore"])
                ),
                "comparison": comparison,
            }
            for row_index, label in enumerate(labels[target]):
                rows.append(
                    {
                        "direction": direction,
                        "variant": variant,
                        "target_row": row_index,
                        "label": int(label),
                        "score_qscore": float(scores["qscore"][row_index]),
                        "score_hcrd_plus_qscore": float(
                            scores["hcrd_plus_qscore"][row_index]
                        ),
                    }
                )

    summary = {
        "protocol": "qscore-implementation-sensitivity-v1",
        "bootstrap_replicates": bootstrap,
        "directions": results,
        "falkor_author_output_fidelity": _fidelity(
            {
                name: pair["qscore"]
                for name, pair in values["falkor"].items()
            },
            keep_masks["falkor"],
        ),
        "limitation": "author per-detected-peak qscore outputs are unavailable for MESOSCOPE",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    protocol = PROJECT / "docs" / "qscore_implementation_sensitivity_protocol.md"
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "protocol": protocol.relative_to(PROJECT).as_posix(),
        "protocol_sha256": _sha256(protocol),
        "runner_sha256": _sha256(Path(__file__)),
    }
    _write_json(output / "metadata.json", metadata)
    print(json.dumps(summary, indent=2))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser("extract-min5")
    extract_parser.add_argument("--dataset", choices=DATASETS, required=True)
    extract_parser.add_argument("--mzml-dir", type=Path, required=True)
    extract_parser.add_argument("--labels", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--workers", type=int, default=4)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--bootstrap", type=int, default=10_000)
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "qscore_implementation_sensitivity",
    )
    args = parser.parse_args()
    if args.command == "extract-min5":
        extract(args.dataset, args.mzml_dir, args.labels, args.output, args.workers)
    else:
        evaluate(args.output, args.bootstrap)


if __name__ == "__main__":
    main()
