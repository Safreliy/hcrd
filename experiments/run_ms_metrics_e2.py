#!/usr/bin/env python3
"""Prospective cross-dataset E2 HILIC peak-quality benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import beta, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from experiments.run_lcms_eic_e1 import (
        _holm,
        _weighted_ap,
        _weighted_ap_preparation,
    )
except ModuleNotFoundError:  # Direct ``python experiments/script.py`` execution.
    from run_lcms_eic_e1 import _holm, _weighted_ap, _weighted_ap_preparation
from hcrd.lcms import eic_feature_bank


SEED = 20260825
PER_FILE_WIDTHS = {
    "qscore": 2,
    "domain": 111,
    "hcrd_1": 177,
    "hcrd_8": 948,
    "hcrd_geometry": 297,
    "area_only": 48,
}
FINAL_NAMES = (
    "qscore",
    "domain_q",
    "hcrd_1_q",
    "hcrd_8_q",
    "hcrd_geometry_q",
    "area_only_q",
)

_BOUNDS: pd.DataFrame | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialise_worker(labels_path: str) -> None:
    global _BOUNDS
    _BOUNDS = pd.read_csv(labels_path)


def _retention_time_seconds(spectrum: dict) -> float | None:
    scans = spectrum.get("scanList", {}).get("scan", [])
    value = scans[0].get("scan start time") if scans else None
    if value is None:
        value = spectrum.get("scan start time")
    if value is None:
        return None
    unit = str(getattr(value, "unit_info", "")).lower()
    result = float(value)
    if "minute" in unit:
        result *= 60.0
    return result


def global_window_qscore(
    retention_time: NDArray[np.float64], intensity: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Independent global-window implementation of the two-component score."""

    if retention_time.size < 5:
        return np.asarray([np.nan, np.nan])
    span = float(np.max(retention_time) - np.min(retention_time))
    if span <= 0.0 or float(np.max(intensity)) <= 0.0:
        return np.asarray([np.nan, np.nan])
    scaled = (retention_time - np.min(retention_time)) / span
    correlations = []
    curves = []
    for alpha in (2.5, 3.0, 4.0, 5.0):
        curve = beta.pdf(scaled, alpha, 5.0)
        curves.append(curve)
        correlations.append(np.corrcoef(curve, intensity)[0, 1])
    correlations_array = np.asarray(correlations)
    if not np.any(np.isfinite(correlations_array)):
        return np.asarray([np.nan, np.nan])
    best = int(np.nanargmax(correlations_array))
    curve = curves[best]
    residual = intensity / np.max(intensity) - curve / np.max(curve)
    differenced = np.diff(residual)
    denominator = float(np.std(differenced * np.max(intensity), ddof=1))
    snr = (
        float((np.max(intensity) - np.min(intensity)) / denominator)
        if denominator > 0.0
        else np.nan
    )
    return np.asarray([snr, correlations_array[best]], dtype=float)


def _process_mzml(path_string: str) -> dict[str, NDArray[np.float32]]:
    from pyteomics import mzml

    if _BOUNDS is None:
        raise RuntimeError("worker bounds were not initialized")
    bounds = _BOUNDS
    count = bounds.shape[0]
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
            for feature_index, start, stop in zip(active, left, right, strict=True):
                if stop <= start:
                    continue
                value = float(np.sum(intensity[start:stop]))
                if not np.isfinite(value):
                    continue
                times[int(feature_index)].append(retention_time)
                intensities[int(feature_index)].append(value)

    output = {
        name: np.full((count, width), np.nan, dtype=np.float32)
        for name, width in PER_FILE_WIDTHS.items()
    }
    for feature_index, (time_values, intensity_values) in enumerate(
        zip(times, intensities, strict=True)
    ):
        if len(set(time_values)) < 8:
            continue
        x = np.asarray(time_values, dtype=float)
        y = np.asarray(intensity_values, dtype=float)
        try:
            bank = eic_feature_bank(y, x)
        except (ValueError, RuntimeError):
            continue
        output["qscore"][feature_index] = global_window_qscore(x, y)
        for name in PER_FILE_WIDTHS:
            if name != "qscore":
                output[name][feature_index] = getattr(bank, name)
    return output


def _aggregate_per_file(
    output_dir: Path, file_count: int, feature_count: int
) -> None:
    qscore_cube = np.load(output_dir / "per_file_qscore.npy", mmap_mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        qscore = np.nanmedian(qscore_cube, axis=0)
    qscore = np.nan_to_num(qscore, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(output_dir / "qscore.npy", qscore)

    source_to_final = {
        "domain": "domain_q",
        "hcrd_1": "hcrd_1_q",
        "hcrd_8": "hcrd_8_q",
        "hcrd_geometry": "hcrd_geometry_q",
        "area_only": "area_only_q",
    }
    for source, final in source_to_final.items():
        width = PER_FILE_WIDTHS[source]
        cube = np.load(output_dir / f"per_file_{source}.npy", mmap_mode="r")
        final_width = 3 * width + 1 + qscore.shape[1]
        aggregate = np.lib.format.open_memmap(
            output_dir / f"{final}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(feature_count, final_width),
        )
        for start in range(0, feature_count, 32):
            stop = min(feature_count, start + 32)
            block = np.asarray(cube[:, start:stop, :])
            available = np.mean(np.isfinite(block[:, :, 0]), axis=0)[:, None]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median = np.nanmedian(block, axis=0)
                q90 = np.nanquantile(block, 0.9, axis=0)
                maximum = np.nanmax(block, axis=0)
            values = np.concatenate(
                [median, q90, maximum, available, qscore[start:stop]], axis=1
            )
            aggregate[start:stop] = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            )
        aggregate.flush()


def extract_dataset(
    *,
    dataset: str,
    mzml_dir: Path,
    labels_path: Path,
    source_archive: Path,
    repository: Path,
    output_dir: Path,
    workers: int,
) -> None:
    labels = pd.read_csv(labels_path)
    required = {"feature", "min_mz", "max_mz", "min_rt", "max_rt", "feat_class"}
    if not required <= set(labels.columns):
        raise ValueError("manual classification schema mismatch")
    files = sorted(mzml_dir.glob("*.mzML"))
    if not files:
        raise RuntimeError(f"no mzML files in {mzml_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_file = {
        name: np.lib.format.open_memmap(
            output_dir / f"per_file_{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(files), labels.shape[0], width),
        )
        for name, width in PER_FILE_WIDTHS.items()
    }
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_worker,
        initargs=(str(labels_path),),
    ) as executor:
        for file_index, result in enumerate(
            executor.map(_process_mzml, (str(path) for path in files)), start=0
        ):
            for name, values in result.items():
                per_file[name][file_index] = values
            print(f"{dataset}: {file_index + 1}/{len(files)} mzML files", flush=True)
    for values in per_file.values():
        values.flush()

    encoded_labels = np.full(labels.shape[0], -1, dtype=np.int8)
    encoded_labels[labels["feat_class"].eq("Bad").to_numpy()] = 0
    encoded_labels[labels["feat_class"].eq("Good").to_numpy()] = 1
    np.save(output_dir / "labels.npy", encoded_labels)
    (output_dir / "feature_names.txt").write_text(
        "\n".join(labels["feature"].astype(str)) + "\n", encoding="utf-8"
    )
    (output_dir / "mzml_files.txt").write_text(
        "\n".join(path.name for path in files) + "\n", encoding="utf-8"
    )
    _aggregate_per_file(output_dir, len(files), labels.shape[0])
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "protocol": "hcrd-e2-v1",
        "dataset": dataset,
        "mzml_file_count": len(files),
        "feature_count": int(labels.shape[0]),
        "good_count": int(np.sum(encoded_labels == 1)),
        "bad_count": int(np.sum(encoded_labels == 0)),
        "excluded_count": int(np.sum(encoded_labels < 0)),
        "source_archive_sha256": _sha256(source_archive),
        "labels_sha256": _sha256(labels_path),
        "source_repository_commit": commit,
        "final_widths": {
            name: int(np.load(output_dir / f"{name}.npy", mmap_mode="r").shape[1])
            for name in FINAL_NAMES
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


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


def _classification_metrics(
    labels: NDArray[np.int8], scores: NDArray[np.float64]
) -> dict[str, float]:
    predictions = scores >= 0.5
    true_positive = int(np.sum((labels == 1) & predictions))
    false_positive = int(np.sum((labels == 0) & predictions))
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "false_discovery_rate": float(
            false_positive / max(1, true_positive + false_positive)
        ),
        "good_features_found": float(true_positive / np.sum(labels == 1)),
    }


def _bootstrap_comparisons(
    labels: NDArray[np.int8],
    scores: dict[str, NDArray[np.float64]],
    replicates: int,
) -> dict[str, dict[str, object]]:
    preparations = {
        name: _weighted_ap_preparation(value) for name, value in scores.items()
    }
    rng = np.random.default_rng(SEED)
    boot = {name: np.empty(replicates) for name in scores}
    probability = np.full(labels.size, 1.0 / labels.size)
    for replicate in range(replicates):
        weights = rng.multinomial(labels.size, probability).astype(float)
        for name in scores:
            order, starts = preparations[name]
            boot[name][replicate] = _weighted_ap(labels, weights, order, starts)
    output = {}
    for name in ("domain_q", "hcrd_1_q", "hcrd_8_q", "hcrd_geometry_q", "area_only_q"):
        difference = boot[name] - boot["qscore"]
        p_value = min(
            1.0,
            2.0
            * min(
                (np.sum(difference <= 0.0) + 1.0) / (replicates + 1.0),
                (np.sum(difference >= 0.0) + 1.0) / (replicates + 1.0),
            ),
        )
        output[name] = {
            "versus": "qscore",
            "ap_difference": float(
                average_precision_score(labels, scores[name])
                - average_precision_score(labels, scores["qscore"])
            ),
            "bootstrap_95_ci": np.quantile(difference, [0.025, 0.975]).tolist(),
            "two_sided_bootstrap_p": float(p_value),
        }
    difference = boot["hcrd_8_q"] - boot["domain_q"]
    p_value = min(
        1.0,
        2.0
        * min(
            (np.sum(difference <= 0.0) + 1.0) / (replicates + 1.0),
            (np.sum(difference >= 0.0) + 1.0) / (replicates + 1.0),
        ),
    )
    output["hcrd_8_q_vs_domain_q"] = {
        "versus": "domain_q",
        "ap_difference": float(
            average_precision_score(labels, scores["hcrd_8_q"])
            - average_precision_score(labels, scores["domain_q"])
        ),
        "bootstrap_95_ci": np.quantile(difference, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(p_value),
    }
    difference = boot["hcrd_8_q"] - boot["hcrd_1_q"]
    p_value = min(
        1.0,
        2.0
        * min(
            (np.sum(difference <= 0.0) + 1.0) / (replicates + 1.0),
            (np.sum(difference >= 0.0) + 1.0) / (replicates + 1.0),
        ),
    )
    output["hcrd_8_q_vs_hcrd_1_q"] = {
        "versus": "hcrd_1_q",
        "ap_difference": float(
            average_precision_score(labels, scores["hcrd_8_q"])
            - average_precision_score(labels, scores["hcrd_1_q"])
        ),
        "bootstrap_95_ci": np.quantile(difference, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(p_value),
    }
    return output


def _load_dataset(directory: Path) -> tuple[NDArray[np.int8], dict[str, NDArray]]:
    labels = np.load(directory / "labels.npy")
    keep = labels >= 0
    return labels[keep], {
        name: np.asarray(np.load(directory / f"{name}.npy", mmap_mode="r")[keep])
        for name in FINAL_NAMES
    }


def fit_evaluate(
    falkor_dir: Path,
    mesoscope_dir: Path,
    repository: Path,
    output_dir: Path,
    bootstrap: int,
) -> None:
    dataset_directories = {"falkor": falkor_dir, "mesoscope": mesoscope_dir}
    datasets = {
        "falkor": _load_dataset(falkor_dir),
        "mesoscope": _load_dataset(mesoscope_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = (("falkor", "mesoscope"), ("mesoscope", "falkor"))
    results = {}
    primary_p = {}
    domain_p = {}
    level_p = {}
    for source, target in directions:
        source_y, source_x = datasets[source]
        target_y, target_x = datasets[target]
        scores = {}
        metrics = {}
        for name in FINAL_NAMES:
            model = _model()
            model.fit(source_x[name], source_y)
            scores[name] = model.predict_proba(target_x[name])[:, 1]
            metrics[name] = _classification_metrics(target_y, scores[name])
            joblib.dump(model, output_dir / f"model_{source}_to_{target}_{name}.joblib")
        comparisons = _bootstrap_comparisons(target_y, scores, bootstrap)
        key = f"{source}_to_{target}"
        primary_p[key] = comparisons["hcrd_8_q"]["two_sided_bootstrap_p"]
        domain_p[key] = comparisons["hcrd_8_q_vs_domain_q"][
            "two_sided_bootstrap_p"
        ]
        level_p[key] = comparisons["hcrd_8_q_vs_hcrd_1_q"][
            "two_sided_bootstrap_p"
        ]
        results[key] = {"metrics": metrics, "comparisons": comparisons}

    adjusted = _holm(primary_p)
    adjusted_domain = _holm(domain_p)
    adjusted_level = _holm(level_p)
    success_components = {}
    for key, result in results.items():
        result["comparisons"]["hcrd_8_q"]["holm_across_directions_p"] = adjusted[key]
        result["comparisons"]["hcrd_8_q_vs_domain_q"][
            "holm_across_directions_p"
        ] = adjusted_domain[key]
        result["comparisons"]["hcrd_8_q_vs_hcrd_1_q"][
            "holm_across_directions_p"
        ] = adjusted_level[key]
        main = result["comparisons"]["hcrd_8_q"]
        success_components[key] = {
            "positive_hcrd8_qscore_ap": main["ap_difference"] > 0.0,
            "positive_bootstrap_ci_lower": main["bootstrap_95_ci"][0] > 0.0,
            "hcrd8_exceeds_hcrd1": result["metrics"]["hcrd_8_q"][
                "average_precision"
            ]
            > result["metrics"]["hcrd_1_q"]["average_precision"],
            "holm_p_below_0_05": adjusted[key] < 0.05,
        }

    audit = {}
    published_path = repository / "made_data_FT2040" / "features_extracted.csv"
    if published_path.exists():
        published = pd.read_csv(published_path).set_index("feature")
        feature_names = (falkor_dir / "feature_names.txt").read_text().splitlines()
        computed = np.load(falkor_dir / "qscore.npy")
        common = [name for name in feature_names if name in published.index]
        indices = np.asarray([feature_names.index(name) for name in common])
        for computed_column, published_column in ((0, "med_SNR"), (1, "med_cor")):
            left = computed[indices, computed_column]
            right = published.loc[common, published_column].to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            audit[published_column] = {
                "n": int(np.sum(finite)),
                "pearson": float(pearsonr(left[finite], right[finite]).statistic),
                "spearman": float(spearmanr(left[finite], right[finite]).statistic),
            }
    result = {
        "protocol": "hcrd-e2-v1",
        "dataset_metadata": {
            name: {
                **json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
                "final_feature_sha256": {
                    representation: _sha256(directory / f"{representation}.npy")
                    for representation in FINAL_NAMES
                },
                "encoded_labels_sha256": _sha256(directory / "labels.npy"),
                "feature_names_sha256": _sha256(directory / "feature_names.txt"),
            }
            for name, directory in dataset_directories.items()
        },
        "runner_sha256": _sha256(Path(__file__)),
        "directions": results,
        "qscore_fidelity_audit_falkor": audit,
        "prospective_success": all(all(v.values()) for v in success_components.values()),
        "success_components": success_components,
        "bootstrap_replicates": bootstrap,
        "seed": SEED,
    }
    (output_dir / "e2_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-dataset")
    extract.add_argument("--dataset", required=True)
    extract.add_argument("--mzml-dir", type=Path, required=True)
    extract.add_argument("--labels", type=Path, required=True)
    extract.add_argument("--source-archive", type=Path, required=True)
    extract.add_argument("--repository", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    evaluate = commands.add_parser("fit-evaluate")
    evaluate.add_argument("--falkor-dir", type=Path, required=True)
    evaluate.add_argument("--mesoscope-dir", type=Path, required=True)
    evaluate.add_argument("--repository", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "extract-dataset":
        extract_dataset(
            dataset=args.dataset,
            mzml_dir=args.mzml_dir,
            labels_path=args.labels,
            source_archive=args.source_archive,
            repository=args.repository,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    else:
        fit_evaluate(
            args.falkor_dir,
            args.mesoscope_dir,
            args.repository,
            args.output_dir,
            args.bootstrap,
        )


if __name__ == "__main__":
    main()
