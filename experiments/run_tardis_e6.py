#!/usr/bin/env python3
"""Frozen E6 no-refit transfer to the released positive-polarity TARDIS/FAME QC data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from experiments.run_ms_metrics_e2 import (
        FINAL_NAMES,
        PER_FILE_WIDTHS,
        SEED,
        _aggregate_per_file,
        _bootstrap_comparisons,
        _load_dataset,
        _model,
        global_window_qscore,
    )
except ModuleNotFoundError:  # Direct ``python experiments/script.py`` execution.
    from run_ms_metrics_e2 import (
        FINAL_NAMES,
        PER_FILE_WIDTHS,
        SEED,
        _aggregate_per_file,
        _bootstrap_comparisons,
        _load_dataset,
        _model,
        global_window_qscore,
    )

from hcrd.lcms import eic_feature_bank


E6_SEED = 20260825
EXPECTED_ARCHIVE_MD5 = "38ddb2822551b1d281b57d610eb56986"
_BOUNDS: pd.DataFrame | None = None


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retention_time_seconds(spectrum: dict) -> float | None:
    """Return a pyteomics mzXML retention time in seconds."""

    value = spectrum.get("retentionTime")
    if value is None:
        return None
    unit = str(getattr(value, "unit_info", "")).lower()
    result = float(value)
    if "minute" in unit:
        result *= 60.0
    return result


def _prepare_positive_labels(workbook: Path) -> pd.DataFrame:
    frame = pd.read_excel(workbook)
    required = {"Component", "m.z", "tr", "Rating"}
    if not required <= set(frame.columns):
        raise ValueError(f"TARDIS label schema mismatch: {sorted(frame.columns)}")
    usable = frame[["Component", "m.z", "tr"]].notna().all(axis=1)
    if (frame["Rating"].notna() & ~usable).any():
        raise ValueError("a rated positive FAME target lacks component, m/z or retention time")
    frame = frame.loc[usable].copy()
    output = pd.DataFrame(
        {
            "feature": frame["Component"].astype("Int64").astype(str),
            "mz": pd.to_numeric(frame["m.z"], errors="coerce"),
            "expected_rt_seconds": pd.to_numeric(frame["tr"], errors="coerce"),
            "rating": frame["Rating"].astype("string"),
        }
    )
    if output[["mz", "expected_rt_seconds"]].isna().any().any():
        raise ValueError("positive FAME targets contain non-numeric m/z or retention time")
    if output["feature"].duplicated().any():
        raise ValueError("positive FAME target identifiers are not unique")
    ppm = 10.0e-6
    output["min_mz"] = output["mz"] * (1.0 - ppm)
    output["max_mz"] = output["mz"] * (1.0 + ppm)
    output["min_rt"] = np.maximum(0.0, output["expected_rt_seconds"] - 30.0)
    output["max_rt"] = output["expected_rt_seconds"] + 30.0
    return output


def _initialise_worker(bounds_csv: str) -> None:
    global _BOUNDS
    _BOUNDS = pd.read_csv(bounds_csv)


def _process_mzxml(path_string: str) -> dict[str, NDArray[np.float32]]:
    from pyteomics import mzxml

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
    with mzxml.MzXML(path_string, use_index=False) as reader:
        for spectrum in reader:
            if int(spectrum.get("msLevel", 1)) != 1:
                continue
            if spectrum.get("polarity") not in (None, "+"):
                raise ValueError(f"non-positive scan in {path_string}")
            retention_time = _retention_time_seconds(spectrum)
            if retention_time is None:
                continue
            active = np.flatnonzero((min_rt <= retention_time) & (retention_time <= max_rt))
            if active.size == 0:
                continue
            mz = np.asarray(spectrum.get("m/z array", []), dtype=float)
            intensity = np.asarray(spectrum.get("intensity array", []), dtype=float)
            if mz.size != intensity.size:
                continue
            left = np.searchsorted(mz, min_mz[active], side="left")
            right = np.searchsorted(mz, max_mz[active], side="right")
            for feature_index, start, stop in zip(active, left, right, strict=True):
                value = float(np.sum(intensity[start:stop])) if stop > start else 0.0
                if not np.isfinite(value):
                    value = 0.0
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


def extract_positive_fame(
    *,
    mzxml_dir: Path,
    labels_workbook: Path,
    archive: Path,
    output_dir: Path,
    workers: int,
    verify_archive: bool,
) -> None:
    if verify_archive:
        archive_md5 = _hash(archive, "md5")
        if archive_md5 != EXPECTED_ARCHIVE_MD5:
            raise ValueError(f"archive MD5 mismatch: {archive_md5}")
    else:
        archive_md5 = EXPECTED_ARCHIVE_MD5

    labels = _prepare_positive_labels(labels_workbook)
    files = sorted(mzxml_dir.glob("*QC*.mzXML"))
    if len(files) != 119:
        raise ValueError(f"expected 119 positive QC mzXML files, found {len(files)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds_csv = output_dir / "fame_positive_targets.csv"
    labels.to_csv(bounds_csv, index=False)

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
        initargs=(str(bounds_csv),),
    ) as executor:
        for file_index, result in enumerate(
            executor.map(_process_mzxml, (str(path) for path in files)), start=0
        ):
            for name, values in result.items():
                per_file[name][file_index] = values
            print(f"tardis_fame_positive: {file_index + 1}/{len(files)} QC files", flush=True)
    for values in per_file.values():
        values.flush()

    encoded_labels = np.full(labels.shape[0], -1, dtype=np.int8)
    encoded_labels[labels["rating"].eq("Bad").fillna(False).to_numpy()] = 0
    encoded_labels[labels["rating"].eq("Good").fillna(False).to_numpy()] = 1
    np.save(output_dir / "labels.npy", encoded_labels)
    (output_dir / "feature_names.txt").write_text(
        "\n".join(labels["feature"]) + "\n", encoding="utf-8"
    )
    (output_dir / "mzxml_files.txt").write_text(
        "\n".join(path.name for path in files) + "\n", encoding="utf-8"
    )
    _aggregate_per_file(output_dir, len(files), labels.shape[0])

    qscore_cube = np.load(output_dir / "per_file_qscore.npy", mmap_mode="r")
    per_target_available = np.mean(np.all(np.isfinite(qscore_cube), axis=2), axis=0)
    metadata = {
        "protocol": "hcrd-e6-v1-positive-availability-amendment",
        "dataset": "tardis_fame_positive",
        "archive_size_bytes": archive.stat().st_size,
        "archive_md5": archive_md5,
        "labels_workbook_sha256": _hash(labels_workbook),
        "mzxml_file_count": len(files),
        "feature_count": int(labels.shape[0]),
        "good_count": int(np.sum(encoded_labels == 1)),
        "bad_count": int(np.sum(encoded_labels == 0)),
        "excluded_count": int(np.sum(encoded_labels < 0)),
        "targets_with_any_valid_feature_bank": int(np.sum(per_target_available > 0.0)),
        "median_valid_qc_fraction": float(np.median(per_target_available)),
        "window_ppm": 10.0,
        "window_seconds": 60.0,
        "final_widths": {
            name: int(np.load(output_dir / f"{name}.npy", mmap_mode="r").shape[1])
            for name in FINAL_NAMES
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def _top_decile_enrichment(labels_bad: NDArray[np.int8], scores_bad: NDArray[np.float64]) -> float:
    count = max(1, math.ceil(0.1 * labels_bad.size))
    order = np.argsort(scores_bad, kind="mergesort")[::-1][:count]
    prevalence = float(np.mean(labels_bad))
    return float(np.mean(labels_bad[order]) / prevalence) if prevalence > 0.0 else math.nan


def evaluate_transfer(
    *,
    falkor_dir: Path,
    mesoscope_dir: Path,
    target_dir: Path,
    output_dir: Path,
    bootstrap: int,
) -> None:
    source_parts = [_load_dataset(path) for path in (falkor_dir, mesoscope_dir)]
    source_y = np.concatenate([part[0] for part in source_parts])
    source_x = {
        name: np.concatenate([part[1][name] for part in source_parts], axis=0)
        for name in FINAL_NAMES
    }
    target_labels_all = np.load(target_dir / "labels.npy")
    keep = target_labels_all >= 0
    target_y_good = target_labels_all[keep]
    target_y_bad = (1 - target_y_good).astype(np.int8)
    target_x_all = {
        name: np.asarray(np.load(target_dir / f"{name}.npy", mmap_mode="r"))
        for name in FINAL_NAMES
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_bad: dict[str, NDArray[np.float64]] = {}
    scores_bad_all: dict[str, NDArray[np.float64]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name in FINAL_NAMES:
        model = _model()
        model.fit(source_x[name], source_y)
        scores_bad_all[name] = 1.0 - model.predict_proba(target_x_all[name])[:, 1]
        scores_bad[name] = scores_bad_all[name][keep]
        metrics[name] = {
            "average_precision_bad": float(
                average_precision_score(target_y_bad, scores_bad[name])
            ),
            "roc_auc_bad": float(roc_auc_score(target_y_bad, scores_bad[name])),
            "top_decile_bad_enrichment": _top_decile_enrichment(
                target_y_bad, scores_bad[name]
            ),
        }
        joblib.dump(model, output_dir / f"pooled_source_model_{name}.joblib")

    comparisons = _bootstrap_comparisons(target_y_bad, scores_bad, bootstrap)
    target_table = pd.read_csv(target_dir / "fame_positive_targets.csv")
    ordinal = target_table["rating"].map({"Good": 0.0, "Ambiguous": 1.0, "Bad": 2.0})
    ordinal_keep = ordinal.notna().to_numpy()
    ordered_sensitivity = {
        name: {
            "n": int(np.sum(ordinal_keep)),
            "spearman": float(
                spearmanr(
                    ordinal.to_numpy(float, na_value=np.nan)[ordinal_keep],
                    score[ordinal_keep],
                ).statistic
            ),
        }
        for name, score in scores_bad_all.items()
    }

    main_q = comparisons["hcrd_8_q"]
    main_level = comparisons["hcrd_8_q_vs_hcrd_1_q"]
    domain = comparisons["hcrd_8_q_vs_domain_q"]
    result = {
        "protocol": "hcrd-e6-v1-positive-availability-amendment",
        "source": {
            "datasets": ["falkor", "mesoscope"],
            "unambiguous_count": int(source_y.size),
            "good_count": int(np.sum(source_y == 1)),
            "bad_count": int(np.sum(source_y == 0)),
            "fit": "one pooled StandardScaler + balanced L2 logistic model per frozen representation",
        },
        "target_metadata": json.loads((target_dir / "metadata.json").read_text()),
        "target_unambiguous_count": int(target_y_bad.size),
        "target_bad_count": int(np.sum(target_y_bad == 1)),
        "target_good_count": int(np.sum(target_y_bad == 0)),
        "metrics": metrics,
        "comparisons": comparisons,
        "ordered_label_sensitivity": ordered_sensitivity,
        "success_components": {
            "hcrd8_minus_qscore_ci_strictly_positive": main_q["bootstrap_95_ci"][0]
            > 0.0,
            "hcrd8_minus_hcrd1_ci_strictly_positive": main_level["bootstrap_95_ci"][0]
            > 0.0,
            "hcrd8_domain_noninferior_margin_minus_0_02": domain["bootstrap_95_ci"][0]
            > -0.02,
        },
        "prospective_primary_success": main_q["bootstrap_95_ci"][0] > 0.0
        and main_level["bootstrap_95_ci"][0] > 0.0,
        "bootstrap_replicates": bootstrap,
        "seed": E6_SEED,
        "runner_sha256": _hash(Path(__file__)),
    }
    (output_dir / "e6_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--mzxml-dir", type=Path, required=True)
    extract.add_argument("--labels-workbook", type=Path, required=True)
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    extract.add_argument("--skip-archive-md5", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--falkor-dir", type=Path, required=True)
    evaluate.add_argument("--mesoscope-dir", type=Path, required=True)
    evaluate.add_argument("--target-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    if args.command == "extract":
        extract_positive_fame(
            mzxml_dir=args.mzxml_dir,
            labels_workbook=args.labels_workbook,
            archive=args.archive,
            output_dir=args.output_dir,
            workers=args.workers,
            verify_archive=not args.skip_archive_md5,
        )
    else:
        evaluate_transfer(
            falkor_dir=args.falkor_dir,
            mesoscope_dir=args.mesoscope_dir,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            bootstrap=args.bootstrap,
        )


if __name__ == "__main__":
    main()
