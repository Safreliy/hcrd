#!/usr/bin/env python3
"""Frozen conditional external verification on the Pttime HILIC dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from experiments.run_lcms_eic_e1 import _weighted_ap, _weighted_ap_preparation
    from experiments.run_ms_metrics_e2 import (
        FINAL_NAMES,
        PER_FILE_WIDTHS,
        _aggregate_per_file,
        _model,
        _sha256,
        global_window_qscore,
    )
except ModuleNotFoundError:  # Direct ``python experiments/script.py`` execution.
    from run_lcms_eic_e1 import _weighted_ap, _weighted_ap_preparation
    from run_ms_metrics_e2 import (
        FINAL_NAMES,
        PER_FILE_WIDTHS,
        _aggregate_per_file,
        _model,
        _sha256,
        global_window_qscore,
    )
from hcrd.lcms import eic_feature_bank


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260825
PROTOCOL = ROOT / "docs/pttime_e5_protocol.md"
EXPECTED_LABEL_COUNTS = {"Good": 348, "Bad": 17, "Ambiguous": 35}
EXPECTED_POSITIVE_RAW_COUNT = 52

_BOUNDS: pd.DataFrame | None = None


def _initialise_worker(labels_path: str) -> None:
    global _BOUNDS
    _BOUNDS = pd.read_csv(labels_path)


def _duration_seconds(value: object) -> float | None:
    """Return an mzML/mzXML retention-time value in seconds."""

    if value is None:
        return None
    if isinstance(value, str):
        match = re.fullmatch(
            r"PT(?:(?P<h>[0-9.]+)H)?(?:(?P<m>[0-9.]+)M)?(?:(?P<s>[0-9.]+)S)?",
            value,
            flags=re.IGNORECASE,
        )
        if match:
            return (
                3600.0 * float(match.group("h") or 0.0)
                + 60.0 * float(match.group("m") or 0.0)
                + float(match.group("s") or 0.0)
            )
    result = float(value)
    unit = str(getattr(value, "unit_info", "")).lower()
    if "minute" in unit:
        result *= 60.0
    return result


def _retention_time_seconds(spectrum: dict) -> float | None:
    scans = spectrum.get("scanList", {}).get("scan", [])
    value = scans[0].get("scan start time") if scans else None
    if value is None:
        value = spectrum.get("scan start time")
    if value is None:
        value = spectrum.get("retentionTime")
    return _duration_seconds(value)


def _reader(path: str):
    if Path(path).suffix.lower() == ".mzxml":
        from pyteomics import mzxml

        return mzxml.MzXML(path, use_index=False)
    from pyteomics import mzml

    return mzml.MzML(path, use_index=False)


def _process_raw_file(path_string: str) -> dict[str, NDArray[np.float32]]:
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

    with _reader(path_string) as reader:
        for spectrum in reader:
            level = spectrum.get("ms level", spectrum.get("msLevel", 1))
            if int(level) != 1:
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
            order = None
            if mz.size > 1 and np.any(np.diff(mz) < 0.0):
                order = np.argsort(mz)
            if order is not None:
                mz = mz[order]
                intensity = intensity[order]
            left = np.searchsorted(mz, min_mz[active], side="left")
            right = np.searchsorted(mz, max_mz[active], side="right")
            for feature_index, start, stop in zip(active, left, right, strict=True):
                if stop <= start:
                    continue
                value = float(np.sum(intensity[start:stop]))
                if np.isfinite(value):
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


def _raw_files(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mzml", ".mzxml"}
    ]
    return sorted(files, key=lambda path: str(path).lower())


def _pttime_positive_raw_files(directory: Path) -> list[Path]:
    """Return the positive-mode files used by the published Pttime pipeline."""

    return [path for path in _raw_files(directory) if "_POS_" in path.name.upper()]


def extract_target(
    *,
    raw_dir: Path,
    labels_path: Path,
    source_archive: Path,
    repository: Path,
    output_dir: Path,
    workers: int,
) -> None:
    if not PROTOCOL.is_file():
        raise FileNotFoundError("frozen E5 protocol is missing")
    labels = pd.read_csv(labels_path)
    required = {"feature", "min_mz", "max_mz", "min_rt", "max_rt", "feat_class"}
    if not required <= set(labels.columns):
        raise ValueError("manual classification schema mismatch")
    observed_counts = labels["feat_class"].value_counts().to_dict()
    if observed_counts != EXPECTED_LABEL_COUNTS:
        raise RuntimeError(
            f"Pttime label counts changed: {observed_counts}; expected {EXPECTED_LABEL_COUNTS}"
        )
    files = _pttime_positive_raw_files(raw_dir)
    if len(files) != EXPECTED_POSITIVE_RAW_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_POSITIVE_RAW_COUNT} Pttime POS mzML/mzXML files "
            f"under {raw_dir}, found {len(files)}"
        )
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
            executor.map(_process_raw_file, (str(path) for path in files)), start=0
        ):
            for name, values in result.items():
                per_file[name][file_index] = values
            print(f"pttime: {file_index + 1}/{len(files)} raw files", flush=True)
    for values in per_file.values():
        values.flush()

    encoded = np.full(labels.shape[0], -1, dtype=np.int8)
    encoded[labels["feat_class"].eq("Bad").to_numpy()] = 0
    encoded[labels["feat_class"].eq("Good").to_numpy()] = 1
    np.save(output_dir / "labels.npy", encoded)
    (output_dir / "feature_names.txt").write_text(
        "\n".join(labels["feature"].astype(str)) + "\n", encoding="utf-8"
    )
    (output_dir / "raw_files.txt").write_text(
        "\n".join(str(path.relative_to(raw_dir)) for path in files) + "\n",
        encoding="utf-8",
    )
    _aggregate_per_file(output_dir, len(files), labels.shape[0])
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "protocol": "hcrd-e5-v1",
        "ion_mode": "positive",
        "raw_file_selection": "basename contains _POS_",
        "expected_raw_file_count": EXPECTED_POSITIVE_RAW_COUNT,
        "raw_file_count": len(files),
        "feature_count": int(labels.shape[0]),
        "good_count": int(np.sum(encoded == 1)),
        "bad_count": int(np.sum(encoded == 0)),
        "excluded_count": int(np.sum(encoded < 0)),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": _sha256(source_archive),
        "labels_sha256": _sha256(labels_path),
        "source_repository_commit": commit,
        "protocol_sha256": _sha256(PROTOCOL),
        "final_widths": {
            name: int(np.load(output_dir / f"{name}.npy", mmap_mode="r").shape[1])
            for name in FINAL_NAMES
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _load(directory: Path) -> tuple[NDArray[np.int8], dict[str, NDArray[np.float32]]]:
    labels = np.load(directory / "labels.npy")
    keep = labels >= 0
    return labels[keep], {
        name: np.asarray(np.load(directory / f"{name}.npy", mmap_mode="r")[keep])
        for name in FINAL_NAMES
    }


def _bad_metrics(
    labels_bad: NDArray[np.int8], scores_bad: NDArray[np.float64]
) -> dict[str, float | int]:
    order = np.argsort(-scores_bad, kind="stable")
    top_17 = order[:17]
    top_five_percent = order[: int(np.ceil(0.05 * labels_bad.size))]
    return {
        "average_precision_bad": float(average_precision_score(labels_bad, scores_bad)),
        "roc_auc_bad": float(roc_auc_score(labels_bad, scores_bad)),
        "bad_in_top_17": int(np.sum(labels_bad[top_17])),
        "bad_in_top_5_percent": int(np.sum(labels_bad[top_five_percent])),
        "top_5_percent_count": int(top_five_percent.size),
    }


def _two_sided_bootstrap_p(difference: NDArray[np.float64]) -> float:
    size = difference.size
    return float(
        min(
            1.0,
            2.0
            * min(
                (np.sum(difference <= 0.0) + 1.0) / (size + 1.0),
                (np.sum(difference >= 0.0) + 1.0) / (size + 1.0),
            ),
        )
    )


def _stratified_bootstrap(
    labels_bad: NDArray[np.int8],
    scores: dict[str, NDArray[np.float64]],
    replicates: int,
) -> dict[str, dict[str, object]]:
    preparations = {name: _weighted_ap_preparation(value) for name, value in scores.items()}
    bad_indices = np.flatnonzero(labels_bad == 1)
    good_indices = np.flatnonzero(labels_bad == 0)
    rng = np.random.default_rng(SEED)
    boot = {name: np.empty(replicates) for name in scores}
    weights = np.zeros(labels_bad.size, dtype=float)
    for replicate in range(replicates):
        weights.fill(0.0)
        weights[bad_indices] = rng.multinomial(
            bad_indices.size, np.full(bad_indices.size, 1.0 / bad_indices.size)
        )
        weights[good_indices] = rng.multinomial(
            good_indices.size, np.full(good_indices.size, 1.0 / good_indices.size)
        )
        for name in scores:
            order, starts = preparations[name]
            boot[name][replicate] = _weighted_ap(labels_bad, weights, order, starts)

    contrasts = {
        name: (name, "qscore")
        for name in FINAL_NAMES
        if name != "qscore"
    }
    contrasts.update(
        {
            "hcrd_8_q_vs_domain_q": ("hcrd_8_q", "domain_q"),
            "hcrd_8_q_vs_hcrd_1_q": ("hcrd_8_q", "hcrd_1_q"),
        }
    )
    output = {}
    for key, (left, right) in contrasts.items():
        difference = boot[left] - boot[right]
        output[key] = {
            "left": left,
            "right": right,
            "ap_bad_difference": float(
                average_precision_score(labels_bad, scores[left])
                - average_precision_score(labels_bad, scores[right])
            ),
            "stratified_bootstrap_95_ci": np.quantile(
                difference, [0.025, 0.975]
            ).tolist(),
            "two_sided_bootstrap_p": _two_sided_bootstrap_p(difference),
        }
    return output


def fit_evaluate(
    *,
    falkor_dir: Path,
    mesoscope_dir: Path,
    pttime_dir: Path,
    repository: Path,
    output_dir: Path,
    bootstrap: int,
) -> None:
    if not PROTOCOL.is_file():
        raise FileNotFoundError("frozen E5 protocol is missing")
    falkor_y, falkor_x = _load(falkor_dir)
    mesoscope_y, mesoscope_x = _load(mesoscope_dir)
    target_y, target_x = _load(pttime_dir)
    source_y = np.concatenate([falkor_y, mesoscope_y])
    source_x = {
        name: np.concatenate([falkor_x[name], mesoscope_x[name]], axis=0)
        for name in FINAL_NAMES
    }
    labels_bad = (target_y == 0).astype(np.int8)
    if (int(np.sum(labels_bad)), int(np.sum(1 - labels_bad))) != (17, 348):
        raise RuntimeError("unexpected evaluable Pttime class counts")

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_bad = {}
    metrics = {}
    predictions = pd.DataFrame(
        {
            "feature": (pttime_dir / "feature_names.txt").read_text(
                encoding="utf-8"
            ).splitlines()
        }
    )
    encoded_full = np.load(pttime_dir / "labels.npy")
    keep = encoded_full >= 0
    predictions = predictions.loc[keep].reset_index(drop=True)
    predictions["label"] = np.where(target_y == 1, "Good", "Bad")
    for name in FINAL_NAMES:
        model = _model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(source_x[name], source_y)
        p_good = model.predict_proba(target_x[name])[:, 1]
        scores_bad[name] = 1.0 - p_good
        metrics[name] = _bad_metrics(labels_bad, scores_bad[name])
        predictions[f"bad_score_{name}"] = scores_bad[name]
        joblib.dump(model, output_dir / f"model_pooled_to_pttime_{name}.joblib")

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    comparisons = _stratified_bootstrap(labels_bad, scores_bad, bootstrap)
    main = comparisons["hcrd_8_q"]
    multilevel = comparisons["hcrd_8_q_vs_hcrd_1_q"]
    success_components = {
        "hcrd8_exceeds_qscore": main["ap_bad_difference"] > 0.0,
        "positive_primary_ci_lower": main["stratified_bootstrap_95_ci"][0] > 0.0,
        "hcrd8_exceeds_hcrd1": multilevel["ap_bad_difference"] > 0.0,
    }
    dataset_dirs = {
        "falkor": falkor_dir,
        "mesoscope": mesoscope_dir,
        "pttime": pttime_dir,
    }
    result = {
        "protocol": "hcrd-e5-v1",
        "protocol_sha256": _sha256(PROTOCOL),
        "runner_sha256": _sha256(Path(__file__)),
        "source_repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_metadata": {
            name: json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            for name, directory in dataset_dirs.items()
        },
        "target_evaluable": {
            "total": int(target_y.size),
            "good": int(np.sum(target_y == 1)),
            "bad": int(np.sum(target_y == 0)),
            "excluded_ambiguous": 35,
            "selection": "source two-variable model probability > 0.9",
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "prospective_success": all(success_components.values()),
        "success_components": success_components,
        "bootstrap": {
            "replicates": bootstrap,
            "sampling": "paired class-stratified target-feature bootstrap",
            "seed": SEED,
        },
        "claim_boundary": (
            "Conditional reranking of the source-model-selected Pttime subset; "
            "not performance on the complete 7,781-feature population."
        ),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-target")
    extract.add_argument("--raw-dir", type=Path, required=True)
    extract.add_argument("--labels", type=Path, required=True)
    extract.add_argument("--source-archive", type=Path, required=True)
    extract.add_argument("--repository", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1))
    )
    evaluate = commands.add_parser("fit-evaluate")
    evaluate.add_argument("--falkor-dir", type=Path, required=True)
    evaluate.add_argument("--mesoscope-dir", type=Path, required=True)
    evaluate.add_argument("--pttime-dir", type=Path, required=True)
    evaluate.add_argument("--repository", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "extract-target":
        extract_target(
            raw_dir=args.raw_dir,
            labels_path=args.labels,
            source_archive=args.source_archive,
            repository=args.repository,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    else:
        fit_evaluate(
            falkor_dir=args.falkor_dir,
            mesoscope_dir=args.mesoscope_dir,
            pttime_dir=args.pttime_dir,
            repository=args.repository,
            output_dir=args.output_dir,
            bootstrap=args.bootstrap,
        )


if __name__ == "__main__":
    main()
