"""Frozen real-data evaluation of replicate-curve SCI on DNase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from shapecontrast import (  # noqa: E402
    build_shape_contrast_family,
    invert_s_shaped_inflection,
    replicated_t_shape_band,
)

ALPHA = 0.05
SEPARATIONS = (1, 2)
EXPECTED_DATA_SHA256 = (
    "d9af548405c6772dfbc7caf9998544f27fc133b7dbaa4340ae198787da98d5f4"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths() -> dict[str, Path]:
    return {
        "driver": Path(__file__).resolve(),
        "protocol": PROJECT / "docs/sci_e37_dnase_replicate_protocol.md",
        "inference_module": PROJECT / "src/shapecontrast/inference.py",
        "replicate_module": PROJECT / "src/shapecontrast/replicated.py",
        "data": PROJECT / "data/external/dnase/DNase.csv",
    }


def _hashes() -> dict[str, str]:
    missing = [str(path) for path in _paths().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    hashes = {name: _sha256(path) for name, path in _paths().items()}
    if hashes["data"] != EXPECTED_DATA_SHA256:
        raise RuntimeError("DNase data do not match the frozen source checksum")
    return hashes


def _config() -> dict[str, object]:
    return {
        "alpha": ALPHA,
        "separation_multipliers": list(SEPARATIONS),
        "design_transform": "log2(concentration)",
        "sampling_unit": "assay run",
        "technical_duplicate_rule": "mean within run and concentration",
        "expected_data_sha256": EXPECTED_DATA_SHA256,
    }


def freeze(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_config.json"
    if frozen_path.exists() or (output_dir / "result.json").exists():
        raise RuntimeError("this output directory was already frozen or evaluated")
    payload = {
        "status": "frozen_before_evaluation",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config(),
        "hashes": _hashes(),
    }
    frozen_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _validate_freeze(output_dir: Path) -> dict[str, object]:
    path = output_dir / "frozen_config.json"
    if not path.is_file():
        raise RuntimeError("run --stage freeze before evaluation")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if (
        frozen.get("status") != "frozen_before_evaluation"
        or frozen.get("config") != _config()
        or frozen.get("hashes") != _hashes()
    ):
        raise RuntimeError("configuration, code, or data changed after the freeze")
    return frozen


def _load_curves() -> tuple[np.ndarray, np.ndarray, list[int]]:
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    with _paths()["data"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(int(row["Run"]), float(row["conc"]))].append(
                float(row["density"])
            )
    runs = sorted({key[0] for key in grouped})
    concentrations = np.asarray(sorted({key[1] for key in grouped}), dtype=float)
    if len(runs) != 11 or concentrations.size != 8:
        raise RuntimeError("unexpected DNase run or concentration count")
    if any(len(grouped[(run, concentration)]) != 2 for run in runs for concentration in concentrations):
        raise RuntimeError("each run-concentration cell must contain two duplicates")
    curves = np.asarray(
        [
            [np.mean(grouped[(run, concentration)]) for concentration in concentrations]
            for run in runs
        ],
        dtype=float,
    )
    return concentrations, curves, runs


def _logistic(x: np.ndarray, bottom: float, top: float, slope: float, midpoint: float) -> np.ndarray:
    return bottom + (top - bottom) / (1.0 + np.exp(-slope * (x - midpoint)))


def evaluate(output_dir: Path) -> None:
    frozen = _validate_freeze(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError("evaluation output already exists")

    concentrations, curves, runs = _load_curves()
    x = np.log2(concentrations)
    mean_curve = np.mean(curves, axis=0)
    family = build_shape_contrast_family(
        x, separation_multipliers=SEPARATIONS
    )
    band = replicated_t_shape_band(family, curves, alpha=ALPHA)
    confidence = invert_s_shaped_inflection(
        family, band, domain=(float(x[0]), float(x[-1]))
    )

    fitted, _ = curve_fit(
        _logistic,
        x,
        mean_curve,
        p0=(float(mean_curve.min()), float(mean_curve.max()), 1.0, float(np.median(x))),
        maxfev=100000,
    )
    logistic_midpoint = float(fitted[3])
    logistic_concentration = float(2.0**logistic_midpoint)

    replicate_path = output_dir / "replicate_curves.csv"
    with replicate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run", *[f"conc_{value:g}" for value in concentrations]])
        for run, curve in zip(runs, curves, strict=True):
            writer.writerow([run, *curve])

    mean_path = output_dir / "mean_curve.csv"
    with mean_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["concentration", "log2_concentration", "mean_density", "sd_across_runs"])
        for concentration, design, mean, sd in zip(
            concentrations, x, mean_curve, np.std(curves, axis=0, ddof=1), strict=True
        ):
            writer.writerow([concentration, design, mean, sd])

    contrast_path = output_dir / "contrast_table.csv"
    signs = band.certified_signs
    with contrast_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "contrast",
                "block_size",
                "separation",
                "support_left_log2",
                "support_right_log2",
                "estimate",
                "radius",
                "lower",
                "upper",
                "certified_sign",
            ]
        )
        for index in range(family.contrast_count):
            writer.writerow(
                [
                    index,
                    family.block_size[index],
                    family.separation[index],
                    family.support_left[index],
                    family.support_right[index],
                    band.estimate[index],
                    band.radius[index],
                    band.lower[index],
                    band.upper[index],
                    signs[index],
                ]
            )

    left_concentration = None if confidence.empty else float(2.0**confidence.left)
    right_concentration = None if confidence.empty else float(2.0**confidence.right)
    result = {
        "replicate_count": int(curves.shape[0]),
        "design_point_count": int(curves.shape[1]),
        "raw_measurement_count": 176,
        "contrast_count": family.contrast_count,
        "critical_value": band.critical_value,
        "certified_positive_contrasts": confidence.positive_contrast_count,
        "certified_negative_contrasts": confidence.negative_contrast_count,
        "confidence_level": 1.0 - ALPHA,
        "sci_empty": confidence.empty,
        "sci_log2_interval": None if confidence.empty else [confidence.left, confidence.right],
        "sci_concentration_interval": None if confidence.empty else [left_concentration, right_concentration],
        "logistic_midpoint_log2": logistic_midpoint,
        "logistic_midpoint_concentration": logistic_concentration,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    grid = np.linspace(x[0], x[-1], 400)
    figure_path = output_dir / "dnase_replicate_sci.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for curve in curves:
        ax.plot(concentrations, curve, color="#7f8c8d", alpha=0.28, linewidth=0.8)
    ax.scatter(concentrations, mean_curve, color="#111111", s=28, label="mean across runs", zorder=3)
    ax.plot(2.0**grid, _logistic(grid, *fitted), color="#2667ff", linewidth=2.0, label="logistic point fit")
    if not confidence.empty:
        ax.axvspan(left_concentration, right_concentration, color="#f4a261", alpha=0.22, label="95% replicate SCI")
    ax.axvline(logistic_concentration, color="#2667ff", linestyle="--", linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("DNase concentration")
    ax.set_ylabel("Optical density")
    ax.set_title("DNase assay: transition uncertainty from 11 runs")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = output_dir / "report.md"
    if confidence.empty:
        interval_text = "empty"
        interpretation = (
            "The data reject the assumed convex-to-concave shape at the "
            "simultaneous confidence level."
        )
    else:
        interval_text = f"[{left_concentration:.6g}, {right_concentration:.6g}]"
        interpretation = (
            "It contains a lower bound but reaches the largest observed "
            f"concentration. Thus the data support that the transition does "
            f"not occur below `{left_concentration:.6g}`, but they do not give "
            "a reliable upper bound within the observed range."
        )
    report = f"""# E37 replicate-curve SCI on the DNase assay

The analysis used 11 assay runs, eight concentrations, and all 176 raw
measurements. Technical duplicates were averaged inside each run. The run,
not the individual optical-density reading, was the independent sampling unit.

The 95% SCI set for the mean curve's convex-to-concave transition is
`{interval_text}` in concentration units. {interpretation}

A four-parameter logistic fit places its descriptive transition at
`{logistic_concentration:.6g}`. This point estimate lies inside the SCI set, but
it does not have the same finite-sample guarantee.

The Student construction allows arbitrary dependence and unequal variance
between concentrations inside an assay run. Its exact coverage statement
assumes that the 11 run-level curves are independent Gaussian replicates with
a common mean and covariance.
"""
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "experiment": "E37 exact replicate-curve SCI on DNase",
        "status": "evaluated_after_freeze",
        "created_utc": datetime.now(UTC).isoformat(),
        "frozen": frozen,
        "result": result,
        "result_hashes": {
            "replicate_curves": _sha256(replicate_path),
            "mean_curve": _sha256(mean_path),
            "contrast_table": _sha256(contrast_path),
            "result": _sha256(result_path),
            "figure": _sha256(figure_path),
            "report": _sha256(report_path),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/sci/dnase_replicate_e37",
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output_dir)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
