"""Frozen high-precision coverage audit for matrix-free SCI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from shapecontrast import build_shape_contrast_family  # noqa: E402

SAMPLE_SIZES = (500, 1000)
DESIGNS = ("uniform", "beta_4_8")
SIGNALS = (
    "paper_f1_cusp",
    "paper_f2_onset",
    "paper_f3_jump",
    "paper_f4_logistic",
)
TRIALS = 5000
SEED = 20262111
SIGMA = 0.1
ALPHA = 0.05
SEPARATIONS = (1, 2, 4)
BATCH_SIZE = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths() -> dict[str, Path]:
    return {
        "driver": Path(__file__).resolve(),
        "protocol": PROJECT / "docs/sci_e36_high_precision_coverage_protocol.md",
        "inference_module": PROJECT / "src/shapecontrast/inference.py",
    }


def _hashes() -> dict[str, str]:
    paths = _paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return {name: _sha256(path) for name, path in paths.items()}


def _config() -> dict[str, object]:
    return {
        "sample_sizes": list(SAMPLE_SIZES),
        "designs": list(DESIGNS),
        "signals": list(SIGNALS),
        "trials": TRIALS,
        "seed": SEED,
        "sigma": SIGMA,
        "alpha": ALPHA,
        "separation_multipliers": list(SEPARATIONS),
        "batch_size": BATCH_SIZE,
    }


def freeze(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = output_dir / "frozen_config.json"
    if frozen.exists() or (output_dir / "trial_scores.csv").exists():
        raise RuntimeError("this output directory was already frozen or evaluated")
    payload = {
        "status": "frozen_before_evaluation",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config(),
        "hashes": _hashes(),
    }
    frozen.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _validate_freeze(output_dir: Path) -> dict[str, object]:
    path = output_dir / "frozen_config.json"
    if not path.is_file():
        raise RuntimeError("run --stage freeze before evaluation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "frozen_before_evaluation"
        or payload.get("config") != _config()
        or payload.get("hashes") != _hashes()
    ):
        raise RuntimeError("configuration or code changed after the freeze")
    return payload


def _design_points(name: str, n: int) -> np.ndarray:
    probabilities = np.arange(1, n + 1, dtype=float) / (n + 1.0)
    if name == "uniform":
        return probabilities
    if name == "beta_4_8":
        return np.asarray(beta.ppf(probabilities, 4.0, 8.0), dtype=float)
    raise ValueError(name)


def _signal(name: str, x: np.ndarray) -> np.ndarray:
    if name == "paper_f1_cusp":
        left = 2.0 * (0.3 - np.sqrt(np.maximum(0.09 - x**2, 0.0)))
        right = 2.0 * (
            0.3 + np.sqrt(np.maximum(0.49 - (1.0 - x) ** 2, 0.0))
        )
        return np.where(x < 0.3, left, right)
    if name == "paper_f2_onset":
        return np.where(x < 0.3, 0.0, np.sin((x - 0.3) * np.pi / 1.4))
    if name == "paper_f3_jump":
        return x + (x >= 0.3).astype(float)
    if name == "paper_f4_logistic":
        return 4.0 / (1.0 + np.exp(-2.0 * (x - 0.3)))
    raise ValueError(name)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def evaluate(output_dir: Path) -> None:
    frozen = _validate_freeze(output_dir)
    trial_path = output_dir / "trial_scores.csv"
    if trial_path.exists():
        raise RuntimeError("evaluation output already exists")

    rng = np.random.default_rng(SEED)
    summary: list[dict[str, object]] = []
    trial_fields = [
        "cell",
        "trial",
        "covered",
        "left",
        "right",
        "width",
        "empty",
        "joint_contrast_covered",
        "positive_contrasts",
        "negative_contrasts",
    ]
    with trial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields, lineterminator="\n")
        writer.writeheader()
        for signal_name in SIGNALS:
            for design_name in DESIGNS:
                for n in SAMPLE_SIZES:
                    cell = f"{signal_name}__{design_name}__n{n}"
                    x = _design_points(design_name, n)
                    mean = _signal(signal_name, x)
                    family = build_shape_contrast_family(
                        x, separation_multipliers=SEPARATIONS
                    )
                    truth = family.means(mean)
                    critical = float(
                        norm.ppf(1.0 - ALPHA / (2.0 * family.contrast_count))
                    )
                    radius = critical * SIGMA * family.weight_l2
                    support_left = family.support_left
                    support_right = family.support_right

                    covered_count = 0
                    joint_count = 0
                    empty_count = 0
                    widths: list[float] = []
                    nontrivial_count = 0
                    for first in range(0, TRIALS, BATCH_SIZE):
                        batch = min(BATCH_SIZE, TRIALS - first)
                        observed = mean + rng.normal(0.0, SIGMA, size=(batch, n))
                        estimates = family.means_many(observed)
                        numerical_scale = np.maximum(
                            1.0, np.max(np.abs(estimates), axis=1)
                        )
                        numerical_scale = np.maximum(
                            numerical_scale, float(np.max(np.abs(radius)))
                        )
                        tolerance = 64.0 * np.finfo(float).eps * numerical_scale
                        positive = estimates - radius > tolerance[:, None]
                        negative = estimates + radius < -tolerance[:, None]
                        left = np.max(
                            np.where(positive, support_left[None, :], -np.inf),
                            axis=1,
                        )
                        right = np.min(
                            np.where(negative, support_right[None, :], np.inf),
                            axis=1,
                        )
                        left[~np.any(positive, axis=1)] = 0.0
                        right[~np.any(negative, axis=1)] = 1.0
                        empty = left > right
                        covered = (~empty) & (left <= 0.3) & (0.3 <= right)
                        joint = np.all(np.abs(estimates - truth) <= radius, axis=1)
                        width = np.where(empty, np.nan, right - left)

                        covered_count += int(np.sum(covered))
                        joint_count += int(np.sum(joint))
                        empty_count += int(np.sum(empty))
                        nontrivial_count += int(np.sum((~empty) & (width < 1.0 - 1e-12)))
                        widths.extend(width[np.isfinite(width)].tolist())
                        for offset in range(batch):
                            writer.writerow(
                                {
                                    "cell": cell,
                                    "trial": first + offset,
                                    "covered": bool(covered[offset]),
                                    "left": float(left[offset]),
                                    "right": float(right[offset]),
                                    "width": float(width[offset]),
                                    "empty": bool(empty[offset]),
                                    "joint_contrast_covered": bool(joint[offset]),
                                    "positive_contrasts": int(np.sum(positive[offset])),
                                    "negative_contrasts": int(np.sum(negative[offset])),
                                }
                            )

                    low, high = _wilson(covered_count, TRIALS)
                    joint_low, joint_high = _wilson(joint_count, TRIALS)
                    summary.append(
                        {
                            "cell": cell,
                            "signal": signal_name,
                            "design": design_name,
                            "n": n,
                            "trials": TRIALS,
                            "coverage": covered_count / TRIALS,
                            "coverage_wilson_low": low,
                            "coverage_wilson_high": high,
                            "joint_contrast_coverage": joint_count / TRIALS,
                            "joint_wilson_low": joint_low,
                            "joint_wilson_high": joint_high,
                            "mean_width": float(np.mean(widths)),
                            "median_width": float(np.median(widths)),
                            "nontrivial_probability": nontrivial_count / TRIALS,
                            "empty_probability": empty_count / TRIALS,
                        }
                    )
                    print(cell, summary[-1]["coverage"], summary[-1]["median_width"])

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    gates = {
        "all_16_cells_retained": len(summary) == 16,
        "coverage_at_least_0_94_every_cell": min(
            float(row["coverage"]) for row in summary
        )
        >= 0.94,
        "joint_coverage_at_least_0_94_every_cell": min(
            float(row["joint_contrast_coverage"]) for row in summary
        )
        >= 0.94,
        "zero_empty_sets": all(float(row["empty_probability"]) == 0.0 for row in summary),
        "weak_logistic_cells_retained": sum(
            str(row["signal"]) == "paper_f4_logistic" for row in summary
        )
        == 4,
    }
    gates["all_pass"] = all(gates.values())

    report_path = output_dir / "report.md"
    report = [
        "# E36 high-precision SCI coverage audit",
        "",
        f"All frozen checks passed: **{gates['all_pass']}**.",
        "",
        "| signal | design | n | coverage | 95% MC interval | median width |",
        "|---|---|---:|---:|---:|---:|",
    ]
    short_names = {
        "paper_f1_cusp": "cusp",
        "paper_f2_onset": "onset",
        "paper_f3_jump": "jump",
        "paper_f4_logistic": "logistic",
    }
    for row in summary:
        report.append(
            "| {name} | {design} | {n} | {coverage:.4f} | "
            "[{coverage_wilson_low:.4f}, {coverage_wilson_high:.4f}] | "
            "{median_width:.4f} |".format(
                name=short_names[str(row["signal"])], **row
            )
        )
    report.extend(
        [
            "",
            "These simulations measure Monte Carlo behaviour of the code. The",
            "finite-sample guarantee comes from the theorem, not from this table.",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "E36 high-precision SCI coverage audit",
        "status": "evaluated_after_freeze",
        "created_utc": datetime.now(UTC).isoformat(),
        "frozen": frozen,
        "gates": gates,
        "result_hashes": {
            "trial_scores": _sha256(trial_path),
            "summary": _sha256(summary_path),
            "report": _sha256(report_path),
        },
        "summary": summary,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gates, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/sci/high_precision_coverage_e36",
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output_dir)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
