"""Frozen comparison with an honest pointwise confidence-region projection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from shapecontrast import (
    build_shape_contrast_family,
    design_identified_transition_set,
    gaussian_bonferroni_shape_band,
    gaussian_pointwise_shape_projection,
    invert_s_shaped_inflection,
)

SAMPLE_SIZES = (500, 1000)
DESIGNS = ("uniform", "beta_4_8")
SIGNALS = (
    "paper_f1_cusp",
    "paper_f2_onset",
    "paper_f3_jump",
    "paper_f4_logistic",
)
TRIALS = 200
SEED = 20262211
SIGMA = 0.1
ALPHA = 0.05
SEPARATIONS = (1, 2, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths() -> dict[str, Path]:
    return {
        "driver": Path(__file__).resolve(),
        "protocol": PROJECT / "docs/sci_e38_matched_honest_baseline_protocol.md",
        "inference_module": PROJECT / "src/shapecontrast/inference.py",
        "projection_module": PROJECT / "src/shapecontrast/projection.py",
        "identified_set_module": PROJECT / "src/shapecontrast/identified_set.py",
    }


def _hashes() -> dict[str, str]:
    missing = [str(path) for path in _paths().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return {name: _sha256(path) for name, path in _paths().items()}


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
    }


def freeze(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_config.json"
    if frozen_path.exists() or (output_dir / "trial_scores.csv").exists():
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "frozen_before_evaluation"
        or payload.get("config") != _config()
        or payload.get("hashes") != _hashes()
    ):
        raise RuntimeError("configuration or code changed after the freeze")
    return payload


def _design(name: str, n: int) -> np.ndarray:
    probability = np.arange(1, n + 1, dtype=float) / (n + 1.0)
    if name == "uniform":
        return probability
    if name == "beta_4_8":
        return np.asarray(beta.ppf(probability, 4.0, 8.0), dtype=float)
    raise ValueError(name)


def _signal(name: str, x: np.ndarray) -> np.ndarray:
    if name == "paper_f1_cusp":
        left = 2.0 * (0.3 - np.sqrt(np.maximum(0.09 - x**2, 0.0)))
        right = 2.0 * (0.3 + np.sqrt(np.maximum(0.49 - (1.0 - x) ** 2, 0.0)))
        return np.where(x < 0.3, left, right)
    if name == "paper_f2_onset":
        return np.where(x < 0.3, 0.0, np.sin((x - 0.3) * np.pi / 1.4))
    if name == "paper_f3_jump":
        return x + (x >= 0.3).astype(float)
    if name == "paper_f4_logistic":
        return 4.0 / (1.0 + np.exp(-2.0 * (x - 0.3)))
    raise ValueError(name)


def _cell_seed(index: int) -> int:
    return int(np.random.SeedSequence(SEED).spawn(16)[index].generate_state(1)[0])


def _evaluate_cell(
    args: tuple[int, str, str, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    index, signal_name, design_name, n = args
    rng = np.random.default_rng(_cell_seed(index))
    x = _design(design_name, n)
    mean = _signal(signal_name, x)
    family = build_shape_contrast_family(x, separation_multipliers=SEPARATIONS)
    domain = (float(x[0]), float(x[-1]))
    target = design_identified_transition_set(x, mean, domain=domain)
    target_hull = target.hull
    if target_hull is None:
        raise RuntimeError(
            f"empty identified target in {signal_name}, {design_name}, {n}"
        )
    target_left, target_right = target_hull
    rows: list[dict[str, object]] = []
    for trial in range(TRIALS):
        observed = mean + rng.normal(0.0, SIGMA, size=n)
        sci_band = gaussian_bonferroni_shape_band(
            family, observed, noise_scale=SIGMA, alpha=ALPHA
        )
        sci = invert_s_shaped_inflection(family, sci_band, domain=domain)
        projection = gaussian_pointwise_shape_projection(
            x, observed, noise_scale=SIGMA, alpha=ALPHA, domain=domain
        )
        cell = f"{signal_name}__{design_name}__n{n}"
        for method, result in (("SCI", sci), ("PBP", projection)):
            generating_point_covered = (
                not result.empty and result.left <= 0.3 <= result.right
            )
            covered = (
                not result.empty
                and result.left <= target_left
                and target_right <= result.right
            )
            rows.append(
                {
                    "cell": cell,
                    "trial": trial,
                    "method": method,
                    "covered": covered,
                    "generating_point_covered": generating_point_covered,
                    "target_left": target_left,
                    "target_right": target_right,
                    "left": result.left,
                    "right": result.right,
                    "width": result.width,
                    "empty": result.empty,
                }
            )
    summary: dict[str, object] = {
        "cell": f"{signal_name}__{design_name}__n{n}",
        "signal": signal_name,
        "design": design_name,
        "n": n,
        "trials": TRIALS,
        "target_left": target_left,
        "target_right": target_right,
    }
    for method in ("SCI", "PBP"):
        selected = [row for row in rows if row["method"] == method]
        widths = np.asarray(
            [float(row["width"]) for row in selected if not bool(row["empty"])],
            dtype=float,
        )
        prefix = method.lower()
        summary[f"{prefix}_coverage"] = float(
            np.mean([bool(row["covered"]) for row in selected])
        )
        summary[f"{prefix}_median_width"] = float(np.median(widths))
        summary[f"{prefix}_mean_width"] = float(np.mean(widths))
        summary[f"{prefix}_empty_probability"] = float(
            np.mean([bool(row["empty"]) for row in selected])
        )
    return rows, summary


def evaluate(output_dir: Path, workers: int) -> None:
    frozen = _validate_freeze(output_dir)
    trial_path = output_dir / "trial_scores.csv"
    if trial_path.exists():
        raise RuntimeError("evaluation output already exists")
    cells = [
        (index, signal, design, n)
        for index, (signal, design, n) in enumerate(
            (s, d, n) for s in SIGNALS for d in DESIGNS for n in SAMPLE_SIZES
        )
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        evaluated = list(executor.map(_evaluate_cell, cells))
    trial_rows = [row for rows, _ in evaluated for row in rows]
    summary = [cell_summary for _, cell_summary in evaluated]

    with trial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(trial_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(trial_rows)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    width_reductions = [
        1.0 - float(row["sci_median_width"]) / float(row["pbp_median_width"])
        for row in summary
    ]
    gates = {
        "all_16_cells_retained": len(summary) == 16,
        "sci_coverage_at_least_0_90_every_cell": min(
            float(row["sci_coverage"]) for row in summary
        )
        >= 0.90,
        "pbp_coverage_at_least_0_90_every_cell": min(
            float(row["pbp_coverage"]) for row in summary
        )
        >= 0.90,
        "sci_never_more_than_0_01_wider": max(
            float(row["sci_median_width"]) - float(row["pbp_median_width"])
            for row in summary
        )
        <= 0.01,
        "at_least_8_cells_with_10pct_reduction": sum(
            reduction >= 0.10 for reduction in width_reductions
        )
        >= 8,
        "weak_logistic_cells_retained": sum(
            str(row["signal"]) == "paper_f4_logistic" for row in summary
        )
        == 4,
    }
    gates["all_pass"] = all(gates.values())

    report_path = output_dir / "report.md"
    lines = [
        "# E38r1 SCI versus a conservative pointwise-band baseline",
        "",
        f"All frozen checks passed: **{gates['all_pass']}**.",
        "",
        "PBP is our conservative discrete split relaxation, not an exact projection onto the SCI function class or an official implementation of Davies et al.",
        "",
        "| signal | design | n | SCI cov. | PBP cov. | SCI width | PBP width | reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    short = {
        "paper_f1_cusp": "cusp",
        "paper_f2_onset": "onset",
        "paper_f3_jump": "jump",
        "paper_f4_logistic": "logistic",
    }
    for row, reduction in zip(summary, width_reductions, strict=True):
        lines.append(
            "| {signal_name} | {design} | {n} | {sci_coverage:.3f} | "
            "{pbp_coverage:.3f} | {sci_median_width:.4f} | "
            "{pbp_median_width:.4f} | {reduction:.1%} |".format(
                signal_name=short[str(row["signal"])],
                design=row["design"],
                n=row["n"],
                sci_coverage=row["sci_coverage"],
                pbp_coverage=row["pbp_coverage"],
                sci_median_width=row["sci_median_width"],
                pbp_median_width=row["pbp_median_width"],
                reduction=reduction,
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "experiment": "E38r1 SCI versus pointwise-band projection",
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
            "workers": workers,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gates, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/sci/matched_honest_baseline_e38_r1",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.stage == "freeze":
        freeze(args.output_dir)
    else:
        evaluate(args.output_dir, args.workers)


if __name__ == "__main__":
    main()
