"""E34 development of honest unknown-scale shape-contrast inversion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.noise_scale_confidence import (  # noqa: E402
    gaussian_block_upper_scale,
)
from hcrd.shape_inflection_confidence import (  # noqa: E402
    ShapeContrastBand,
    build_shape_contrast_family,
    invert_s_shaped_inflection,
)


ALPHA = 0.05
SCALE_FAILURE = 0.01
CONTRAST_ALPHA = ALPHA - SCALE_FAILURE
SIGMA = 0.1
DOMAIN = (0.0, 1.0)
SEPARATIONS = (1, 2, 4)
BLOCK_LENGTHS = (2, 4, 8, 16)
SAMPLE_SIZES = (100, 200, 500, 1000)
DESIGNS = ("uniform", "beta_4_8")
SIGNALS = (
    "paper_f1_cusp",
    "paper_f2_onset",
    "paper_f3_jump",
    "paper_f4_logistic",
    "control_affine",
    "control_convex",
    "control_concave",
)
TRIALS = 80
SEED = 20261811


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def design_points(name: str, n: int) -> np.ndarray:
    probabilities = np.arange(1, n + 1, dtype=float) / (n + 1.0)
    if name == "uniform":
        return probabilities
    if name == "beta_4_8":
        return np.asarray(beta.ppf(probabilities, 4.0, 8.0), dtype=float)
    raise ValueError(name)


def signal_values(name: str, x: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    if name == "paper_f1_cusp":
        left = 2.0 * (0.3 - np.sqrt(np.maximum(0.09 - x**2, 0.0)))
        right = 2.0 * (0.3 + np.sqrt(np.maximum(0.49 - (1.0 - x) ** 2, 0.0)))
        return np.where(x < 0.3, left, right), (0.3, 0.3)
    if name == "paper_f2_onset":
        return np.where(x < 0.3, 0.0, np.sin((x - 0.3) * np.pi / 1.4)), (0.3, 0.3)
    if name == "paper_f3_jump":
        return x + (x >= 0.3).astype(float), (0.3, 0.3)
    if name == "paper_f4_logistic":
        return 4.0 / (1.0 + np.exp(-2.0 * (x - 0.3))), (0.3, 0.3)
    if name == "control_affine":
        return 2.0 * x, (0.0, 1.0)
    if name == "control_convex":
        return 2.0 * x**2, (1.0, 1.0)
    if name == "control_concave":
        return 2.0 * (2.0 * x - x**2), (0.0, 0.0)
    raise ValueError(name)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2)) / denominator
    return float(center - radius), float(center + radius)


def run() -> list[dict[str, object]]:
    specifications = [
        (n, design, signal)
        for n in SAMPLE_SIZES
        for design in DESIGNS
        for signal in SIGNALS
    ]
    children = np.random.SeedSequence(SEED).spawn(len(specifications))
    rows: list[dict[str, object]] = []
    family_cache = {}
    for (n, design, signal), child in zip(specifications, children, strict=True):
        x = design_points(design, n)
        family_key = (n, design)
        if family_key not in family_cache:
            family_cache[family_key] = build_shape_contrast_family(
                x, separation_multipliers=SEPARATIONS
            )
        family = family_cache[family_key]
        truth, target = signal_values(signal, x)
        true_contrasts = family.means(truth)
        rng = np.random.default_rng(child)
        responses = truth + rng.normal(0.0, SIGMA, size=(TRIALS, n))
        for trial, response in enumerate(responses):
            estimate = family.means(response)
            methods: list[tuple[str, float, float, bool]] = [
                ("known_sigma", SIGMA, ALPHA, True)
            ]
            for length in BLOCK_LENGTHS:
                bound = gaussian_block_upper_scale(
                    response,
                    int(np.ceil(n / length)),
                    failure_probability=SCALE_FAILURE,
                )
                methods.append(
                    (
                        f"unknown_block_{length}",
                        bound.upper_scale,
                        CONTRAST_ALPHA,
                        bound.upper_scale >= SIGMA - 1e-12,
                    )
                )
            for method, scale, contrast_alpha, scale_covers in methods:
                critical = float(
                    norm.ppf(
                        1.0 - contrast_alpha / (2.0 * family.contrast_count)
                    )
                )
                radius = critical * scale * family.weight_l2
                band = ShapeContrastBand(
                    estimate=estimate,
                    lower=estimate - radius,
                    upper=estimate + radius,
                    radius=radius,
                    critical_value=critical,
                    alpha=contrast_alpha,
                    noise_scale=scale,
                )
                confidence = invert_s_shaped_inflection(family, band, domain=DOMAIN)
                joint = bool(
                    np.all(true_contrasts >= band.lower - 1e-12)
                    and np.all(true_contrasts <= band.upper + 1e-12)
                )
                covers = bool(
                    not confidence.empty
                    and confidence.left <= target[0] + 1e-12
                    and confidence.right >= target[1] - 1e-12
                )
                rows.append(
                    {
                        "cell": f"{signal}__{design}__n{n}",
                        "signal": signal,
                        "design": design,
                        "n": n,
                        "trial": trial,
                        "method": method,
                        "target_left": target[0],
                        "target_right": target[1],
                        "confidence_left": confidence.left,
                        "confidence_right": confidence.right,
                        "confidence_width": confidence.width,
                        "covers_target": covers,
                        "nontrivial": not confidence.empty and confidence.width < 1.0 - 1e-12,
                        "empty": confidence.empty,
                        "joint_contrast_coverage": joint,
                        "unexplained_empty": confidence.empty and joint,
                        "scale_upper": scale,
                        "scale_ratio": scale / SIGMA,
                        "scale_covers": scale_covers,
                    }
                )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(str(row["cell"]), str(row["method"])) for row in rows})
    for cell, method in keys:
        selected = [row for row in rows if row["cell"] == cell and row["method"] == method]
        covered = sum(bool(row["covers_target"]) for row in selected)
        interval = _wilson(covered, len(selected))
        widths = np.asarray([row["confidence_width"] for row in selected], dtype=float)
        output.append(
            {
                "cell": cell,
                "signal": selected[0]["signal"],
                "design": selected[0]["design"],
                "n": selected[0]["n"],
                "method": method,
                "trials": len(selected),
                "coverage": covered / len(selected),
                "coverage_wilson_low": interval[0],
                "coverage_wilson_high": interval[1],
                "mean_width": float(np.nanmean(widths)),
                "median_width": float(np.nanmedian(widths)),
                "nontrivial_probability": float(np.mean([row["nontrivial"] for row in selected])),
                "empty_probability": float(np.mean([row["empty"] for row in selected])),
                "joint_contrast_coverage": float(np.mean([row["joint_contrast_coverage"] for row in selected])),
                "unexplained_empty_count": sum(bool(row["unexplained_empty"]) for row in selected),
                "mean_scale_ratio": float(np.mean([row["scale_ratio"] for row in selected])),
                "median_scale_ratio": float(np.median([row["scale_ratio"] for row in selected])),
                "scale_coverage": float(np.mean([row["scale_covers"] for row in selected])),
            }
        )
    return output


def select(summary: list[dict[str, object]]) -> dict[str, object]:
    candidates = []
    known_width = {
        row["cell"]: float(row["mean_width"])
        for row in summary
        if row["method"] == "known_sigma"
    }
    for length in BLOCK_LENGTHS:
        method = f"unknown_block_{length}"
        selected = [row for row in summary if row["method"] == method]
        valid = all(
            float(row["coverage"]) >= 0.925
            and int(row["unexplained_empty_count"]) == 0
            for row in selected
        )
        inflation = float(
            np.mean(
                [
                    float(row["mean_width"]) / max(known_width[row["cell"]], 1e-15)
                    for row in selected
                ]
            )
        )
        candidates.append(
            {
                "block_length": length,
                "development_gate": valid,
                "mean_width_inflation": inflation,
                "minimum_coverage": min(float(row["coverage"]) for row in selected),
                "minimum_scale_coverage": min(float(row["scale_coverage"]) for row in selected),
                "mean_scale_ratio": float(np.mean([row["mean_scale_ratio"] for row in selected])),
            }
        )
    eligible = [row for row in candidates if row["development_gate"]]
    chosen = min(eligible, key=lambda row: (row["mean_width_inflation"], row["block_length"])) if eligible else None
    return {"candidates": candidates, "chosen": chosen}


def make_figure(summary: list[dict[str, object]], path: Path) -> None:
    paper = [row for row in summary if str(row["signal"]).startswith("paper_")]
    methods = ["known_sigma"] + [f"unknown_block_{length}" for length in BLOCK_LENGTHS]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for method in methods:
        selected = [row for row in paper if row["method"] == method]
        axes[0].scatter(
            [row["n"] for row in selected],
            [row["coverage"] for row in selected],
            s=15,
            alpha=0.65,
            label=method,
        )
        axes[1].scatter(
            [row["n"] for row in selected],
            [row["median_width"] for row in selected],
            s=15,
            alpha=0.65,
            label=method,
        )
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.85, 1.01)
    axes[0].set_title("Paper-signal coverage")
    axes[0].set_xlabel("n")
    axes[1].set_xscale("log")
    axes[1].set_title("Paper-signal median widths")
    axes[1].set_xlabel("n")
    axes[0].legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/hct/unknown_scale_shape_inversion_e34_development",
    )
    args = parser.parse_args()
    rows = run()
    summary = summarize(rows)
    selection = select(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "trial_scores.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    make_figure(summary, args.output_dir / "comparison.png")
    script = Path(__file__).resolve()
    protocol = PROJECT / "docs/hct_e34_unknown_scale_protocol.md"
    manifest = {
        "experiment": "E34 honest unknown Gaussian scale development",
        "status": "development",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": {
            "sample_sizes": SAMPLE_SIZES,
            "designs": DESIGNS,
            "signals": SIGNALS,
            "trials": TRIALS,
            "seed": SEED,
            "sigma": SIGMA,
            "alpha": ALPHA,
            "scale_failure": SCALE_FAILURE,
            "block_lengths": BLOCK_LENGTHS,
            "separations": SEPARATIONS,
        },
        "selection": selection,
        "hashes": {
            "driver": _sha256(script),
            "protocol": _sha256(protocol),
            "scale_module": _sha256(PROJECT / "src/hcrd/noise_scale_confidence.py"),
            "shape_module": _sha256(PROJECT / "src/hcrd/shape_inflection_confidence.py"),
            "trial_scores": _sha256(args.output_dir / "trial_scores.csv"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
