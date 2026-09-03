"""Frozen E34 confirmation of honest unknown-scale shape inversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_unknown_scale_shape_inversion_e34 as development  # noqa: E402
from hcrd.noise_scale_confidence import gaussian_block_upper_scale  # noqa: E402
from hcrd.shape_inflection_confidence import (  # noqa: E402
    ShapeContrastBand,
    build_shape_contrast_family,
    invert_s_shaped_inflection,
)

SAMPLE_SIZES = (500, 1000)
DESIGNS = development.DESIGNS
SIGNALS = development.SIGNALS
TRIALS = 200
SEED = 20261911
SIGMA = development.SIGMA
ALPHA = development.ALPHA
SCALE_FAILURE = development.SCALE_FAILURE
CONTRAST_ALPHA = development.CONTRAST_ALPHA
SEPARATIONS = development.SEPARATIONS
BLOCK_LENGTH = 2
DOMAIN = development.DOMAIN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths() -> dict[str, Path]:
    return {
        "driver": Path(__file__).resolve(),
        "development_driver": Path(development.__file__).resolve(),
        "protocol": PROJECT / "docs/hct_e34_unknown_scale_protocol.md",
        "scale_module": PROJECT / "src/hcrd/noise_scale_confidence.py",
        "shape_module": PROJECT / "src/hcrd/shape_inflection_confidence.py",
        "development_manifest": PROJECT
        / "results/hct/unknown_scale_shape_inversion_e34_development/manifest.json",
    }


def _hashes() -> dict[str, str]:
    paths = _paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    return {name: _sha256(path) for name, path in paths.items()}


def _config() -> dict[str, object]:
    return json.loads(
        json.dumps(
            {
                "sample_sizes": SAMPLE_SIZES,
                "designs": DESIGNS,
                "signals": SIGNALS,
                "trials": TRIALS,
                "seed": SEED,
                "sigma": SIGMA,
                "alpha": ALPHA,
                "scale_failure": SCALE_FAILURE,
                "contrast_alpha": CONTRAST_ALPHA,
                "separations": SEPARATIONS,
                "block_length": BLOCK_LENGTH,
            }
        )
    )


def freeze(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "frozen_config.json"
    if freeze_path.exists() or (output_dir / "trial_scores.csv").exists():
        raise RuntimeError("confirmation was already frozen or executed")
    payload = {
        "status": "frozen_before_confirmation_execution",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config(),
        "hashes": _hashes(),
    }
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _validate_freeze(output_dir: Path) -> dict[str, object]:
    path = output_dir / "frozen_config.json"
    if not path.exists():
        raise RuntimeError("confirmation is locked until --stage freeze")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_confirmation_execution":
        raise RuntimeError("invalid freeze status")
    if payload.get("config") != _config() or payload.get("hashes") != _hashes():
        raise RuntimeError("configuration or code changed after freeze")
    return payload


def _band(family, estimate: np.ndarray, scale: float, alpha: float) -> ShapeContrastBand:
    critical = float(norm.ppf(1.0 - alpha / (2.0 * family.contrast_count)))
    radius = critical * scale * family.weight_l2
    return ShapeContrastBand(
        estimate=estimate,
        lower=estimate - radius,
        upper=estimate + radius,
        radius=radius,
        critical_value=critical,
        alpha=alpha,
        noise_scale=scale,
    )


def _scores() -> list[dict[str, object]]:
    specifications = [
        (n, design, signal)
        for n in SAMPLE_SIZES
        for design in DESIGNS
        for signal in SIGNALS
    ]
    children = np.random.SeedSequence(SEED).spawn(len(specifications))
    rows: list[dict[str, object]] = []
    families = {}
    for (n, design, signal), child in zip(specifications, children, strict=True):
        x = development.design_points(design, n)
        if (n, design) not in families:
            families[(n, design)] = build_shape_contrast_family(
                x, separation_multipliers=SEPARATIONS
            )
        family = families[(n, design)]
        truth, target = development.signal_values(signal, x)
        true_contrasts = family.means(truth)
        responses = truth + np.random.default_rng(child).normal(
            0.0, SIGMA, size=(TRIALS, n)
        )
        for trial, response in enumerate(responses):
            estimate = family.means(response)
            scale_bound = gaussian_block_upper_scale(
                response,
                int(np.ceil(n / BLOCK_LENGTH)),
                failure_probability=SCALE_FAILURE,
            )
            methods = (
                ("known_sigma", SIGMA, ALPHA, True),
                (
                    "unknown_block_2",
                    scale_bound.upper_scale,
                    CONTRAST_ALPHA,
                    scale_bound.upper_scale >= SIGMA - 1e-12,
                ),
            )
            for method, scale, band_alpha, scale_covers in methods:
                band = _band(family, estimate, scale, band_alpha)
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
                        "nontrivial": not confidence.empty
                        and confidence.width < 1.0 - 1e-12,
                        "empty": confidence.empty,
                        "joint_contrast_coverage": joint,
                        "unexplained_empty": confidence.empty and joint,
                        "scale_upper": scale,
                        "scale_ratio": scale / SIGMA,
                        "scale_covers": scale_covers,
                    }
                )
    return rows


def _gates(summary: list[dict[str, object]]) -> tuple[dict[str, bool], float]:
    unknown = [row for row in summary if row["method"] == "unknown_block_2"]
    known = {row["cell"]: row for row in summary if row["method"] == "known_sigma"}
    informative = [
        row
        for row in unknown
        if int(row["n"]) == 1000
        and row["signal"]
        in ("paper_f1_cusp", "paper_f2_onset", "paper_f3_jump")
    ]
    limits = {
        "paper_f1_cusp": 0.20,
        "paper_f2_onset": 0.90,
        "paper_f3_jump": 0.025,
    }
    inflation = float(
        np.mean(
            [
                float(row["mean_width"])
                / max(float(known[row["cell"]]["mean_width"]), 1e-15)
                for row in informative
            ]
        )
    )
    gates = {
        "coverage_at_least_0_93_every_cell": all(
            float(row["coverage"]) >= 0.93 for row in unknown
        ),
        "scale_coverage_at_least_0_95_every_cell": all(
            float(row["scale_coverage"]) >= 0.95 for row in unknown
        ),
        "zero_unexplained_empty_sets": all(
            int(row["unexplained_empty_count"]) == 0 for row in unknown
        ),
        "large_n_width_limits": len(informative) == 6
        and all(
            float(row["median_width"]) < limits[str(row["signal"])]
            for row in informative
        ),
        "mean_informative_width_inflation_below_1_5": inflation < 1.5,
        "weak_logistic_cells_retained": len(
            [row for row in unknown if row["signal"] == "paper_f4_logistic"]
        )
        == 4,
    }
    return gates, inflation


def _figure(summary: list[dict[str, object]], path: Path) -> None:
    unknown = [
        row
        for row in summary
        if row["method"] == "unknown_block_2"
        and str(row["signal"]).startswith("paper_")
    ]
    known = {row["cell"]: row for row in summary if row["method"] == "known_sigma"}
    labels = [
        f"{str(row['signal']).replace('paper_', '')}\n{row['design']}, n={row['n']}"
        for row in unknown
    ]
    positions = np.arange(len(unknown))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].bar(
        positions - width / 2,
        [known[row["cell"]]["coverage"] for row in unknown],
        width,
        label="known sigma",
    )
    axes[0].bar(
        positions + width / 2,
        [row["coverage"] for row in unknown],
        width,
        label="unknown sigma",
    )
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.85, 1.01)
    axes[0].set_ylabel("coverage")
    axes[0].legend()
    axes[1].bar(
        positions - width / 2,
        [known[row["cell"]]["median_width"] for row in unknown],
        width,
        label="known sigma",
    )
    axes[1].bar(
        positions + width / 2,
        [row["median_width"] for row in unknown],
        width,
        label="unknown sigma",
    )
    axes[1].set_ylabel("median width")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _report(summary, gates: dict[str, bool], inflation: float) -> Path:
    unknown = [row for row in summary if row["method"] == "unknown_block_2"]
    lines = [
        "# E34: honest unknown-scale confirmation",
        "",
        "Fresh-seed confirmation using a finite-sample upper Gaussian noise-scale bound.",
        "",
        "| signal | design | n | coverage | median width | scale coverage | mean scale ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in unknown:
        lines.append(
            f"| {str(row['signal']).replace('paper_', '').replace('control_', '')} | "
            f"{row['design']} | {row['n']} | {float(row['coverage']):.3f} | "
            f"{float(row['median_width']):.4f} | {float(row['scale_coverage']):.3f} | "
            f"{float(row['mean_scale_ratio']):.3f} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`"
        for name, value in gates.items()
    )
    lines.extend(
        [
            "",
            f"Mean informative width inflation versus known sigma: `{inflation:.3f}`.",
            "",
            "The guarantee remains specific to independent homoskedastic Gaussian errors. "
            "Projection lack of fit can widen intervals but cannot invalidate coverage.",
        ]
    )
    path = PROJECT / "reports/hct/e34_unknown_scale_confirmation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate(output_dir: Path) -> None:
    frozen = _validate_freeze(output_dir)
    if (output_dir / "trial_scores.csv").exists():
        raise RuntimeError("confirmation already executed; refusing overwrite")
    rows = _scores()
    summary = development.summarize(rows)
    gates, inflation = _gates(summary)
    development._write_csv(output_dir / "trial_scores.csv", rows)
    development._write_csv(output_dir / "summary.csv", summary)
    _figure(summary, output_dir / "comparison.png")
    report = _report(summary, gates, inflation)
    manifest = {
        "experiment": "E34 unknown-scale shape inversion confirmation",
        "status": "confirmation_executed_after_freeze",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config(),
        "freeze_sha256": _sha256(output_dir / "frozen_config.json"),
        "frozen": frozen,
        "gates": {"all_pass": all(gates.values()), **gates},
        "mean_informative_width_inflation": inflation,
        "hashes": {
            **_hashes(),
            "trial_scores": _sha256(output_dir / "trial_scores.csv"),
            "summary": _sha256(output_dir / "summary.csv"),
            "report": _sha256(report),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output_dir), "gates": manifest["gates"], "inflation": inflation},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/hct/unknown_scale_shape_inversion_e34_confirmation",
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output_dir)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
