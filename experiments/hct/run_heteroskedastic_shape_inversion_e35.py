"""Frozen E35 confirmation and LIDAR sensitivity illustration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_unknown_scale_shape_inversion_e34 as signals  # noqa: E402
from hcrd.heteroskedastic_scale_confidence import (  # noqa: E402
    gaussian_heteroskedastic_upper_envelope,
)
from hcrd.shape_inflection_confidence import (  # noqa: E402
    ShapeContrastBand,
    build_shape_contrast_family,
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
VARIANCE_PROFILES = ("constant", "linear", "plume_peak")
TRIALS = 200
SEED = 20262011
AVERAGE_SIGMA = 0.1
ALPHA = 0.05
SCALE_FAILURE = 0.01
CONTRAST_ALPHA = ALPHA - SCALE_FAILURE
SEPARATIONS = (1, 2, 4)
DOMAIN = (0.0, 1.0)
LIDAR_KAPPAS = (1.0, 1.5, 2.0, 3.0, 4.0)
LIDAR_BOOTSTRAP_LOOPS = 2000
LIDAR_SEED = 20262012


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths() -> dict[str, Path]:
    return {
        "driver": Path(__file__).resolve(),
        "r_bridge": PROJECT / "experiments/hct/run_lidar_hybrid_e35.R",
        "protocol": PROJECT / "docs/hct_e35_heteroskedastic_protocol.md",
        "theory": PROJECT / "theory/hct/heteroskedastic_gaussian_extension.md",
        "heteroskedastic_module": PROJECT
        / "src/hcrd/heteroskedastic_scale_confidence.py",
        "shape_module": PROJECT / "src/hcrd/shape_inflection_confidence.py",
        "semipar_source": PROJECT / "third_party/SemiPar_1.0-4.2.tar.gz",
        "shapechange_source": PROJECT / "third_party/ShapeChange_1.5.tar.gz",
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
                "variance_profiles": VARIANCE_PROFILES,
                "trials": TRIALS,
                "seed": SEED,
                "average_sigma": AVERAGE_SIGMA,
                "alpha": ALPHA,
                "scale_failure": SCALE_FAILURE,
                "contrast_alpha": CONTRAST_ALPHA,
                "separations": SEPARATIONS,
                "lidar_kappas": LIDAR_KAPPAS,
                "lidar_bootstrap_loops": LIDAR_BOOTSTRAP_LOOPS,
                "lidar_seed": LIDAR_SEED,
            }
        )
    )


def freeze(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "frozen_config.json"
    if path.exists() or (output_dir / "trial_scores.csv").exists():
        raise RuntimeError("E35 was already frozen or executed")
    payload = {
        "status": "frozen_before_confirmation_execution",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config(),
        "hashes": _hashes(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _validate_freeze(output_dir: Path) -> dict[str, object]:
    path = output_dir / "frozen_config.json"
    if not path.exists():
        raise RuntimeError("run --stage freeze before evaluate")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_confirmation_execution":
        raise RuntimeError("invalid freeze status")
    if frozen.get("config") != _config() or frozen.get("hashes") != _hashes():
        raise RuntimeError("configuration or code changed after freeze")
    return frozen


def _relative_variance(name: str, x: np.ndarray) -> np.ndarray:
    if name == "constant":
        raw = np.ones_like(x)
    elif name == "linear":
        raw = 0.35 + 1.30 * x
    elif name == "plume_peak":
        raw = 0.25 + 2.75 * np.exp(-((x - 0.58) / 0.12) ** 2)
    else:
        raise ValueError(name)
    return raw / np.mean(raw)


def _band(estimate: np.ndarray, radius: np.ndarray, alpha: float) -> ShapeContrastBand:
    return ShapeContrastBand(
        estimate=estimate,
        lower=estimate - radius,
        upper=estimate + radius,
        radius=radius,
        critical_value=float("nan"),
        alpha=alpha,
        noise_scale=float("nan"),
    )


def _simulation_scores() -> list[dict[str, object]]:
    specifications = [
        (n, design, signal, profile)
        for n in SAMPLE_SIZES
        for design in DESIGNS
        for signal in SIGNALS
        for profile in VARIANCE_PROFILES
    ]
    children = np.random.SeedSequence(SEED).spawn(len(specifications))
    families: dict[tuple[int, str], object] = {}
    rows: list[dict[str, object]] = []
    for (n, design, signal, profile), child in zip(
        specifications, children, strict=True
    ):
        x = signals.design_points(design, n)
        family = families.setdefault(
            (n, design),
            build_shape_contrast_family(x, separation_multipliers=SEPARATIONS),
        )
        truth, target = signals.signal_values(signal, x)
        true_contrasts = family.means(truth)
        relative_variance = _relative_variance(profile, x)
        variances = AVERAGE_SIGMA**2 * relative_variance
        standard_deviations = np.sqrt(variances)
        kappa = float(relative_variance.max())
        rng = np.random.default_rng(child)
        responses = truth + rng.normal(0.0, standard_deviations, size=(TRIALS, n))
        oracle_standard_errors = np.sqrt((family.operator**2) @ variances)
        oracle_critical = float(
            norm.ppf(1.0 - ALPHA / (2.0 * family.contrast_count))
        )
        unknown_critical = float(
            norm.ppf(1.0 - CONTRAST_ALPHA / (2.0 * family.contrast_count))
        )
        for trial, response in enumerate(responses):
            estimate = family.means(response)
            envelope = gaussian_heteroskedastic_upper_envelope(
                response,
                max_to_mean_variance_ratio=kappa,
                failure_probability=SCALE_FAILURE,
            )
            methods = (
                (
                    "oracle_variances",
                    oracle_critical * oracle_standard_errors,
                    ALPHA,
                    True,
                    float(standard_deviations.max()),
                ),
                (
                    "unknown_heteroskedastic",
                    unknown_critical * envelope.upper_scale * family.weight_l2,
                    CONTRAST_ALPHA,
                    envelope.upper_scale >= standard_deviations.max() - 1e-12,
                    envelope.upper_scale,
                ),
            )
            for method, radius, band_alpha, scale_covers, scale_upper in methods:
                band = _band(estimate, radius, band_alpha)
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
                        "cell": f"{signal}__{design}__{profile}__n{n}",
                        "signal": signal,
                        "design": design,
                        "variance_profile": profile,
                        "n": n,
                        "trial": trial,
                        "method": method,
                        "kappa": kappa,
                        "target_left": target[0],
                        "target_right": target[1],
                        "confidence_left": confidence.left,
                        "confidence_right": confidence.right,
                        "confidence_width": confidence.width,
                        "covers_target": covers,
                        "empty": confidence.empty,
                        "joint_contrast_coverage": joint,
                        "unexplained_empty": confidence.empty and joint,
                        "scale_upper": scale_upper,
                        "scale_ratio_to_true_max": scale_upper
                        / float(standard_deviations.max()),
                        "scale_covers": scale_covers,
                    }
                )
    return rows


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["cell"],
            row["signal"],
            row["design"],
            row["variance_profile"],
            row["n"],
            row["method"],
            row["kappa"],
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        widths = np.asarray([row["confidence_width"] for row in group], dtype=float)
        summary.append(
            {
                "cell": key[0],
                "signal": key[1],
                "design": key[2],
                "variance_profile": key[3],
                "n": key[4],
                "method": key[5],
                "kappa": key[6],
                "coverage": np.mean([row["covers_target"] for row in group]),
                "mean_width": float(np.nanmean(widths)),
                "median_width": float(np.nanmedian(widths)),
                "empty_count": sum(row["empty"] for row in group),
                "unexplained_empty_count": sum(
                    row["unexplained_empty"] for row in group
                ),
                "joint_contrast_coverage": np.mean(
                    [row["joint_contrast_coverage"] for row in group]
                ),
                "scale_coverage": np.mean([row["scale_covers"] for row in group]),
                "mean_scale_ratio_to_true_max": np.mean(
                    [row["scale_ratio_to_true_max"] for row in group]
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_lidar(output_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rscript = PROJECT / "third_party/r_runtime/R-4.6.1/bin/Rscript.exe"
    subprocess.run(
        [
            str(rscript),
            str(_paths()["r_bridge"]),
            str(output_dir),
            str(LIDAR_BOOTSTRAP_LOOPS),
            str(LIDAR_SEED),
        ],
        cwd=PROJECT,
        check=True,
    )
    data = np.genfromtxt(
        output_dir / "lidar_data_and_fit.csv", delimiter=",", names=True
    )
    with (output_dir / "lidar_external_fits.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        external = next(csv.DictReader(handle))
    x = np.asarray(data["x_normalized"], dtype=float)
    y = np.asarray(data["minus_logratio"], dtype=float)
    family = build_shape_contrast_family(x, separation_multipliers=SEPARATIONS)
    estimate = family.means(y)
    critical = float(
        norm.ppf(1.0 - CONTRAST_ALPHA / (2.0 * family.contrast_count))
    )
    minimum_range = float(np.min(data["range"]))
    range_span = float(np.max(data["range"]) - minimum_range)
    rows = []
    for kappa in LIDAR_KAPPAS:
        envelope = gaussian_heteroskedastic_upper_envelope(
            y,
            max_to_mean_variance_ratio=kappa,
            failure_probability=SCALE_FAILURE,
        )
        radius = critical * envelope.upper_scale * family.weight_l2
        confidence = invert_s_shaped_inflection(
            family, _band(estimate, radius, CONTRAST_ALPHA), domain=DOMAIN
        )
        rows.append(
            {
                "kappa": kappa,
                "upper_noise_scale": envelope.upper_scale,
                "concentration_denominator": envelope.concentration_denominator,
                "left_normalized": confidence.left,
                "right_normalized": confidence.right,
                "left_metres": minimum_range + range_span * confidence.left,
                "right_metres": minimum_range + range_span * confidence.right,
                "width_metres": range_span * confidence.width,
                "empty": confidence.empty,
                "positive_contrasts": confidence.positive_contrast_count,
                "negative_contrasts": confidence.negative_contrast_count,
            }
        )
    return rows, external


def _gates(summary: list[dict[str, object]]) -> dict[str, bool]:
    unknown = [row for row in summary if row["method"] == "unknown_heteroskedastic"]
    return {
        "all_48_unknown_cells_retained": len(unknown) == 48,
        "coverage_at_least_0_93_every_cell": all(
            float(row["coverage"]) >= 0.93 for row in unknown
        ),
        "scale_coverage_at_least_0_93_every_cell": all(
            float(row["scale_coverage"]) >= 0.93 for row in unknown
        ),
        "zero_unexplained_empty_sets": all(
            int(row["unexplained_empty_count"]) == 0 for row in unknown
        ),
        "weak_signal_cells_retained": len(
            [row for row in unknown if row["signal"] == "paper_f4_logistic"]
        )
        == 12,
    }


def _figure(output_dir: Path, lidar_rows: list[dict[str, object]], external) -> None:
    data = np.genfromtxt(
        output_dir / "lidar_data_and_fit.csv", delimiter=",", names=True
    )
    point = float(external["sshaped_point_metres"])
    figure, axes = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [2.2, 1.0]}, constrained_layout=True
    )
    axes[0].scatter(data["range"], data["minus_logratio"], s=11, alpha=0.55)
    axes[0].plot(data["range"], data["sshaped_fitted"], color="black", linewidth=2)
    axes[0].axvline(point, color="tab:orange", linestyle="--", label="S-shaped LSE")
    axes[0].set_ylabel("-logratio")
    axes[0].set_xlabel("range (m)")
    axes[0].legend()
    for index, row in enumerate(lidar_rows):
        axes[1].plot(
            [row["left_metres"], row["right_metres"]],
            [index, index],
            linewidth=7,
            solid_capstyle="butt",
        )
        axes[1].plot(point, index, marker="|", color="black", markersize=13)
    axes[1].set_yticks(
        np.arange(len(lidar_rows)), [f"kappa={row['kappa']:g}" for row in lidar_rows]
    )
    axes[1].set_xlabel("HCT sensitivity interval for plume centre (m)")
    axes[1].set_xlim(float(data["range"].min()), float(data["range"].max()))
    figure.savefig(output_dir / "lidar_sensitivity.png", dpi=180)
    figure.savefig(output_dir / "lidar_sensitivity.pdf")
    plt.close(figure)


def _report(
    summary: list[dict[str, object]],
    lidar_rows: list[dict[str, object]],
    external: dict[str, object],
    gates: dict[str, bool],
) -> Path:
    unknown = [row for row in summary if row["method"] == "unknown_heteroskedastic"]
    lines = [
        "# E35: unknown heteroskedastic HCT confirmation",
        "",
        "## Frozen simulation result",
        "",
        f"Minimum HCT coverage across 48 cells: `{min(float(row['coverage']) for row in unknown):.3f}`.",
        f"Minimum variance-envelope coverage: `{min(float(row['scale_coverage']) for row in unknown):.3f}`.",
        "",
        "### Frozen gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in gates.items()
    )
    lines.extend(
        [
            "",
            "## LIDAR sensitivity illustration",
            "",
            f"Current `Sshaped` point estimate: `{float(external['sshaped_point_metres']):.1f} m`.",
            "",
            "| kappa | HCT interval (m) | width (m) | upper noise scale |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in lidar_rows:
        lines.append(
            f"| {float(row['kappa']):.1f} | "
            f"[{float(row['left_metres']):.1f}, {float(row['right_metres']):.1f}] | "
            f"{float(row['width_metres']):.1f} | {float(row['upper_noise_scale']):.4f} |"
        )
    if external["shapechange_status"] == "ok":
        left = 390.0 + 330.0 * float(external["shapechange_left_normalized"])
        right = 390.0 + 330.0 * float(external["shapechange_right_normalized"])
        point = 390.0 + 330.0 * float(external["shapechange_point_normalized"])
        lines.extend(
            [
                "",
                f"Nominal ShapeChange residual-bootstrap estimate: `{point:.1f} m`, "
                f"interval `[{left:.1f}, {right:.1f}] m`.",
            ]
        )
    lines.extend(
        [
            "",
            "The HCT rows are finite-sample confidence statements only when errors are "
            "independent Gaussian and the displayed kappa is a valid upper bound. The "
            "ShapeChange interval is descriptive here because its iid residual bootstrap "
            "does not model the visible heteroskedasticity.",
        ]
    )
    path = PROJECT / "reports/hct/e35_heteroskedastic_lidar.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate(output_dir: Path) -> None:
    frozen = _validate_freeze(output_dir)
    if (output_dir / "trial_scores.csv").exists():
        raise RuntimeError("E35 already executed; refusing overwrite")
    rows = _simulation_scores()
    summary = _summarize(rows)
    _write_csv(output_dir / "trial_scores.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    lidar_rows, external = _run_lidar(output_dir)
    _write_csv(output_dir / "lidar_hct_sensitivity.csv", lidar_rows)
    gates = _gates(summary)
    _figure(output_dir, lidar_rows, external)
    report = _report(summary, lidar_rows, external, gates)
    products = {
        name: _sha256(output_dir / name)
        for name in (
            "trial_scores.csv",
            "summary.csv",
            "lidar_data_and_fit.csv",
            "lidar_external_fits.csv",
            "lidar_hct_sensitivity.csv",
            "lidar_sensitivity.png",
        )
    }
    manifest = {
        "experiment": "E35 bounded-heteroskedastic HCT confirmation and LIDAR",
        "status": "confirmation_executed_after_freeze",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config(),
        "freeze_sha256": _sha256(output_dir / "frozen_config.json"),
        "frozen": frozen,
        "gates": {"all_pass": all(gates.values()), **gates},
        "hashes": {**_hashes(), **products, "report": _sha256(report)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "lidar": {"external": external, "hct_sensitivity": lidar_rows},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"gates": manifest["gates"], "lidar": lidar_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/hct/heteroskedastic_shape_inversion_e35_confirmation",
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output_dir)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()

