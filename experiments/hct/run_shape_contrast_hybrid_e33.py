"""E33 frozen comparison of honest shape inversion and S-shaped frontiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binomtest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.shape_inflection_confidence import (  # noqa: E402
    build_shape_contrast_family,
    gaussian_bonferroni_shape_band,
    invert_s_shaped_inflection,
)


ALPHA = 0.05
SIGMA = 0.1
DOMAIN = (0.0, 1.0)
SEPARATION_MULTIPLIERS = (1, 2, 4)
PAPER_SIGNALS = (
    "paper_f1_cusp",
    "paper_f2_onset",
    "paper_f3_jump",
    "paper_f4_logistic",
)
CONTROL_SIGNALS = (
    "control_affine",
    "control_convex",
    "control_concave",
)
SIGNALS = PAPER_SIGNALS + CONTROL_SIGNALS


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    status: str
    sample_sizes: tuple[int, ...]
    designs: tuple[str, ...]
    trials: int
    data_seed: int
    bootstrap_seed: int
    bootstrap_loops: int
    shards: int


PHASES = {
    "smoke": PhaseConfig(
        name="smoke",
        status="smoke_only",
        sample_sizes=(100,),
        designs=("uniform",),
        trials=2,
        data_seed=20261701,
        bootstrap_seed=20261702,
        bootstrap_loops=20,
        shards=1,
    ),
    "confirmation": PhaseConfig(
        name="confirmation",
        status="confirmation",
        sample_sizes=(500, 1000),
        designs=("uniform", "beta_4_8"),
        trials=200,
        data_seed=20261711,
        bootstrap_seed=20261712,
        bootstrap_loops=1000,
        shards=4,
    ),
}


def _config_payload(config: PhaseConfig) -> dict[str, object]:
    """Return the JSON representation used in frozen equality checks."""

    return json.loads(json.dumps(asdict(config)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite(values: list[object]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=float)
    return array[np.isfinite(array)]


def _finite_mean(values: list[object]) -> float:
    array = _finite(values)
    return float(np.mean(array)) if array.size else float("nan")


def _finite_median(values: list[object]) -> float:
    array = _finite(values)
    return float(np.median(array)) if array.size else float("nan")


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        return float("nan"), float("nan")
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


def design_points(name: str, n: int) -> np.ndarray:
    probabilities = np.arange(1, n + 1, dtype=float) / (n + 1.0)
    if name == "uniform":
        return probabilities
    if name == "beta_4_8":
        return np.asarray(beta.ppf(probabilities, 4.0, 8.0), dtype=float)
    raise ValueError(f"unknown design: {name}")


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
    raise ValueError(f"unknown signal: {name}")


def _cell(signal: str, design: str, n: int) -> str:
    return f"{signal}__{design}__n{n}"


def _code_paths() -> dict[str, Path]:
    script = Path(__file__).resolve()
    return {
        "driver": script,
        "r_bridge": script.with_name("run_shape_contrast_comparators_e33.R"),
        "protocol": PROJECT / "docs" / "hct_e33_shape_contrast_inflection_protocol.md",
        "shape_module": PROJECT / "src" / "hcrd" / "shape_inflection_confidence.py",
        "shapechange_source": PROJECT / "third_party" / "ShapeChange_1.5.tar.gz",
        "sshaped_description": PROJECT
        / "third_party/r_runtime/R-4.6.1/library/Sshaped/DESCRIPTION",
        "shapechange_description": PROJECT
        / "third_party/r_runtime/R-4.6.1/library/ShapeChange/DESCRIPTION",
        "coneproj_description": PROJECT
        / "third_party/r_runtime/R-4.6.1/library/coneproj/DESCRIPTION",
        "quadprog_description": PROJECT
        / "third_party/r_runtime/R-4.6.1/library/quadprog/DESCRIPTION",
    }


def _code_hashes() -> dict[str, str]:
    paths = _code_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen dependency: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def freeze_confirmation(config: PhaseConfig, output_dir: Path) -> None:
    if config.name != "confirmation":
        raise ValueError("only the confirmation phase can be frozen")
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = [
        output_dir / "hct_scores.csv",
        output_dir / "trial_scores.csv",
        output_dir / "manifest.json",
    ] + [output_dir / f"observations_shard_{index}.csv" for index in range(config.shards)]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("confirmation output already exists; refusing to refreeze")
    payload = {
        "status": "frozen_before_confirmation_execution",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config_payload(config),
        "alpha": ALPHA,
        "sigma": SIGMA,
        "domain": DOMAIN,
        "signals": SIGNALS,
        "paper_signals": PAPER_SIGNALS,
        "separation_multipliers": SEPARATION_MULTIPLIERS,
        "calibration": "two-sided analytic Gaussian Bonferroni",
        "external_methods": {
            "Sshaped": {"version": "1.2", "role": "point estimator"},
            "ShapeChange": {
                "version": "1.5",
                "role": "nominal 95% residual-bootstrap interval",
                "bootstrap_loops": config.bootstrap_loops,
            },
        },
        "hashes": _code_hashes(),
    }
    (output_dir / "frozen_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def _load_and_validate_freeze(config: PhaseConfig, output_dir: Path) -> dict[str, object]:
    freeze_path = output_dir / "frozen_config.json"
    if not freeze_path.exists():
        raise RuntimeError("confirmation is locked until --stage freeze succeeds")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_confirmation_execution":
        raise RuntimeError("invalid confirmation freeze status")
    if payload.get("config") != _config_payload(config):
        raise RuntimeError("runtime configuration differs from frozen configuration")
    if payload.get("hashes") != _code_hashes():
        raise RuntimeError("code or dependency changed after confirmation freeze")
    return payload


def _load_and_validate_pre_manifest(
    config: PhaseConfig, output_dir: Path
) -> dict[str, object]:
    path = output_dir / "pre_manifest.json"
    if not path.exists():
        raise RuntimeError("run --stage prepare before comparator evaluation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config") != _config_payload(config):
        raise RuntimeError("prepared responses use a different configuration")
    outputs = payload.get("outputs", {})
    expected = {
        "designs": _sha256(output_dir / "designs.csv"),
        "hct_scores": _sha256(output_dir / "hct_scores.csv"),
        "families": _sha256(output_dir / "families.csv"),
        "observation_shards": {
            str(index): _sha256(output_dir / f"observations_shard_{index}.csv")
            for index in range(config.shards)
        },
    }
    if outputs != expected:
        raise RuntimeError("prepared response files changed after generation")
    return payload


def prepare(config: PhaseConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = None
    if config.name == "confirmation":
        freeze = _load_and_validate_freeze(config, output_dir)
        generated = [output_dir / "hct_scores.csv"] + [
            output_dir / f"observations_shard_{index}.csv" for index in range(config.shards)
        ]
        if any(path.exists() for path in generated):
            raise RuntimeError("confirmation responses already generated; refusing overwrite")

    designs_rows: list[dict[str, object]] = []
    for n in config.sample_sizes:
        for design in config.designs:
            for index, value in enumerate(design_points(design, n)):
                designs_rows.append({"design": design, "n": n, "index": index, "x": value})
    _write_csv(output_dir / "designs.csv", designs_rows)

    specifications = [
        (n, design, signal)
        for n in config.sample_sizes
        for design in config.designs
        for signal in SIGNALS
    ]
    data_children = np.random.SeedSequence(config.data_seed).spawn(len(specifications))
    bootstrap_rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_seeds = iter(
        bootstrap_rng.integers(
            1,
            np.iinfo(np.int32).max,
            size=len(config.sample_sizes)
            * len(config.designs)
            * len(PAPER_SIGNALS)
            * config.trials,
        )
    )
    hct_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    shard_rows: list[list[dict[str, object]]] = [[] for _ in range(config.shards)]
    paper_trial_index = 0
    family_cache = {}

    for (n, design, signal), child in zip(specifications, data_children, strict=True):
        x = design_points(design, n)
        family_key = (n, design)
        if family_key not in family_cache:
            family_started = perf_counter()
            family = build_shape_contrast_family(
                x, separation_multipliers=SEPARATION_MULTIPLIERS
            )
            family_seconds = perf_counter() - family_started
            family_cache[family_key] = family
            critical = gaussian_bonferroni_shape_band(
                family, np.zeros(n), noise_scale=SIGMA, alpha=ALPHA
            ).critical_value
            family_rows.append(
                {
                    "design": design,
                    "n": n,
                    "contrast_count": family.contrast_count,
                    "critical_value": critical,
                    "minimum_block_size": int(family.block_size.min()),
                    "maximum_block_size": int(family.block_size.max()),
                    "maximum_separation": int(family.separation.max()),
                    "preparation_seconds": family_seconds,
                }
            )
        family = family_cache[family_key]

        truth, target = signal_values(signal, x)
        true_contrasts = family.means(truth)
        rng = np.random.default_rng(child)
        responses = truth + rng.normal(0.0, SIGMA, size=(config.trials, n))
        cell = _cell(signal, design, n)
        for trial, response in enumerate(responses):
            started = perf_counter()
            band = gaussian_bonferroni_shape_band(
                family, response, noise_scale=SIGMA, alpha=ALPHA
            )
            confidence_set = invert_s_shaped_inflection(family, band, domain=DOMAIN)
            elapsed = perf_counter() - started
            joint_coverage = bool(
                np.all(true_contrasts >= band.lower - 1e-12)
                and np.all(true_contrasts <= band.upper + 1e-12)
            )
            target_coverage = bool(
                not confidence_set.empty
                and confidence_set.left <= target[0] + 1e-12
                and confidence_set.right >= target[1] - 1e-12
            )
            hct_rows.append(
                {
                    "cell": cell,
                    "signal": signal,
                    "design": design,
                    "n": n,
                    "trial": trial,
                    "target_left": target[0],
                    "target_right": target[1],
                    "hct_left": confidence_set.left,
                    "hct_right": confidence_set.right,
                    "hct_width": confidence_set.width,
                    "hct_empty": confidence_set.empty,
                    "hct_covers_target": target_coverage,
                    "hct_nontrivial": (
                        not confidence_set.empty and confidence_set.width < 1.0 - 1e-12
                    ),
                    "joint_contrast_coverage": joint_coverage,
                    "unexplained_empty": confidence_set.empty and joint_coverage,
                    "positive_contrast_count": confidence_set.positive_contrast_count,
                    "negative_contrast_count": confidence_set.negative_contrast_count,
                    "hct_runtime_seconds": elapsed,
                }
            )
            if signal in PAPER_SIGNALS:
                bootstrap_seed = int(next(bootstrap_seeds))
                shard_rows[paper_trial_index % config.shards].append(
                    {
                        "cell": cell,
                        "signal": signal,
                        "design": design,
                        "n": n,
                        "trial": trial,
                        "bootstrap_seed": bootstrap_seed,
                        "y": "|".join(format(float(value), ".17g") for value in response),
                    }
                )
                paper_trial_index += 1

    _write_csv(output_dir / "hct_scores.csv", hct_rows)
    _write_csv(output_dir / "families.csv", family_rows)
    for index, rows in enumerate(shard_rows):
        _write_csv(output_dir / f"observations_shard_{index}.csv", rows)
    try:
        next(bootstrap_seeds)
    except StopIteration:
        pass
    else:
        raise RuntimeError("unused bootstrap seeds indicate a preparation bug")

    outputs = {
        "designs": _sha256(output_dir / "designs.csv"),
        "hct_scores": _sha256(output_dir / "hct_scores.csv"),
        "families": _sha256(output_dir / "families.csv"),
        "observation_shards": {
            str(index): _sha256(output_dir / f"observations_shard_{index}.csv")
            for index in range(config.shards)
        },
    }
    pre_manifest = {
        "status": "responses_generated_after_freeze" if freeze else "smoke_responses",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config_payload(config),
        "freeze_sha256": (
            _sha256(output_dir / "frozen_config.json") if freeze else None
        ),
        "outputs": outputs,
    }
    (output_dir / "pre_manifest.json").write_text(
        json.dumps(pre_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(pre_manifest, indent=2))


def run_comparator_shard(
    config: PhaseConfig, output_dir: Path, shard_index: int, rscript: Path
) -> None:
    if not 0 <= shard_index < config.shards:
        raise ValueError(f"shard must lie in [0,{config.shards})")
    if config.name == "confirmation":
        _load_and_validate_freeze(config, output_dir)
    _load_and_validate_pre_manifest(config, output_dir)
    bridge = Path(__file__).with_name("run_shape_contrast_comparators_e33.R")
    command = [
        str(rscript),
        str(bridge),
        str(output_dir / f"observations_shard_{shard_index}.csv"),
        str(output_dir / "designs.csv"),
        str(output_dir / f"comparators_shard_{shard_index}.csv"),
        str(config.bootstrap_loops),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"R comparator shard {shard_index} failed")


def _comparator_map(config: PhaseConfig, output_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(config.shards):
        path = output_dir / f"comparators_shard_{index}.csv"
        if not path.exists():
            raise RuntimeError(f"missing comparator shard: {path}")
        rows.extend(_read_csv(path))
    output: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["cell"], int(row["trial"]))
        if key in output:
            raise RuntimeError(f"duplicate comparator result: {key}")
        output[key] = row
    expected = (
        len(config.sample_sizes)
        * len(config.designs)
        * len(PAPER_SIGNALS)
        * config.trials
    )
    if len(output) != expected:
        raise RuntimeError(f"expected {expected} comparator rows, found {len(output)}")
    return output


def merge_scores(config: PhaseConfig, output_dir: Path) -> list[dict[str, object]]:
    external = _comparator_map(config, output_dir)
    output: list[dict[str, object]] = []
    for hct in _read_csv(output_dir / "hct_scores.csv"):
        signal = hct["signal"]
        row: dict[str, object] = dict(hct)
        if signal in PAPER_SIGNALS:
            comparator = external[(hct["cell"], int(hct["trial"]))]
            sshaped_ok = comparator["sshaped_status"] == "ok"
            shapechange_ok = comparator["shapechange_status"] == "ok"
            root = float(hct["target_left"])
            sshaped = (
                float(comparator["sshaped_inflection"]) if sshaped_ok else float("nan")
            )
            shape_left = (
                float(comparator["shapechange_left"])
                if shapechange_ok
                else float("nan")
            )
            shape_right = (
                float(comparator["shapechange_right"])
                if shapechange_ok
                else float("nan")
            )
            hct_empty = _as_bool(hct["hct_empty"])
            projected = (
                float(np.clip(sshaped, float(hct["hct_left"]), float(hct["hct_right"])))
                if sshaped_ok and not hct_empty
                else float("nan")
            )
            sshaped_error = abs(sshaped - root) if sshaped_ok else float("nan")
            projected_error = (
                abs(projected - root) if np.isfinite(projected) else float("nan")
            )
            row.update(
                {
                    "sshaped_status": comparator["sshaped_status"],
                    "sshaped_inflection": sshaped,
                    "sshaped_root_error": sshaped_error,
                    "sshaped_runtime_seconds": comparator["sshaped_runtime_seconds"],
                    "shapechange_status": comparator["shapechange_status"],
                    "shapechange_inflection": comparator["shapechange_inflection"],
                    "shapechange_left": shape_left,
                    "shapechange_right": shape_right,
                    "shapechange_width": (
                        shape_right - shape_left if shapechange_ok else float("nan")
                    ),
                    "shapechange_covers_target": bool(
                        shapechange_ok and shape_left <= root + 1e-12 and shape_right >= root - 1e-12
                    ),
                    "shapechange_root_error": (
                        abs(float(comparator["shapechange_inflection"]) - root)
                        if shapechange_ok
                        else float("nan")
                    ),
                    "shapechange_runtime_seconds": comparator[
                        "shapechange_runtime_seconds"
                    ],
                    "hct_projected_sshaped": projected,
                    "hct_projected_root_error": projected_error,
                    "projection_error_change": projected_error - sshaped_error,
                    "covered_projection_increase": bool(
                        _as_bool(hct["hct_covers_target"])
                        and np.isfinite(projected_error)
                        and projected_error > sshaped_error + 1e-12
                    ),
                }
            )
        else:
            row.update(
                {
                    "sshaped_status": "not_run_control",
                    "sshaped_inflection": float("nan"),
                    "sshaped_root_error": float("nan"),
                    "sshaped_runtime_seconds": float("nan"),
                    "shapechange_status": "not_run_control",
                    "shapechange_inflection": float("nan"),
                    "shapechange_left": float("nan"),
                    "shapechange_right": float("nan"),
                    "shapechange_width": float("nan"),
                    "shapechange_covers_target": False,
                    "shapechange_root_error": float("nan"),
                    "shapechange_runtime_seconds": float("nan"),
                    "hct_projected_sshaped": float("nan"),
                    "hct_projected_root_error": float("nan"),
                    "projection_error_change": float("nan"),
                    "covered_projection_increase": False,
                }
            )
        output.append(row)
    return output


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for cell in sorted({str(row["cell"]) for row in rows}):
        selected = [row for row in rows if row["cell"] == cell]
        total = len(selected)
        signal = str(selected[0]["signal"])
        paper = signal in PAPER_SIGNALS
        hct_covered = sum(_as_bool(row["hct_covers_target"]) for row in selected)
        hct_interval = _wilson(hct_covered, total)
        shape_covered = (
            sum(_as_bool(row["shapechange_covers_target"]) for row in selected)
            if paper
            else 0
        )
        shape_interval = _wilson(shape_covered, total) if paper else (float("nan"),) * 2
        both = (
            sum(
                _as_bool(row["hct_covers_target"])
                and _as_bool(row["shapechange_covers_target"])
                for row in selected
            )
            if paper
            else 0
        )
        hct_only = hct_covered - both
        shape_only = shape_covered - both
        mcnemar_p = (
            float(binomtest(hct_only, hct_only + shape_only, 0.5).pvalue)
            if paper and hct_only + shape_only > 0
            else float("nan")
        )
        output.append(
            {
                "cell": cell,
                "signal": signal,
                "design": selected[0]["design"],
                "n": int(selected[0]["n"]),
                "trials": total,
                "hct_coverage": hct_covered / total,
                "hct_coverage_wilson_low": hct_interval[0],
                "hct_coverage_wilson_high": hct_interval[1],
                "hct_mean_width": _finite_mean([row["hct_width"] for row in selected]),
                "hct_median_width": _finite_median([row["hct_width"] for row in selected]),
                "hct_nontrivial_probability": float(
                    np.mean([_as_bool(row["hct_nontrivial"]) for row in selected])
                ),
                "hct_empty_probability": float(
                    np.mean([_as_bool(row["hct_empty"]) for row in selected])
                ),
                "joint_contrast_coverage": float(
                    np.mean([_as_bool(row["joint_contrast_coverage"]) for row in selected])
                ),
                "unexplained_empty_count": sum(
                    _as_bool(row["unexplained_empty"]) for row in selected
                ),
                "mean_hct_runtime_seconds": _finite_mean(
                    [row["hct_runtime_seconds"] for row in selected]
                ),
                "sshaped_success_probability": (
                    float(np.mean([row["sshaped_status"] == "ok" for row in selected]))
                    if paper
                    else float("nan")
                ),
                "sshaped_mean_root_error": _finite_mean(
                    [row["sshaped_root_error"] for row in selected]
                ),
                "sshaped_median_root_error": _finite_median(
                    [row["sshaped_root_error"] for row in selected]
                ),
                "projected_mean_root_error": _finite_mean(
                    [row["hct_projected_root_error"] for row in selected]
                ),
                "projected_median_root_error": _finite_median(
                    [row["hct_projected_root_error"] for row in selected]
                ),
                "mean_projection_error_change": _finite_mean(
                    [row["projection_error_change"] for row in selected]
                ),
                "covered_projection_increase_count": sum(
                    _as_bool(row["covered_projection_increase"]) for row in selected
                ),
                "shapechange_success_probability": (
                    float(np.mean([row["shapechange_status"] == "ok" for row in selected]))
                    if paper
                    else float("nan")
                ),
                "shapechange_coverage": shape_covered / total if paper else float("nan"),
                "shapechange_coverage_wilson_low": shape_interval[0],
                "shapechange_coverage_wilson_high": shape_interval[1],
                "shapechange_mean_width": _finite_mean(
                    [row["shapechange_width"] for row in selected]
                ),
                "shapechange_median_width": _finite_median(
                    [row["shapechange_width"] for row in selected]
                ),
                "shapechange_mean_root_error": _finite_mean(
                    [row["shapechange_root_error"] for row in selected]
                ),
                "coverage_difference_hct_minus_shapechange": (
                    (hct_covered - shape_covered) / total if paper else float("nan")
                ),
                "paired_hct_only_coverage": hct_only if paper else "",
                "paired_shapechange_only_coverage": shape_only if paper else "",
                "paired_exact_p_value": mcnemar_p,
                "mean_sshaped_runtime_seconds": _finite_mean(
                    [row["sshaped_runtime_seconds"] for row in selected]
                ),
                "mean_shapechange_runtime_seconds": _finite_mean(
                    [row["shapechange_runtime_seconds"] for row in selected]
                ),
            }
        )
    return output


def evaluate_gates(summary: list[dict[str, object]]) -> dict[str, object]:
    paper = [row for row in summary if row["signal"] in PAPER_SIGNALS]
    all_rows = list(summary)
    width_limits = {
        "paper_f1_cusp": 0.15,
        "paper_f2_onset": 0.85,
        "paper_f3_jump": 0.02,
    }
    width_rows = [
        row
        for row in paper
        if int(row["n"]) == 1000 and row["signal"] in width_limits
    ]
    nonsmooth_advantages = [
        row
        for row in paper
        if row["signal"] in tuple(width_limits)
        and float(row["coverage_difference_hct_minus_shapechange"]) >= 0.10
    ]
    total_sshaped_error = _finite_mean(
        [row["sshaped_mean_root_error"] for row in paper]
    )
    total_projected_error = _finite_mean(
        [row["projected_mean_root_error"] for row in paper]
    )
    gates = {
        "hct_coverage_at_least_0_93_every_cell": all(
            float(row["hct_coverage"]) >= 0.93 for row in all_rows
        ),
        "zero_unexplained_empty_sets": all(
            int(row["unexplained_empty_count"]) == 0 for row in all_rows
        ),
        "external_fit_success_at_least_0_98_every_paper_cell": all(
            float(row["sshaped_success_probability"]) >= 0.98
            and float(row["shapechange_success_probability"]) >= 0.98
            for row in paper
        ),
        "predeclared_large_n_width_limits": all(
            float(row["hct_median_width"]) < width_limits[str(row["signal"])]
            for row in width_rows
        )
        and len(width_rows) == 6,
        "no_projection_increase_on_covered_trials": all(
            int(row["covered_projection_increase_count"]) == 0 for row in paper
        ),
        "aggregate_projection_does_not_increase_mean_error": (
            total_projected_error <= total_sshaped_error + 1e-12
        ),
        "publication_separation_at_least_two_nonsmooth_cells": (
            len(nonsmooth_advantages) >= 2
        ),
        "weak_logistic_regime_retained": len(
            [row for row in paper if row["signal"] == "paper_f4_logistic"]
        )
        == 4,
    }
    return {
        "all_core_validity_gates_pass": all(
            gates[name]
            for name in (
                "hct_coverage_at_least_0_93_every_cell",
                "zero_unexplained_empty_sets",
                "external_fit_success_at_least_0_98_every_paper_cell",
                "predeclared_large_n_width_limits",
                "no_projection_increase_on_covered_trials",
                "aggregate_projection_does_not_increase_mean_error",
                "weak_logistic_regime_retained",
            )
        ),
        "publication_separation_gate_pass": gates[
            "publication_separation_at_least_two_nonsmooth_cells"
        ],
        "aggregate_sshaped_mean_error": total_sshaped_error,
        "aggregate_projected_mean_error": total_projected_error,
        "nonsmooth_advantage_cells": [row["cell"] for row in nonsmooth_advantages],
        "gates": gates,
    }


def make_figure(summary: list[dict[str, object]], path: Path) -> None:
    paper = [row for row in summary if row["signal"] in PAPER_SIGNALS]
    labels = [
        f"{str(row['signal']).replace('paper_', '')}\n{row['design']}, n={row['n']}"
        for row in paper
    ]
    positions = np.arange(len(paper))
    width = 0.38
    figure, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    axes[0].bar(
        positions - width / 2,
        [row["hct_coverage"] for row in paper],
        width,
        label="E33 honest shape inversion",
    )
    axes[0].bar(
        positions + width / 2,
        [row["shapechange_coverage"] for row in paper],
        width,
        label="ShapeChange bootstrap",
    )
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_title("Nominal 95% inflection intervals")
    axes[0].legend()

    axes[1].bar(
        positions - width / 2,
        [row["hct_median_width"] for row in paper],
        width,
        label="E33",
    )
    axes[1].bar(
        positions + width / 2,
        [row["shapechange_median_width"] for row in paper],
        width,
        label="ShapeChange",
    )
    axes[1].set_ylabel("median interval width")
    axes[1].set_title("Informativeness (unconditional)")
    axes[1].legend()

    axes[2].bar(
        positions - width / 2,
        [row["sshaped_mean_root_error"] for row in paper],
        width,
        label="official Sshaped point",
    )
    axes[2].bar(
        positions + width / 2,
        [row["projected_mean_root_error"] for row in paper],
        width,
        label="Sshaped projected into E33 set",
    )
    axes[2].set_ylabel("mean absolute localization error")
    axes[2].set_title("Estimator-agnostic E33 hybrid")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(
    config: PhaseConfig,
    summary: list[dict[str, object]],
    gates: dict[str, object],
    output_dir: Path,
) -> Path:
    paper = [row for row in summary if row["signal"] in PAPER_SIGNALS]
    lines = [
        "# E33: honest S-shaped inflection UQ and frontier hybrid",
        "",
        f"Status: **{config.status}**.  Nominal level: 95%.  "
        f"Repetitions: {config.trials} per cell.",
        "",
        "E33 uses a fixed multiscale family of sign-valid chord contrasts with "
        "analytic Gaussian Bonferroni calibration. `Sshaped` 1.2 is the external "
        "point estimator; `ShapeChange` 1.5 with its documented residual-bootstrap "
        f"interval ({config.bootstrap_loops} resamples) is the UQ comparator.",
        "",
        "| signal | design | n | E33 cover | E33 med. width | ShapeChange cover | ShapeChange med. width | Sshaped MAE | projected MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper:
        lines.append(
            "| {signal} | {design} | {n} | {hc:.3f} | {hw:.4f} | {sc:.3f} | "
            "{sw:.4f} | {se:.4f} | {pe:.4f} |".format(
                signal=str(row["signal"]).replace("paper_", ""),
                design=row["design"],
                n=row["n"],
                hc=float(row["hct_coverage"]),
                hw=float(row["hct_median_width"]),
                sc=float(row["shapechange_coverage"]),
                sw=float(row["shapechange_median_width"]),
                se=float(row["sshaped_mean_root_error"]),
                pe=float(row["projected_mean_root_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
        ]
    )
    for name, passed in dict(gates["gates"]).items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "Coverage of `ShapeChange` outside its documented smooth spline regime "
            "is a robustness comparison, not a claim that its assumptions are met. "
            "The logistic cell is retained as the smooth in-class check. Failures are "
            "counted as noncoverage. E33's theorem requires known Gaussian noise scale; "
            "this experiment does not establish an unknown-scale result or real-data utility.",
            "",
            "The confidence-set theorem does not depend on either comparator. Projection "
            "is a hybridization step: on every covered trial it cannot increase absolute "
            "error of the external point estimate.",
            "",
            f"Raw result directory: `{output_dir}`.",
        ]
    )
    report = PROJECT / "reports" / "hct" / f"e33_shape_contrast_hybrid_{config.name}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _json_safe(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def finalize(config: PhaseConfig, output_dir: Path) -> None:
    freeze = None
    if config.name == "confirmation":
        freeze = _load_and_validate_freeze(config, output_dir)
    _load_and_validate_pre_manifest(config, output_dir)
    rows = merge_scores(config, output_dir)
    summary = summarize(rows)
    gates = evaluate_gates(summary)
    _write_csv(output_dir / "trial_scores.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    make_figure(summary, output_dir / "comparison.png")
    report = write_report(config, summary, gates, output_dir)
    comparator_hashes = {
        str(index): _sha256(output_dir / f"comparators_shard_{index}.csv")
        for index in range(config.shards)
    }
    manifest = {
        "experiment": "E33 honest S-shaped inflection UQ and frontier hybrid",
        "status": (
            "confirmation_executed_after_freeze" if freeze else "smoke_completed"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "config": _config_payload(config),
        "freeze_sha256": (
            _sha256(output_dir / "frozen_config.json") if freeze else None
        ),
        "pre_manifest_sha256": _sha256(output_dir / "pre_manifest.json"),
        "code_hashes": _code_hashes(),
        "result_hashes": {
            "hct_scores": _sha256(output_dir / "hct_scores.csv"),
            "comparators": comparator_hashes,
            "trial_scores": _sha256(output_dir / "trial_scores.csv"),
            "summary": _sha256(output_dir / "summary.csv"),
            "report": _sha256(report),
        },
        "gates": gates,
        "summary": summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_safe({"output": str(output_dir), "gates": gates}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASES), default="smoke")
    parser.add_argument(
        "--stage", choices=("freeze", "prepare", "compare", "finalize"), required=True
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--rscript",
        type=Path,
        default=PROJECT / "third_party/r_runtime/R-4.6.1/bin/Rscript.exe",
    )
    args = parser.parse_args()
    config = PHASES[args.phase]
    output_dir = args.output_dir or (
        PROJECT / "results" / "hct" / f"shape_contrast_hybrid_e33_{config.name}"
    )
    if args.stage == "freeze":
        freeze_confirmation(config, output_dir)
    elif args.stage == "prepare":
        prepare(config, output_dir)
    elif args.stage == "compare":
        if args.shard_index is None:
            raise ValueError("--shard-index is required for comparator evaluation")
        run_comparator_shard(config, output_dir, args.shard_index, args.rscript)
    else:
        finalize(config, output_dir)


if __name__ == "__main__":
    main()
