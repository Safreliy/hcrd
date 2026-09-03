"""Verify hashes frozen by the E33--E38 SCI experiment manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    failures: list[str] = []
    verified = 0

    def check(relative: str, expected: str) -> None:
        nonlocal verified
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif _digest(path) != expected:
            failures.append(f"sha256: {relative}")
        else:
            verified += 1

    e33_dir = ROOT / "results/hct/shape_contrast_hybrid_e33_confirmation"
    e33_frozen = _load(e33_dir / "frozen_config.json")
    e33_code = {
        "driver": "experiments/hct/run_shape_contrast_hybrid_e33.py",
        "r_bridge": "experiments/hct/run_shape_contrast_comparators_e33.R",
        "protocol": "docs/hct_e33_shape_contrast_inflection_protocol.md",
        "shape_module": "src/hcrd/shape_inflection_confidence.py",
    }
    for key, relative in e33_code.items():
        check(relative, e33_frozen["hashes"][key])

    e33_pre = _load(e33_dir / "pre_manifest.json")
    for key, filename in {
        "designs": "designs.csv",
        "hct_scores": "hct_scores.csv",
        "families": "families.csv",
    }.items():
        check(str((e33_dir / filename).relative_to(ROOT)), e33_pre["outputs"][key])
    omitted_observations = e33_pre["outputs"]["observation_shards"]
    if set(omitted_observations) != {"0", "1", "2", "3"}:
        failures.append("E33 observation-shard declarations are incomplete")

    e33 = _load(e33_dir / "manifest.json")
    check(
        str((e33_dir / "trial_scores.csv").relative_to(ROOT)),
        e33["result_hashes"]["trial_scores"],
    )
    check(
        str((e33_dir / "summary.csv").relative_to(ROOT)),
        e33["result_hashes"]["summary"],
    )
    check(
        "reports/hct/e33_shape_contrast_hybrid_confirmation.md",
        e33["result_hashes"]["report"],
    )
    for shard, expected in e33["result_hashes"]["comparators"].items():
        check(
            str((e33_dir / f"comparators_shard_{shard}.csv").relative_to(ROOT)),
            expected,
        )

    e34_dir = ROOT / "results/hct/unknown_scale_shape_inversion_e34_confirmation"
    e34 = _load(e34_dir / "manifest.json")
    e34_paths = {
        "driver": "experiments/hct/run_unknown_scale_confirmation_e34.py",
        "development_driver": "experiments/hct/run_unknown_scale_shape_inversion_e34.py",
        "protocol": "docs/hct_e34_unknown_scale_protocol.md",
        "scale_module": "src/hcrd/noise_scale_confidence.py",
        "shape_module": "src/hcrd/shape_inflection_confidence.py",
        "development_manifest": "results/hct/unknown_scale_shape_inversion_e34_development/manifest.json",
        "trial_scores": "results/hct/unknown_scale_shape_inversion_e34_confirmation/trial_scores.csv",
        "summary": "results/hct/unknown_scale_shape_inversion_e34_confirmation/summary.csv",
        "report": "reports/hct/e34_unknown_scale_confirmation.md",
    }
    for key, relative in e34_paths.items():
        check(relative, e34["hashes"][key])

    e35_dir = ROOT / "results/hct/heteroskedastic_shape_inversion_e35_confirmation"
    e35 = _load(e35_dir / "manifest.json")
    e35_paths = {
        "driver": "experiments/hct/run_heteroskedastic_shape_inversion_e35.py",
        "r_bridge": "experiments/hct/run_lidar_hybrid_e35.R",
        "protocol": "docs/hct_e35_heteroskedastic_protocol.md",
        "theory": "theory/hct/heteroskedastic_gaussian_extension.md",
        "heteroskedastic_module": "src/hcrd/heteroskedastic_scale_confidence.py",
        "shape_module": "src/hcrd/shape_inflection_confidence.py",
        "trial_scores.csv": "results/hct/heteroskedastic_shape_inversion_e35_confirmation/trial_scores.csv",
        "summary.csv": "results/hct/heteroskedastic_shape_inversion_e35_confirmation/summary.csv",
        "lidar_data_and_fit.csv": "results/hct/heteroskedastic_shape_inversion_e35_confirmation/lidar_data_and_fit.csv",
        "lidar_external_fits.csv": "results/hct/heteroskedastic_shape_inversion_e35_confirmation/lidar_external_fits.csv",
        "lidar_hct_sensitivity.csv": "results/hct/heteroskedastic_shape_inversion_e35_confirmation/lidar_hct_sensitivity.csv",
        "lidar_sensitivity.png": "results/hct/heteroskedastic_shape_inversion_e35_confirmation/lidar_sensitivity.png",
        "report": "reports/hct/e35_heteroskedastic_lidar.md",
    }
    for key, relative in e35_paths.items():
        check(relative, e35["hashes"][key])

    e36_dir = ROOT / "results/sci/high_precision_coverage_e36"
    e36 = _load(e36_dir / "manifest.json")
    if _load(e36_dir / "frozen_config.json") != e36["frozen"]:
        failures.append("E36 frozen_config.json differs from manifest")
    e36_code = {
        "driver": "experiments/sci/run_high_precision_coverage_e36.py",
        "protocol": "docs/sci_e36_high_precision_coverage_protocol.md",
        "inference_module": "src/shapecontrast/inference.py",
    }
    for key, relative in e36_code.items():
        check(relative, e36["frozen"]["hashes"][key])
    for key, filename in {
        "trial_scores": "trial_scores.csv",
        "summary": "summary.csv",
        "report": "report.md",
    }.items():
        check(
            str((e36_dir / filename).relative_to(ROOT)),
            e36["result_hashes"][key],
        )

    e37_dir = ROOT / "results/sci/dnase_replicate_e37"
    e37 = _load(e37_dir / "manifest.json")
    if _load(e37_dir / "frozen_config.json") != e37["frozen"]:
        failures.append("E37 frozen_config.json differs from manifest")
    e37_code = {
        "driver": "experiments/sci/run_dnase_replicate_e37.py",
        "protocol": "docs/sci_e37_dnase_replicate_protocol.md",
        "inference_module": "src/shapecontrast/inference.py",
        "replicate_module": "src/shapecontrast/replicated.py",
        "data": "data/external/dnase/DNase.csv",
    }
    for key, relative in e37_code.items():
        check(relative, e37["frozen"]["hashes"][key])
    for key, filename in {
        "replicate_curves": "replicate_curves.csv",
        "mean_curve": "mean_curve.csv",
        "contrast_table": "contrast_table.csv",
        "result": "result.json",
        "figure": "dnase_replicate_sci.png",
        "report": "report.md",
    }.items():
        check(
            str((e37_dir / filename).relative_to(ROOT)),
            e37["result_hashes"][key],
        )

    e38_dir = ROOT / "results/sci/matched_honest_baseline_e38_r1"
    e38 = _load(e38_dir / "manifest.json")
    if _load(e38_dir / "frozen_config.json") != e38["frozen"]:
        failures.append("E38r1 frozen_config.json differs from manifest")
    e38_code = {
        "driver": "experiments/sci/run_matched_honest_baseline_e38.py",
        "protocol": "docs/sci_e38_matched_honest_baseline_protocol.md",
        "inference_module": "src/shapecontrast/inference.py",
        "projection_module": "src/shapecontrast/projection.py",
    }
    for key, relative in e38_code.items():
        check(relative, e38["frozen"]["hashes"][key])
    for key, filename in {
        "trial_scores": "trial_scores.csv",
        "summary": "summary.csv",
        "report": "report.md",
    }.items():
        check(
            str((e38_dir / filename).relative_to(ROOT)),
            e38["result_hashes"][key],
        )

    if failures:
        raise SystemExit("SCI artifact verification failed:\n" + "\n".join(failures))
    print(
        f"verified {verified} frozen SCI files; "
        "4 deterministic E33 response shards are hash-declared but excluded"
    )


if __name__ == "__main__":
    main()
