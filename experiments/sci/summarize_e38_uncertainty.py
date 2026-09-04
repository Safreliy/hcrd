"""Add post-audit uncertainty intervals to the frozen E38 comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/sci/matched_honest_baseline_e38_r1"
TRIAL_PATH = RESULT_DIR / "trial_scores.csv"
SUMMARY_PATH = RESULT_DIR / "uncertainty_summary.csv"
REPORT_PATH = RESULT_DIR / "uncertainty_report.md"
MANIFEST_PATH = RESULT_DIR / "uncertainty_manifest.json"
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20_260_904
CONFIDENCE_LEVEL = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"unexpected Boolean value: {value!r}")


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + CONFIDENCE_LEVEL / 2.0)
    estimate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (estimate + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * sqrt(estimate * (1.0 - estimate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return center - radius, center + radius


def _bootstrap_reduction(
    sci_width: np.ndarray,
    pbp_width: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    trials = sci_width.size
    reductions: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPETITIONS
    while remaining:
        chunk = min(1_000, remaining)
        indices = rng.integers(0, trials, size=(chunk, trials))
        sci_median = np.nanmedian(sci_width[indices], axis=1)
        pbp_median = np.nanmedian(pbp_width[indices], axis=1)
        reductions.append(1.0 - sci_median / pbp_median)
        remaining -= chunk
    values = np.concatenate(reductions)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    cells: dict[str, dict[int, dict[str, tuple[bool, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with TRIAL_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            empty = _as_bool(row["empty"])
            width = float("nan") if empty else float(row["width"])
            cells[row["cell"]][int(row["trial"])][row["method"]] = (
                _as_bool(row["covered"]),
                width,
            )

    seed_sequences = np.random.SeedSequence(BOOTSTRAP_SEED).spawn(len(cells))
    output: list[dict[str, object]] = []
    for cell_index, (cell, by_trial) in enumerate(cells.items()):
        trials = sorted(by_trial)
        if any(set(by_trial[trial]) != {"SCI", "PBP"} for trial in trials):
            raise ValueError(f"incomplete method pair in {cell}")
        sci_covered = np.asarray(
            [by_trial[trial]["SCI"][0] for trial in trials], dtype=bool
        )
        pbp_covered = np.asarray(
            [by_trial[trial]["PBP"][0] for trial in trials], dtype=bool
        )
        sci_width = np.asarray(
            [by_trial[trial]["SCI"][1] for trial in trials], dtype=float
        )
        pbp_width = np.asarray(
            [by_trial[trial]["PBP"][1] for trial in trials], dtype=float
        )
        sci_low, sci_high = _wilson(int(sci_covered.sum()), len(trials))
        pbp_low, pbp_high = _wilson(int(pbp_covered.sum()), len(trials))
        reduction = 1.0 - float(np.nanmedian(sci_width)) / float(
            np.nanmedian(pbp_width)
        )
        boot_low, boot_high = _bootstrap_reduction(
            sci_width,
            pbp_width,
            rng=np.random.default_rng(seed_sequences[cell_index]),
        )
        signal, design, n_text = cell.split("__")
        output.append(
            {
                "cell": cell,
                "signal": signal,
                "design": design,
                "n": int(n_text[1:]),
                "trials": len(trials),
                "sci_coverage": float(sci_covered.mean()),
                "sci_wilson_low": sci_low,
                "sci_wilson_high": sci_high,
                "pbp_coverage": float(pbp_covered.mean()),
                "pbp_wilson_low": pbp_low,
                "pbp_wilson_high": pbp_high,
                "median_width_reduction": reduction,
                "bootstrap_reduction_low": boot_low,
                "bootstrap_reduction_high": boot_high,
            }
        )

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)

    short = {
        "paper_f1_cusp": "Cusp",
        "paper_f2_onset": "Onset",
        "paper_f3_jump": "Jump",
        "paper_f4_logistic": "Logistic",
    }
    lines = [
        "# Post-audit uncertainty summary for E38r1",
        "",
        "The 200 frozen responses per cell were not rerun. Coverage intervals are",
        "95% Wilson intervals. Width-reduction intervals are paired percentile",
        f"bootstrap intervals with {BOOTSTRAP_REPETITIONS:,} resamples and seed",
        f"{BOOTSTRAP_SEED}. Empty sets count as noncoverage and are excluded from",
        "the method-specific median-width calculation, matching the frozen summary.",
        "These intervals are descriptive post-audit analyses, not pre-specified gates.",
        "",
        "| Signal | Design | n | SCI coverage (95% CI) | PBP coverage (95% CI) | Median-width reduction (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            "| {signal} | {design} | {n} | {sci:.3f} [{sci_low:.3f}, {sci_high:.3f}] "
            "| {pbp:.3f} [{pbp_low:.3f}, {pbp_high:.3f}] | "
            "{reduction:.1%} [{low:.1%}, {high:.1%}] |".format(
                signal=short[str(row["signal"])],
                design=row["design"],
                n=row["n"],
                sci=row["sci_coverage"],
                sci_low=row["sci_wilson_low"],
                sci_high=row["sci_wilson_high"],
                pbp=row["pbp_coverage"],
                pbp_low=row["pbp_wilson_low"],
                pbp_high=row["pbp_wilson_high"],
                reduction=row["median_width_reduction"],
                low=row["bootstrap_reduction_low"],
                high=row["bootstrap_reduction_high"],
            )
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "analysis": "post-audit E38r1 uncertainty intervals",
        "input": {
            "path": str(TRIAL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(TRIAL_PATH),
        },
        "configuration": {
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "empty_set_width_handling": "excluded from method-specific medians",
        },
        "script_sha256": _sha256(Path(__file__)),
        "result_hashes": {
            "uncertainty_summary": _sha256(SUMMARY_PATH),
            "uncertainty_report": _sha256(REPORT_PATH),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
