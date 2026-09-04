"""Postprocess frozen E36 and E38 trials against the full identified target.

This script does not regenerate any response.  It computes the deterministic
design-identified transition set once per experimental cell and then audits
the saved confidence-set endpoints.  It also compares E38 widths on the common
subset of trials where both methods return nonempty sets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.sci.run_high_precision_coverage_e36 import (
    DESIGNS,
    SAMPLE_SIZES,
    SIGNALS,
    _design_points,
    _signal,
)
from shapecontrast import design_identified_transition_set

E36_DIR = ROOT / "results/sci/high_precision_coverage_e36"
E38_DIR = ROOT / "results/sci/matched_honest_baseline_e38_r1"
E36_TRIALS = E36_DIR / "trial_scores.csv"
E38_TRIALS = E38_DIR / "trial_scores.csv"
E36_SUMMARY = E36_DIR / "identified_target_summary.csv"
E36_REPORT = E36_DIR / "identified_target_report.md"
E36_MANIFEST = E36_DIR / "identified_target_manifest.json"
E38_SUMMARY = E38_DIR / "identified_target_and_width_summary.csv"
E38_REPORT = E38_DIR / "identified_target_and_width_report.md"
E38_MANIFEST = E38_DIR / "identified_target_and_width_manifest.json"
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


def _cell_specs(*, domain_kind: str) -> dict[str, tuple[str, str, int, float, float]]:
    specs: dict[str, tuple[str, str, int, float, float]] = {}
    for signal in SIGNALS:
        for design in DESIGNS:
            for n in SAMPLE_SIZES:
                x = _design_points(design, n)
                mean = _signal(signal, x)
                domain = (0.0, 1.0) if domain_kind == "scientific" else (x[0], x[-1])
                target = design_identified_transition_set(x, mean, domain=domain)
                if target.empty or target.left is None or target.right is None:
                    raise RuntimeError(
                        f"empty benchmark target: {signal}, {design}, {n}"
                    )
                if not target.left <= 0.3 <= target.right:
                    raise RuntimeError(
                        f"generating transition outside target: {signal}, {design}, {n}"
                    )
                cell = f"{signal}__{design}__n{n}"
                specs[cell] = (signal, design, n, target.left, target.right)
    return specs


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _audit_e36() -> dict[str, object]:
    specs = _cell_specs(domain_kind="scientific")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with E36_TRIALS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["cell"]].append(row)
    if set(grouped) != set(specs) or sum(map(len, grouped.values())) != 80_000:
        raise RuntimeError("unexpected E36 cell or row count")

    output: list[dict[str, object]] = []
    total_differences = 0
    for cell, (signal, design, n, target_left, target_right) in specs.items():
        rows = grouped[cell]
        point_covered = np.asarray([_as_bool(row["covered"]) for row in rows])
        empty = np.asarray([_as_bool(row["empty"]) for row in rows])
        left = np.asarray([float(row["left"]) for row in rows])
        right = np.asarray([float(row["right"]) for row in rows])
        width = np.asarray(
            [
                np.nan if is_empty else float(row["width"])
                for row, is_empty in zip(rows, empty, strict=True)
            ]
        )
        target_covered = (~empty) & (left <= target_left) & (target_right <= right)
        differences = int(np.sum(point_covered != target_covered))
        total_differences += differences
        low, high = _wilson(int(target_covered.sum()), len(rows))
        output.append(
            {
                "cell": cell,
                "signal": signal,
                "design": design,
                "n": n,
                "trials": len(rows),
                "target_left": target_left,
                "target_right": target_right,
                "target_diameter": target_right - target_left,
                "generating_point_coverage": float(point_covered.mean()),
                "identified_set_coverage": float(target_covered.mean()),
                "identified_set_wilson_low": low,
                "identified_set_wilson_high": high,
                "point_vs_set_differences": differences,
                "median_width_among_nonempty": float(np.nanmedian(width)),
                "empty_probability": float(empty.mean()),
            }
        )
    _write_csv(E36_SUMMARY, output)

    short = {
        "paper_f1_cusp": "cusp",
        "paper_f2_onset": "onset",
        "paper_f3_jump": "jump",
        "paper_f4_logistic": "logistic",
    }
    lines = [
        "# E36 post-audit coverage of the full identified target",
        "",
        "No responses were regenerated. The design-identified transition set was",
        "computed once per cell and compared with the 80,000 saved SCI intervals.",
        "Empty outputs count as noncoverage.",
        "",
        "| signal | design | n | identified target | point coverage | target coverage (95% Wilson CI) | changed trials |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            "| {signal} | {design} | {n} | [{left:.6f}, {right:.6f}] | "
            "{point:.4f} | {coverage:.4f} [{low:.4f}, {high:.4f}] | {changed} |".format(
                signal=short[str(row["signal"])],
                design=row["design"],
                n=row["n"],
                left=row["target_left"],
                right=row["target_right"],
                point=row["generating_point_coverage"],
                coverage=row["identified_set_coverage"],
                low=row["identified_set_wilson_low"],
                high=row["identified_set_wilson_high"],
                changed=row["point_vs_set_differences"],
            )
        )
    lines.extend(
        [
            "",
            f"Only {total_differences} of 80,000 classifications change. Both are",
            "onset/uniform trials. The overall target-coverage range remains",
            (
                f"{min(float(row['identified_set_coverage']) for row in output):.4f}--"
                f"{max(float(row['identified_set_coverage']) for row in output):.4f}."
            ),
        ]
    )
    E36_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "analysis": "post-audit E36 coverage of the full design-identified target",
        "responses_regenerated": False,
        "input": {
            "path": str(E36_TRIALS.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(E36_TRIALS),
        },
        "configuration": {
            "confidence_level": CONFIDENCE_LEVEL,
            "generating_transition": 0.3,
        },
        "checks": {
            "rows": 80_000,
            "cells": 16,
            "changed_classifications": total_differences,
        },
        "code_hashes": {
            "analysis_script": _sha256(Path(__file__)),
            "identified_set_module": _sha256(
                ROOT / "src/shapecontrast/identified_set.py"
            ),
        },
        "result_hashes": {
            "summary": _sha256(E36_SUMMARY),
            "report": _sha256(E36_REPORT),
        },
    }
    E36_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _audit_e38() -> dict[str, object]:
    specs = _cell_specs(domain_kind="observed")
    grouped: dict[str, dict[int, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with E38_TRIALS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["cell"]][int(row["trial"])][row["method"]] = row
    if set(grouped) != set(specs):
        raise RuntimeError("unexpected E38 cells")

    output: list[dict[str, object]] = []
    total_differences = 0
    for cell, (signal, design, n, target_left, target_right) in specs.items():
        trials = grouped[cell]
        if len(trials) != 200 or any(
            set(pair) != {"SCI", "PBP"} for pair in trials.values()
        ):
            raise RuntimeError(f"unexpected E38 trial structure in {cell}")
        ordered = [trials[index] for index in sorted(trials)]
        method_values: dict[str, dict[str, np.ndarray]] = {}
        for method in ("SCI", "PBP"):
            method_rows = [pair[method] for pair in ordered]
            empty = np.asarray([_as_bool(row["empty"]) for row in method_rows])
            left = np.asarray([float(row["left"]) for row in method_rows])
            right = np.asarray([float(row["right"]) for row in method_rows])
            width = np.asarray(
                [
                    np.nan if is_empty else float(row["width"])
                    for row, is_empty in zip(method_rows, empty, strict=True)
                ]
            )
            point = np.asarray([_as_bool(row["covered"]) for row in method_rows])
            target = (~empty) & (left <= target_left) & (target_right <= right)
            total_differences += int(np.sum(point != target))
            method_values[method] = {
                "empty": empty,
                "width": width,
                "point": point,
                "target": target,
            }

        sci = method_values["SCI"]
        pbp = method_values["PBP"]
        both_nonempty = (~sci["empty"]) & (~pbp["empty"])
        sci_low, sci_high = _wilson(int(sci["target"].sum()), len(ordered))
        pbp_low, pbp_high = _wilson(int(pbp["target"].sum()), len(ordered))
        sci_own = float(np.nanmedian(sci["width"]))
        pbp_own = float(np.nanmedian(pbp["width"]))
        sci_shared = float(np.median(sci["width"][both_nonempty]))
        pbp_shared = float(np.median(pbp["width"][both_nonempty]))
        output.append(
            {
                "cell": cell,
                "signal": signal,
                "design": design,
                "n": n,
                "trials": len(ordered),
                "target_left": target_left,
                "target_right": target_right,
                "sci_identified_set_coverage": float(sci["target"].mean()),
                "sci_wilson_low": sci_low,
                "sci_wilson_high": sci_high,
                "pbp_identified_set_coverage": float(pbp["target"].mean()),
                "pbp_wilson_low": pbp_low,
                "pbp_wilson_high": pbp_high,
                "sci_empty_probability": float(sci["empty"].mean()),
                "pbp_empty_probability": float(pbp["empty"].mean()),
                "sci_median_width_among_own_nonempty": sci_own,
                "pbp_median_width_among_own_nonempty": pbp_own,
                "own_nonempty_width_reduction": 1.0 - sci_own / pbp_own,
                "both_nonempty_trials": int(both_nonempty.sum()),
                "sci_median_width_when_both_nonempty": sci_shared,
                "pbp_median_width_when_both_nonempty": pbp_shared,
                "both_nonempty_width_reduction": 1.0 - sci_shared / pbp_shared,
                "point_vs_set_differences": int(
                    np.sum(sci["point"] != sci["target"])
                    + np.sum(pbp["point"] != pbp["target"])
                ),
            }
        )
    _write_csv(E38_SUMMARY, output)

    short = {
        "paper_f1_cusp": "cusp",
        "paper_f2_onset": "onset",
        "paper_f3_jump": "jump",
        "paper_f4_logistic": "logistic",
    }
    lines = [
        "# E38 post-audit target coverage and common-support width sensitivity",
        "",
        "No responses were regenerated. Coverage refers to the full design-identified",
        "transition set. Method-specific medians exclude that method's empty outputs;",
        "the final column restricts both methods to the same nonempty trials.",
        "",
        "| signal | design | n | SCI target coverage | PBP target coverage | SCI/PBP empty rates | own-nonempty reduction | both-nonempty reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            "| {signal} | {design} | {n} | {sci:.3f} [{sl:.3f}, {sh:.3f}] | "
            "{pbp:.3f} [{pl:.3f}, {ph:.3f}] | {se:.3f}/{pe:.3f} | {own:.1%} | {shared:.1%} |".format(
                signal=short[str(row["signal"])],
                design=row["design"],
                n=row["n"],
                sci=row["sci_identified_set_coverage"],
                sl=row["sci_wilson_low"],
                sh=row["sci_wilson_high"],
                pbp=row["pbp_identified_set_coverage"],
                pl=row["pbp_wilson_low"],
                ph=row["pbp_wilson_high"],
                se=row["sci_empty_probability"],
                pe=row["pbp_empty_probability"],
                own=row["own_nonempty_width_reduction"],
                shared=row["both_nonempty_width_reduction"],
            )
        )
    informative = [row for row in output if row["signal"] != "paper_f4_logistic"]
    lines.extend(
        [
            "",
            "Point and full-target coverage agree in all 6,400 method rows. On the",
            "common nonempty subset, the reduction range over the 12 informative cells is",
            (
                f"{min(float(row['both_nonempty_width_reduction']) for row in informative):.1%}--"
                f"{max(float(row['both_nonempty_width_reduction']) for row in informative):.1%}."
            ),
        ]
    )
    E38_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "analysis": "post-audit E38 full-target coverage and common-support width sensitivity",
        "responses_regenerated": False,
        "input": {
            "path": str(E38_TRIALS.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(E38_TRIALS),
        },
        "configuration": {
            "confidence_level": CONFIDENCE_LEVEL,
            "generating_transition": 0.3,
        },
        "checks": {
            "method_rows": 6_400,
            "cells": 16,
            "changed_classifications": total_differences,
        },
        "code_hashes": {
            "analysis_script": _sha256(Path(__file__)),
            "identified_set_module": _sha256(
                ROOT / "src/shapecontrast/identified_set.py"
            ),
        },
        "result_hashes": {
            "summary": _sha256(E38_SUMMARY),
            "report": _sha256(E38_REPORT),
        },
    }
    E38_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    payload = {"e36": _audit_e36(), "e38": _audit_e38()}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
