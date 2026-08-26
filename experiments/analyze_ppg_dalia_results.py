"""Paired subject bootstrap and concise report for nested PPG-DaLiA results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results" / "ppg_dalia" / "nested_subjectwise_summary.json"
MOTION_GATE = PROJECT / "results" / "ppg_dalia" / "motion_gate_exploratory.json"
OUTPUT = PROJECT / "results" / "ppg_dalia" / "statistics.json"
REPORT = PROJECT / "reports" / "ppg_dalia_nested.md"
ACTIVITIES = (
    "car_driving",
    "cycling",
    "lunch_break",
    "sitting",
    "stair_climbing",
    "table_soccer",
    "walking",
    "working",
)
HARD = ("stair_climbing", "table_soccer", "walking")


def _records(report):
    return {str(item["key"]): item for item in report["per_record"]}


def _median(items):
    return float(np.median([float(item["f1"]) for item in items]))


def _micro_f1(items):
    tp = sum(int(item["tp"]) for item in items)
    fp = sum(int(item["fp"]) for item in items)
    fn = sum(int(item["fn"]) for item in items)
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _motion(items):
    medians = []
    for activity in HARD:
        relevant = [item for item in items if item["activity"] == activity]
        medians.append(_median(relevant))
    return float(np.mean(medians))


def _interval(values):
    return {
        "estimate": float(values[0]),
        "ci95_low": float(np.quantile(values[1], 0.025)),
        "ci95_high": float(np.quantile(values[1], 0.975)),
        "probability_positive": float(np.mean(values[1] > 0.0)),
    }


def _bootstrap(left_report, right_report, seed=1729, replicates=20000):
    left = _records(left_report)
    right = _records(right_report)
    if set(left) != set(right):
        raise RuntimeError("paired reports contain different records")
    subjects = sorted({str(item["subject"]) for item in left.values()})
    by_subject = {
        subject: sorted(
            [key for key, item in left.items() if item["subject"] == subject]
        )
        for subject in subjects
    }
    observed_left = list(left.values())
    observed_right = list(right.values())
    observed = np.asarray(
        [
            _median(observed_left) - _median(observed_right),
            _micro_f1(observed_left) - _micro_f1(observed_right),
            _motion(observed_left) - _motion(observed_right),
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 3), dtype=float)
    for replicate in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        left_items = []
        right_items = []
        for subject in sampled:
            keys = by_subject[str(subject)]
            left_items.extend(left[key] for key in keys)
            right_items.extend(right[key] for key in keys)
        draws[replicate] = (
            _median(left_items) - _median(right_items),
            _micro_f1(left_items) - _micro_f1(right_items),
            _motion(left_items) - _motion(right_items),
        )
    return {
        "subjects": len(subjects),
        "replicates": replicates,
        "median_record_f1_difference": _interval((observed[0], draws[:, 0])),
        "micro_f1_difference": _interval((observed[1], draws[:, 1])),
        "motion_macro_difference": _interval((observed[2], draws[:, 2])),
    }


def _fmt(value):
    return f"{float(value):.4f}"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    primary = data["cross_fitted_primary"]
    geometry = data["models"]["hgb_geometry"]
    mass = data["models"]["hgb_mass_control"]
    p0 = data["baselines"]["p0"]
    comparisons = {
        "primary_vs_p0": _bootstrap(primary, p0, seed=1729),
        "geometry_vs_p0": _bootstrap(geometry, p0, seed=1730),
        "primary_vs_mass": _bootstrap(primary, mass, seed=1731),
        "geometry_vs_mass": _bootstrap(geometry, mass, seed=1732),
    }
    exploratory = None
    if MOTION_GATE.exists():
        gate = json.loads(MOTION_GATE.read_text(encoding="utf-8"))["pooled"]
        exploratory = {
            "motion_gate_vs_p0": _bootstrap(gate, p0, seed=1733),
            "gate_metrics": {
                "median_record_f1": gate["median_record_f1"],
                "micro_f1": gate["micro_f1"],
                "motion_intensive_macro_median_f1": gate[
                    "motion_intensive_macro_median_f1"
                ],
            },
            "status": "post-outer-test exploratory; requires independent confirmation",
        }
    activity = {}
    for name, report in {
        "primary": primary,
        "geometry": geometry,
        "mass": mass,
        "p0": p0,
    }.items():
        activity[name] = {
            item: {
                "exact_median_f1": report["by_activity"][item][
                    "median_record_f1"
                ],
                "compatible_median_f1": report["by_activity"][item][
                    "compatible_median_record_f1"
                ],
            }
            for item in ACTIVITIES
        }
    primary_wins = sum(
        activity["primary"][item]["exact_median_f1"]
        > activity["p0"][item]["exact_median_f1"]
        for item in ACTIVITIES
    )
    hard_wins = sum(
        activity["primary"][item]["exact_median_f1"]
        > activity["p0"][item]["exact_median_f1"]
        for item in HARD
    )
    result = {
        "source": str(SOURCE.relative_to(PROJECT)).replace("\\", "/"),
        "comparisons": comparisons,
        "activity": activity,
        "primary_activity_wins_vs_p0": int(primary_wins),
        "primary_hard_activity_wins_vs_p0": int(hard_wins),
        "predeclared_success": {
            "overall_median_and_micro": bool(
                primary["median_record_f1"] > p0["median_record_f1"]
                and primary["micro_f1"] > p0["micro_f1"]
            ),
            "at_least_five_activities": bool(primary_wins >= 5),
            "motion_macro": bool(
                primary["motion_intensive_macro_median_f1"]
                > p0["motion_intensive_macro_median_f1"]
            ),
            "full_geometry_over_mass": bool(
                geometry["median_record_f1"] > mass["median_record_f1"]
            ),
            "positive_subject_bootstrap_vs_p0": bool(
                comparisons["primary_vs_p0"]["median_record_f1_difference"][
                    "ci95_low"
                ]
                > 0.0
            ),
        },
        "exploratory_motion_gate": exploratory,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# PPG-DaLiA nested subject-wise HCRD result",
        "",
        "Author: Saveliy Baturin, Independent Researcher",
        "",
        "All values are cross-fitted outer-test results from the frozen five-fold subject protocol.",
        "",
        "| Method | Median exact F1 | Micro-F1 | Motion macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, report in (
        ("P0 find_peaks", p0),
        ("HCRD mass-only", mass),
        ("HCRD geometry", geometry),
        ("HCRD hybrid / primary", primary),
    ):
        lines.append(
            f"| {name} | {_fmt(report['median_record_f1'])} | "
            f"{_fmt(report['micro_f1'])} | "
            f"{_fmt(report['motion_intensive_macro_median_f1'])} |"
        )
    lines.extend(
        [
            "",
            "| Activity | Primary | Geometry | Mass-only | P0 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in ACTIVITIES:
        lines.append(
            f"| {item} | {_fmt(activity['primary'][item]['exact_median_f1'])} | "
            f"{_fmt(activity['geometry'][item]['exact_median_f1'])} | "
            f"{_fmt(activity['mass'][item]['exact_median_f1'])} | "
            f"{_fmt(activity['p0'][item]['exact_median_f1'])} |"
        )
    lines.extend(["", "## Paired subject bootstrap", ""])
    for name, comparison in comparisons.items():
        median = comparison["median_record_f1_difference"]
        motion = comparison["motion_macro_difference"]
        lines.append(
            f"- {name}: overall median difference {_fmt(median['estimate'])} "
            f"(95% CI {_fmt(median['ci95_low'])} to {_fmt(median['ci95_high'])}); "
            f"motion difference {_fmt(motion['estimate'])} "
            f"(95% CI {_fmt(motion['ci95_low'])} to {_fmt(motion['ci95_high'])})."
        )
    if exploratory is not None:
        comparison = exploratory["motion_gate_vs_p0"]
        median = comparison["median_record_f1_difference"]
        motion = comparison["motion_macro_difference"]
        lines.append(
            f"- exploratory motion_gate_vs_p0: overall median difference "
            f"{_fmt(median['estimate'])} (95% CI {_fmt(median['ci95_low'])} to "
            f"{_fmt(median['ci95_high'])}); motion difference "
            f"{_fmt(motion['estimate'])} (95% CI {_fmt(motion['ci95_low'])} to "
            f"{_fmt(motion['ci95_high'])}). This analysis is post-outer-test."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The predeclared broad success rule failed: HCRD does not beat P0 over all activities. "
            "The full hierarchy materially beats the mass-only representation. The primary HCRD "
            "model beats P0 on all three predeclared motion-intensive activities, but the subject "
            "bootstrap determines whether that narrower effect is sufficiently stable for a claim.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
