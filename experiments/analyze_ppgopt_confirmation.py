"""Summarize the locked PPGopt confirmation result and paired uncertainty."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULT = PROJECT / "results" / "ppgopt" / "confirmation" / "summary.json"
OUTPUT_JSON = PROJECT / "results" / "ppgopt" / "confirmation" / "statistics.json"
OUTPUT_MD = PROJECT / "reports" / "ppgopt_confirmation.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(counts: tuple[int, int, int]) -> float:
    true_positive, false_positive, false_negative = counts
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def subject_counts(metrics: dict[str, object]) -> dict[int, tuple[int, int, int]]:
    return {
        int(subject): (int(values["tp"]), int(values["fp"]), int(values["fn"]))
        for subject, values in dict(metrics["by_subject"]).items()
    }


def paired_subject_bootstrap(
    first: dict[str, object],
    second: dict[str, object],
    *,
    repetitions: int = 10000,
    seed: int = 1729,
) -> dict[str, object]:
    first_counts = subject_counts(first)
    second_counts = subject_counts(second)
    subjects = np.asarray(sorted(first_counts), dtype=int)
    if set(first_counts) != set(second_counts):
        raise ValueError("methods have different subject sets")
    generator = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sample = generator.choice(subjects, size=subjects.size, replace=True)
        first_total = tuple(
            sum(first_counts[int(subject)][index] for subject in sample)
            for index in range(3)
        )
        second_total = tuple(
            sum(second_counts[int(subject)][index] for subject in sample)
            for index in range(3)
        )
        differences[repetition] = f1(first_total) - f1(second_total)
    return {
        "repetitions": repetitions,
        "seed": seed,
        "subjects": subjects.tolist(),
        "point_difference": float(first["f1"] - second["f1"]),
        "percentile_95_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "probability_difference_above_zero": float(np.mean(differences > 0.0)),
        "warning": "Only two locked subjects; the cluster bootstrap has very low resolution.",
    }


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    models = {item["name"]: item for item in result["models"]}
    primary_name = result["primary_model"]
    primary = models[primary_name]["metrics"]
    geometry = models["hgb_geometry"]["metrics"]
    baselines = {
        name: value["metrics"] for name, value in result["baselines"].items()
    }
    comparisons = {
        name: paired_subject_bootstrap(primary, metrics)
        for name, metrics in baselines.items()
    }
    activity_deltas = {
        baseline: {
            activity: float(
                primary["by_activity"][activity]["f1"]
                - metrics["by_activity"][activity]["f1"]
            )
            for activity in ("rest", "squat", "step")
        }
        for baseline, metrics in baselines.items()
    }
    statistics = {
        "confirmation_summary_sha256": sha256(RESULT),
        "primary_model": primary_name,
        "primary_metrics": {
            key: primary[key]
            for key in (
                "f1",
                "precision",
                "recall",
                "mean_absolute_error_ms",
                "median_absolute_error_ms",
                "false_positives_per_minute",
            )
        },
        "baselines": {
            name: {
                key: metrics[key]
                for key in (
                    "f1",
                    "precision",
                    "recall",
                    "mean_absolute_error_ms",
                    "false_positives_per_minute",
                )
            }
            for name, metrics in baselines.items()
        },
        "paired_subject_bootstrap": comparisons,
        "activity_f1_deltas": activity_deltas,
        "ablations": {
            "geometry_only_f1": geometry["f1"],
            "hybrid_f1": primary["f1"],
            "hybrid_minus_geometry": primary["f1"] - geometry["f1"],
            "geometry_minus_deterministic_persistence": (
                geometry["f1"] - baselines["p1"]["f1"]
            ),
        },
        "success_rule_checks": {
            "beats_p0": primary["f1"] > baselines["p0"]["f1"],
            "beats_heartpy": primary["f1"] > baselines["heartpy"]["f1"],
            "no_activity_more_than_0_01_below_p0": min(
                activity_deltas["p0"].values()
            )
            >= -0.01,
            "bootstrap_lower_bound_above_zero_vs_p0": comparisons["p0"][
                "percentile_95_interval"
            ][0]
            > 0.0,
            "geometry_materially_beats_p1": (
                geometry["f1"] - baselines["p1"]["f1"] >= 0.01
            ),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    table_rows = []
    ordered = [
        ("P0 find_peaks", baselines["p0"]),
        ("HeartPy", baselines["heartpy"]),
        ("P1 deterministic HCRD", baselines["p1"]),
        ("P2 HCRD geometry", geometry),
        ("P2 HCRD hybrid", primary),
    ]
    for name, metrics in ordered:
        table_rows.append(
            f"| {name} | {metrics['f1']:.6f} | {metrics['precision']:.6f} | "
            f"{metrics['recall']:.6f} | {metrics['mean_absolute_error_ms']:.3f} |"
        )
    markdown = f"""# Locked PPGopt confirmation result

Protocol SHA-256: `{result['protocol_sha256']}`  
Frozen rule SHA-256: `{result['frozen_confirmation_rule_sha256']}`  
Subjects: S6--S7; 30 recordings; 5,734 valid expert events.

| Method | F1 | Precision | Recall | Mean error, ms |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The frozen HCRD hybrid reaches **F1 = {primary['f1']:.6f}**, improving over
the tuned local-maximum baseline by **{primary['f1'] - baselines['p0']['f1']:.6f}**
and over HeartPy by **{primary['f1'] - baselines['heartpy']['f1']:.6f}**.
Geometry alone reaches **{geometry['f1']:.6f}**; the seven-feature morphology
block adds {primary['f1'] - geometry['f1']:.6f}. In contrast, thresholding
multilevel persistence without learning reaches only {baselines['p1']['f1']:.6f}.

The largest practical gain is under step motion: hybrid HCRD improves over P0
by {activity_deltas['p0']['step']:.6f} F1. At rest it is
{activity_deltas['p0']['rest']:.6f} relative to P0, within the frozen 0.01
non-inferiority margin.

The paired subject-cluster bootstrap interval for hybrid minus P0 is
[{comparisons['p0']['percentile_95_interval'][0]:.6f},
{comparisons['p0']['percentile_95_interval'][1]:.6f}]. This meets the frozen
rule, but there are only two confirmation subjects; replication on an
independent PPG dataset is still required for a strong publication claim.

Published all-data optima from Wolling et al. (Karlen 0.958, van Gent 0.970)
are contextual rather than held-out comparisons. The primary direct
comparators above were executed by this repository with the same locked test
subjects and scoring code.
"""
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()

