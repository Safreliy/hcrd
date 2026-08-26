"""Compute deterministic review-queue utility from the frozen E5 predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_COLUMNS = {
    "qscore": "bad_score_qscore",
    "DOMAIN+Q": "bad_score_domain_q",
    "HCRD-1+Q": "bad_score_hcrd_1_q",
    "HCRD-8+Q": "bad_score_hcrd_8_q",
    "HCRD geometry+Q": "bad_score_hcrd_geometry_q",
    "Area/energy+Q": "bad_score_area_only_q",
}


def review_utility(frame: pd.DataFrame, budgets: tuple[float, ...]) -> dict:
    bad = frame["label"].eq("Bad").to_numpy()
    total_bad = int(bad.sum())
    n = len(frame)
    output: dict[str, object] = {
        "n": n,
        "bad": total_bad,
        "budgets": list(budgets),
        "models": {},
    }

    for name, column in MODEL_COLUMNS.items():
        scores = frame[column].to_numpy(dtype=float)
        # Stable sorting makes equal-score handling reproducible from CSV order.
        order = np.argsort(-scores, kind="mergesort")
        ranked_bad = bad[order]
        rows = []
        for budget in budgets:
            k = int(math.ceil(budget * n))
            captured = int(ranked_bad[:k].sum())
            precision = captured / k
            rows.append(
                {
                    "budget_fraction": budget,
                    "reviewed": k,
                    "bad_captured": captured,
                    "recall_bad": captured / total_bad,
                    "precision_bad": precision,
                    "bad_per_100_reviews": 100.0 * precision,
                    "reviews_per_bad_found": None if captured == 0 else k / captured,
                }
            )

        recall_targets = {}
        cumulative = np.cumsum(ranked_bad)
        for target in (0.25, 0.50, 0.75):
            required_bad = int(math.ceil(target * total_bad))
            hit = np.flatnonzero(cumulative >= required_bad)
            reviewed = int(hit[0] + 1) if hit.size else n
            recall_targets[str(target)] = {
                "required_bad": required_bad,
                "reviewed": reviewed,
                "workload_fraction": reviewed / n,
                "workload_reduction_vs_full_review": 1.0 - reviewed / n,
            }
        output["models"][name] = {
            "budget_metrics": rows,
            "fixed_recall_workload": recall_targets,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/pttime_e5/evaluation/predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pttime_e5/evaluation/review_utility.json"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    result = review_utility(frame, (0.01, 0.05, 0.10))
    result["source_predictions"] = args.predictions.as_posix()
    result["ranking"] = "descending bad score; stable CSV-order tie handling"
    result["scope"] = (
        "Conditional utility within the 365 source-model-selected, "
        "unambiguous Pttime features."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
