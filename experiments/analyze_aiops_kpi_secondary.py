"""Post-hoc secondary comparisons for the completed AIOps C3 study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "aiops_kpi_c3"


def bootstrap(difference: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(50_000)
    for start in range(0, 50_000, 1000):
        indices = rng.integers(0, len(difference), (1000, len(difference)))
        means[start : start + 1000] = difference[indices].mean(axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    data = pd.read_csv(OUTPUT / "all_metrics.csv")
    data = data[data["primary_sparse_transient"]]
    comparisons = [
        ("raw_SR", "raw_sr_auc_pr"),
        ("raw_absolute_deviation", "raw_abs_auc_pr"),
    ]
    rows: list[dict[str, object]] = []
    for index, (name, column) in enumerate(comparisons):
        difference = data["hcrd_direct_auc_pr"].to_numpy() - data[column].to_numpy()
        low, high = bootstrap(difference, 20260826 + index)
        nonzero = difference[np.abs(difference) > 1e-12]
        rows.append(
            {
                "analysis_status": "post_hoc_secondary_ablation",
                "baseline": name,
                "series": len(difference),
                "direct_hcrd_mean_auc_pr": float(data["hcrd_direct_auc_pr"].mean()),
                "baseline_mean_auc_pr": float(data[column].mean()),
                "mean_difference": float(difference.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "paired_t_p": float(ttest_1samp(difference, 0.0).pvalue),
                "wilcoxon_p": float(wilcoxon(difference).pvalue),
                "exact_sign_p": float(
                    binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue
                ),
                "wins": int(np.sum(difference > 1e-12)),
                "ties": int(np.sum(np.abs(difference) <= 1e-12)),
                "losses": int(np.sum(difference < -1e-12)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "posthoc_direct_hcrd_comparisons.csv", index=False)
    payload = {
        "status": "post_hoc_secondary_ablation",
        "warning": "not the frozen C3 primary candidate or endpoint",
        "comparisons": rows,
    }
    (OUTPUT / "posthoc_direct_hcrd_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
