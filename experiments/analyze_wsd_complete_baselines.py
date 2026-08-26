"""Analyse C2 using exactly the baseline family eligible at comparator freeze.

This is an execution-only amendment: the frozen endpoint, strata, comparator,
paired statistics, seeds, and multiplicity rule are imported unchanged from the
prospectively frozen runner. Partial official wrappers remain auditable but are
not treated as losses.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "wsd_sparse_transient_c2"


def _protocol():
    path = ROOT / "experiments" / "run_wsd_sparse_transient_confirmation.py"
    spec = importlib.util.spec_from_file_location("wsd_c2_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen C2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    protocol = _protocol()
    comparator_payload = json.loads(
        (OUTPUT / "comparator_frozen.json").read_text(encoding="utf-8")
    )
    comparator = str(comparator_payload["primary_comparator"])
    eligible = [
        method
        for method, item in comparator_payload["baseline_completeness"].items()
        if item["eligible"]
    ]
    if comparator not in eligible or len(eligible) < 4:
        raise RuntimeError("frozen eligible family is invalid")
    manifest = pd.read_csv(OUTPUT / "population_manifest.csv")
    merged = manifest.copy()
    for method in [*eligible, "HCRD"]:
        metrics = protocol.read_method_metrics(OUTPUT, method)
        good = metrics[metrics["status"] == "ok"]
        if len(good) != len(manifest):
            raise RuntimeError(
                f"eligible method {method} has {len(good)}/{len(manifest)} series"
            )
        columns = ["file", "auc_pr", "auc_roc", "vus_pr", "vus_roc"]
        renamed = good[columns].rename(
            columns={name: f"{method}_{name}" for name in columns if name != "file"}
        )
        merged = merged.merge(renamed, on="file", how="left", validate="1:1")
    merged.to_csv(OUTPUT / "all_metrics.csv", index=False)

    comparisons: list[dict[str, object]] = []
    groups = [
        ("primary_sparse_transient", merged[merged["primary_sparse_transient"]]),
        ("all_WSD", merged),
        ("outside_primary", merged[~merged["primary_sparse_transient"]]),
    ]
    for stratum_name, group in groups:
        for index, method in enumerate(eligible):
            item = protocol.paired_comparison(
                group,
                method,
                seed=20260825 + index + 100 * len(stratum_name),
            )
            item["stratum"] = stratum_name
            item["is_frozen_primary_comparison"] = bool(
                stratum_name == "primary_sparse_transient" and method == comparator
            )
            comparisons.append(item)
    table = pd.DataFrame(comparisons)
    for stratum in table["stratum"].unique():
        mask = table["stratum"] == stratum
        for source, target in [
            ("paired_t_p", "holm_paired_t_p"),
            ("wilcoxon_p", "holm_wilcoxon_p"),
            ("exact_sign_p", "holm_exact_sign_p"),
        ]:
            table.loc[mask, target] = protocol.holm_adjust(
                table.loc[mask, source].astype(float).tolist()
            )
    table.to_csv(OUTPUT / "paired_comparisons.csv", index=False)

    primary = merged[merged["primary_sparse_transient"]].copy()
    delta = primary["HCRD_vus_pr"] - primary[f"{comparator}_vus_pr"]
    modifiers: list[dict[str, object]] = []
    for name in ["max_run_fraction", "anomaly_occupancy"]:
        estimate, p_value = spearmanr(primary[name], delta)
        modifiers.append(
            {
                "stratum": "primary_sparse_transient",
                "baseline": comparator,
                "modifier": name,
                "spearman_rho": float(estimate),
                "p_value_exploratory": float(p_value),
            }
        )
    pd.DataFrame(modifiers).to_csv(OUTPUT / "effect_modifiers.csv", index=False)

    completeness_rows = []
    for method, item in comparator_payload["baseline_completeness"].items():
        completeness_rows.append({"method": method, **item})
    pd.DataFrame(completeness_rows).to_csv(
        OUTPUT / "baseline_completeness.csv", index=False
    )
    primary_row = table[
        (table["stratum"] == "primary_sparse_transient")
        & (table["baseline"] == comparator)
    ].iloc[0]
    payload = {
        "status": "complete_with_frozen_eligible_baseline_family",
        "primary_comparator": comparator,
        "eligible_complete_baselines": eligible,
        "incomplete_wrappers_not_counted_as_losses": [
            method
            for method in comparator_payload["baseline_completeness"]
            if method not in eligible
        ],
        "primary_series": int(len(primary)),
        "full_series": int(len(merged)),
        "primary_result": primary_row.to_dict(),
        "analysis_amendment": (
            "The frozen runner expected all eight wrappers at export time, while "
            "the frozen comparator rule only required at least four complete "
            "methods. This exporter applies the unchanged frozen statistics to "
            "the four methods marked eligible in comparator_frozen.json."
        ),
        "timing_claim": "none; diagnostic wall time excluded",
        "files": {
            "all_metrics": "all_metrics.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "effect_modifiers": "effect_modifiers.csv",
            "baseline_completeness": "baseline_completeness.csv",
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
