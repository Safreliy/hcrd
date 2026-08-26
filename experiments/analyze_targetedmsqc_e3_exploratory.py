#!/usr/bin/env python3
"""Post-confirmation dimensionality audit for TargetedMSQC E3.

This script cannot alter or rescue the frozen E3 endpoint.  It diagnoses whether
the negative result is caused by redundant morphology or by a p >> n hybrid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260825
BASE = ("raw64", "hcrd_1", "hcrd_8", "hcrd_geometry", "area_only")
REGULARIZATION = (1.0, 0.1, 0.01, 0.001)


def _compact_geometry(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Discard ranked-lobe coordinates but retain every level summary.

    Each row has four EIC views in the frozen order
    light-short, light-context, heavy-short, heavy-context.  A view contains one
    depth scalar followed by eight blocks of 37 values; the first 13 entries of
    each block are permutation-invariant level summaries.
    """

    if values.shape[1] != 4 * 297:
        raise ValueError("unexpected TargetedMSQC geometry width")
    views = values.reshape(values.shape[0], 4, 297)
    selected = [0]
    for level in range(8):
        start = 1 + 37 * level
        selected.extend(range(start, start + 13))
    return views[:, :, selected].reshape(values.shape[0], -1)


def _model(c: float):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
    )


def _cv(
    features: NDArray[np.float32],
    labels: NDArray[np.uint8],
    groups: NDArray[Any],
    c: float,
) -> NDArray[np.float64]:
    scores = np.full(labels.size, np.nan)
    for held_out in sorted(np.unique(groups).tolist(), key=str):
        test = groups == held_out
        estimator = _model(c)
        estimator.fit(features[~test], labels[~test])
        scores[test] = estimator.predict_proba(features[test])[:, 1]
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("incomplete exploratory CV")
    return scores


def _metrics(labels: NDArray[np.uint8], scores: NDArray[np.float64]) -> dict[str, float]:
    return {
        "ap_ok": float(average_precision_score(labels, scores)),
        "ap_flag": float(average_precision_score(1 - labels, 1.0 - scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def analyze(feature_dir: Path, output: Path) -> None:
    rows = pd.read_csv(feature_dir / "rows.csv")
    labels = rows["label_ok"].to_numpy(dtype=np.uint8)
    runs = rows["FileName"].astype(str).to_numpy()
    peptide_names = sorted(rows["PeptideModifiedSequence"].astype(str).unique())
    peptide_fold = {name: index % 5 for index, name in enumerate(peptide_names)}
    peptides = rows["PeptideModifiedSequence"].map(peptide_fold).to_numpy()
    with np.load(feature_dir / "features.npz") as cache:
        arrays = {name: cache[name].copy() for name in ("qc52", *BASE)}
    arrays["compact_geometry"] = _compact_geometry(arrays["hcrd_geometry"])
    arrays["compact_geometry_area"] = np.c_[arrays["compact_geometry"], arrays["area_only"]]

    configurations: dict[str, tuple[NDArray[np.float32], float]] = {}
    for c in REGULARIZATION:
        configurations[f"qc52_C{c:g}"] = (arrays["qc52"], c)
        for name in (*BASE, "compact_geometry", "compact_geometry_area"):
            configurations[f"{name}_C{c:g}"] = (arrays[name], c)
            configurations[f"qc52_plus_{name}_C{c:g}"] = (
                np.c_[arrays["qc52"], arrays[name]],
                c,
            )

    run_metrics: dict[str, dict[str, float]] = {}
    peptide_metrics: dict[str, dict[str, float]] = {}
    per_run_ap: dict[str, dict[str, float]] = {}
    run_predictions: dict[str, NDArray[np.float64]] = {}
    for index, (name, (features, c)) in enumerate(configurations.items(), start=1):
        run_score = _cv(features, labels, runs, c)
        peptide_score = _cv(features, labels, peptides, c)
        run_predictions[name] = run_score
        run_metrics[name] = _metrics(labels, run_score)
        peptide_metrics[name] = _metrics(labels, peptide_score)
        per_run_ap[name] = {
            run: float(average_precision_score(labels[runs == run], run_score[runs == run]))
            for run in sorted(np.unique(runs).tolist())
        }
        print(f"configuration {index}/{len(configurations)} {name}", flush=True)

    qc_reference = run_metrics["qc52_C1"]
    ranked_run = sorted(
        run_metrics,
        key=lambda name: (-run_metrics[name]["ap_ok"], name),
    )
    ranked_peptide = sorted(
        peptide_metrics,
        key=lambda name: (-peptide_metrics[name]["ap_ok"], name),
    )
    result = {
        "analysis": "post-confirmation exploratory dimensionality audit; not E3 rescue",
        "configuration_count": len(configurations),
        "reference_qc52_C1": qc_reference,
        "top_10_run_transfer": [
            {
                "name": name,
                **run_metrics[name],
                "ap_difference_vs_qc52_C1": run_metrics[name]["ap_ok"]
                - qc_reference["ap_ok"],
                "per_run_ap": per_run_ap[name],
            }
            for name in ranked_run[:10]
        ],
        "top_10_peptide_transfer": [
            {"name": name, **peptide_metrics[name]}
            for name in ranked_peptide[:10]
        ],
        "run_transfer_metrics": run_metrics,
        "peptide_transfer_metrics": peptide_metrics,
        "feature_widths": {name: int(value.shape[1]) for name, value in arrays.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prediction_frame = rows[["FileName", "PeptideModifiedSequence", "label_ok"]].copy()
    for name in ranked_run[:10]:
        prediction_frame[f"score_{name}"] = run_predictions[name]
    prediction_frame.to_csv(output.with_name("exploratory_predictions_top10.csv"), index=False)
    print(json.dumps({
        "top_10_run_transfer": result["top_10_run_transfer"],
        "top_10_peptide_transfer": result["top_10_peptide_transfer"],
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.feature_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
