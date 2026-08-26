#!/usr/bin/env python3
"""Nested-CV audit of the post-confirmation compact HCRD candidate."""

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
CANDIDATE_C = (1.0, 0.1, 0.01, 0.001)


def _compact_geometry(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if values.shape[1] != 4 * 297:
        raise ValueError("unexpected TargetedMSQC geometry width")
    views = values.reshape(values.shape[0], 4, 297)
    selected = [0]
    for level in range(8):
        selected.extend(range(1 + 37 * level, 1 + 37 * level + 13))
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


def _inner_score(
    features: NDArray[np.float32],
    labels: NDArray[np.uint8],
    groups: NDArray[Any],
    c: float,
) -> float:
    predictions = np.full(labels.size, np.nan)
    for held_out in sorted(np.unique(groups).tolist(), key=str):
        test = groups == held_out
        if np.unique(labels[test]).size != 2 or np.unique(labels[~test]).size != 2:
            continue
        estimator = _model(c)
        estimator.fit(features[~test], labels[~test])
        predictions[test] = estimator.predict_proba(features[test])[:, 1]
    valid = np.isfinite(predictions)
    if np.unique(labels[valid]).size != 2:
        return -np.inf
    return float(average_precision_score(labels[valid], predictions[valid]))


def _nested_scores(
    features: NDArray[np.float32],
    labels: NDArray[np.uint8],
    groups: NDArray[Any],
) -> tuple[NDArray[np.float64], dict[str, dict[str, Any]]]:
    scores = np.full(labels.size, np.nan)
    selections: dict[str, dict[str, Any]] = {}
    for outer in sorted(np.unique(groups).tolist(), key=str):
        test = groups == outer
        train = ~test
        inner = {
            c: _inner_score(features[train], labels[train], groups[train], c)
            for c in CANDIDATE_C
        }
        # Prefer stronger regularization on an exact AP tie.
        selected = sorted(inner, key=lambda c: (-inner[c], c))[0]
        estimator = _model(selected)
        estimator.fit(features[train], labels[train])
        scores[test] = estimator.predict_proba(features[test])[:, 1]
        selections[str(outer)] = {
            "selected_C": selected,
            "inner_ap": {str(c): value for c, value in inner.items()},
        }
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("nested CV left missing predictions")
    return scores, selections


def _metrics(labels: NDArray[np.uint8], scores: NDArray[np.float64]) -> dict[str, float]:
    return {
        "ap_ok": float(average_precision_score(labels, scores)),
        "ap_flag": float(average_precision_score(1 - labels, 1.0 - scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _weighted_ap(labels, scores, weights) -> float:
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    w = weights[order]
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]) + 1]
    group_weight = np.add.reduceat(w, starts)
    group_positive = np.add.reduceat(w * y, starts)
    total_positive = float(group_positive.sum())
    if total_positive == 0.0:
        return np.nan
    cumulative_weight = np.cumsum(group_weight)
    cumulative_positive = np.cumsum(group_positive)
    precision = np.divide(
        cumulative_positive,
        cumulative_weight,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_weight > 0,
    )
    return float(np.sum(group_positive * precision) / total_positive)


def _bootstrap_difference(labels, left, right, runs, peptides, replicates: int) -> dict[str, Any]:
    _, run_inverse = np.unique(runs, return_inverse=True)
    _, peptide_inverse = np.unique(peptides, return_inverse=True)
    nr = int(run_inverse.max()) + 1
    np_ = int(peptide_inverse.max()) + 1
    rng = np.random.default_rng(SEED)
    difference = np.empty(replicates)
    for index in range(replicates):
        rw = rng.multinomial(nr, np.full(nr, 1.0 / nr))
        pw = rng.multinomial(np_, np.full(np_, 1.0 / np_))
        weights = (rw[run_inverse] * pw[peptide_inverse]).astype(float)
        difference[index] = _weighted_ap(labels, left, weights) - _weighted_ap(
            labels, right, weights
        )
    finite = difference[np.isfinite(difference)]
    return {
        "difference": float(average_precision_score(labels, left) - average_precision_score(labels, right)),
        "descriptive_two_way_cluster_95_ci": np.quantile(finite, [0.025, 0.975]).tolist(),
        "finite_replicates": int(finite.size),
    }


def analyze(feature_dir: Path, output: Path, bootstrap: int) -> None:
    rows = pd.read_csv(feature_dir / "rows.csv")
    labels = rows["label_ok"].to_numpy(dtype=np.uint8)
    runs = rows["FileName"].astype(str).to_numpy()
    peptide_names = sorted(rows["PeptideModifiedSequence"].astype(str).unique())
    peptide_fold = {name: index % 5 for index, name in enumerate(peptide_names)}
    peptide_groups = rows["PeptideModifiedSequence"].map(peptide_fold).to_numpy()
    peptide_ids = rows["PeptideModifiedSequence"].astype(str).to_numpy()
    with np.load(feature_dir / "features.npz") as cache:
        qc = cache["qc52"].copy()
        compact = _compact_geometry(cache["hcrd_geometry"])
    methods = {
        "qc52": qc,
        "compact_hcrd": compact,
        "qc52_plus_compact_hcrd": np.c_[qc, compact],
    }
    run_scores: dict[str, NDArray[np.float64]] = {}
    result: dict[str, Any] = {
        "analysis": "post-confirmation method-development audit; candidate family chosen after E3",
        "candidate_C": CANDIDATE_C,
        "methods": {},
    }
    for name, features in methods.items():
        scores, selections = _nested_scores(features, labels, runs)
        peptide_scores, peptide_selections = _nested_scores(features, labels, peptide_groups)
        run_scores[name] = scores
        result["methods"][name] = {
            "feature_width": int(features.shape[1]),
            "run_transfer": _metrics(labels, scores),
            "run_outer_selections": selections,
            "peptide_transfer": _metrics(labels, peptide_scores),
            "peptide_outer_selections": peptide_selections,
            "per_run_ap": {
                run: float(average_precision_score(labels[runs == run], scores[runs == run]))
                for run in sorted(np.unique(runs).tolist())
            },
        }
        print(f"nested audit {name}", flush=True)
    result["hybrid_vs_qc52"] = _bootstrap_difference(
        labels,
        run_scores["qc52_plus_compact_hcrd"],
        run_scores["qc52"],
        runs,
        peptide_ids,
        bootstrap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    predictions = rows[["FileName", "PeptideModifiedSequence", "label_ok"]].copy()
    for name, score in run_scores.items():
        predictions[f"score_{name}"] = score
    predictions.to_csv(output.with_name("nested_predictions.csv"), index=False)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    analyze(args.feature_dir.resolve(), args.output.resolve(), args.bootstrap)


if __name__ == "__main__":
    main()
