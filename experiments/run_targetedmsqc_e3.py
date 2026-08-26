#!/usr/bin/env python3
"""Frozen E3 TargetedMSQC run-transfer confirmation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hcrd.lcms import eic_pair_feature_bank


PROTOCOL = "hcrd-e3-targetedmsqc-v1"
SEED = 20260825
SOURCE_COMMIT = "5439ba9f102d043241ccb8207e7ec9f1a35ebc63"
KEY = [
    "FileName",
    "PeptideModifiedSequence",
    "PrecursorCharge",
    "FragmentIon",
    "ProductCharge",
]
QC_ID_COLUMNS = ["File", *KEY]
BASE_REPRESENTATIONS = (
    "raw64",
    "hcrd_1",
    "hcrd_8",
    "hcrd_geometry",
    "area_only",
)
MODEL_REPRESENTATIONS = (
    "qc52",
    "qc52_raw",
    "qc52_hcrd1",
    "qc52_hcrd8",
    "qc52_geometry",
    "qc52_area",
)
BASE_WIDTHS = {
    "raw64": 300,
    "hcrd_1": 708,
    "hcrd_8": 3792,
    "hcrd_geometry": 1188,
    "area_only": 192,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_trace(times: str, intensities: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.fromstring(str(times), sep=",", dtype=float)
    y = np.fromstring(str(intensities), sep=",", dtype=float)
    if x.size != y.size:
        raise ValueError("time/intensity length mismatch")
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _crop_windows(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    start: float,
    end: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("invalid peak boundary")
    width = end - start
    short = (x >= start) & (x <= end)
    context = (x >= start - width) & (x <= end + width)
    if np.unique(x[short]).size < 8:
        raise ValueError("short_window_lt8")
    if np.unique(x[context]).size < 8:
        raise ValueError("context_window_lt8")
    return y[short], x[short], y[context], x[context]


def _extract_isotope(
    times: str,
    intensities: str,
    start: float,
    end: float,
) -> dict[str, NDArray[np.float32]]:
    x, y = _parse_trace(times, intensities)
    short_y, short_x, context_y, context_x = _crop_windows(x, y, start, end)
    bank = eic_pair_feature_bank(short_y, short_x, context_y, context_x)
    return {name: getattr(bank, name) for name in BASE_REPRESENTATIONS}


def _extract_row(task: dict[str, Any]) -> tuple[dict[str, NDArray[np.float32]] | None, str | None]:
    try:
        light = _extract_isotope(
            task["light_Times"],
            task["light_Intensities"],
            task["start"],
            task["end"],
        )
        heavy = _extract_isotope(
            task["heavy_Times"],
            task["heavy_Intensities"],
            task["start"],
            task["end"],
        )
        result = {
            name: np.concatenate([light[name], heavy[name]]).astype(np.float32)
            for name in BASE_REPRESENTATIONS
        }
        for name, width in BASE_WIDTHS.items():
            if result[name].shape != (width,) or not np.all(np.isfinite(result[name])):
                raise RuntimeError(f"invalid_{name}")
        return result, None
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return None, str(error)


def _deduplicate_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "File Name": "FileName",
            "Peptide Modified Sequence": "PeptideModifiedSequence",
            "Precursor Charge": "PrecursorCharge",
            "Min Start Time": "start",
            "Max End Time": "end",
        }
    )
    boundary_key = ["FileName", "PeptideModifiedSequence", "PrecursorCharge"]
    for _, group in renamed.groupby(boundary_key, sort=False):
        if group[["start", "end"]].drop_duplicates().shape[0] != 1:
            raise ValueError("conflicting duplicate peak boundaries")
    return renamed.drop_duplicates(boundary_key)[boundary_key + ["start", "end"]]


def _source_paths(repository: Path) -> dict[str, Path]:
    panel = repository / "inst" / "extdata" / "CSF_Panel"
    return {
        "labels": panel / "Training" / "CSF_Biomarkers_training_annotated.csv",
        "features": panel / "Features" / "features.csv",
        "boundaries": panel / "Peak_boundary" / "CSF_Biomarkers.csv",
        "chromatograms": panel / "Chromatograms" / "CSF_Biomarkers.tsv",
    }


def extract_features(repository: Path, output_dir: Path, workers: int) -> None:
    paths = _source_paths(repository)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TargetedMSQC files: {missing}")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"expected source commit {SOURCE_COMMIT}, found {commit}")

    labels = pd.read_csv(paths["labels"])
    official = pd.read_csv(paths["features"])
    boundaries = _deduplicate_boundaries(pd.read_csv(paths["boundaries"]))
    chromatograms = pd.read_csv(paths["chromatograms"], sep="\t")

    if labels.duplicated(KEY).any() or official.duplicated(KEY).any():
        raise ValueError("duplicate label or official-feature key")
    chromatogram_key = [*KEY, "IsotopeLabelType"]
    if chromatograms.duplicated(chromatogram_key).any():
        raise ValueError("duplicate chromatogram/isotope key")
    qc_columns = [column for column in official.columns if column not in QC_ID_COLUMNS]
    if len(qc_columns) != 52:
        raise ValueError(f"expected 52 official metrics, found {len(qc_columns)}")

    merged = labels.merge(
        official[KEY + qc_columns], on=KEY, how="left", indicator="official_match"
    )
    exclusion_counts: dict[str, int] = {
        "missing_official_features": int((merged["official_match"] != "both").sum())
    }
    merged = merged.loc[merged["official_match"] == "both"].drop(columns="official_match")
    boundary_key = ["FileName", "PeptideModifiedSequence", "PrecursorCharge"]
    merged = merged.merge(boundaries, on=boundary_key, how="left", indicator="boundary_match")
    exclusion_counts["missing_boundary"] = int((merged["boundary_match"] != "both").sum())
    merged = merged.loc[merged["boundary_match"] == "both"].drop(columns="boundary_match")

    trace_columns = chromatogram_key + ["Times", "Intensities"]
    for isotope in ("light", "heavy"):
        trace = chromatograms.loc[
            chromatograms["IsotopeLabelType"].eq(isotope), trace_columns
        ].drop(columns="IsotopeLabelType")
        merged = merged.merge(
            trace.rename(
                columns={
                    "Times": f"{isotope}_Times",
                    "Intensities": f"{isotope}_Intensities",
                }
            ),
            on=KEY,
            how="left",
            indicator=f"{isotope}_match",
        )
        exclusion_counts[f"missing_{isotope}_chromatogram"] = int(
            (merged[f"{isotope}_match"] != "both").sum()
        )
        merged = merged.loc[merged[f"{isotope}_match"] == "both"].drop(
            columns=f"{isotope}_match"
        )

    tasks = [
        {
            "light_Times": row.light_Times,
            "light_Intensities": row.light_Intensities,
            "heavy_Times": row.heavy_Times,
            "heavy_Intensities": row.heavy_Intensities,
            "start": float(row.start),
            "end": float(row.end),
        }
        for row in merged.itertuples(index=False)
    ]
    if workers == 1:
        extracted = map(_extract_row, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        extracted = executor.map(_extract_row, tasks, chunksize=8)
    feature_rows: list[dict[str, NDArray[np.float32]]] = []
    keep: list[int] = []
    signal_exclusions: dict[str, int] = {}
    try:
        for index, (features, reason) in enumerate(extracted):
            if features is None:
                key = reason or "unknown_signal_error"
                signal_exclusions[key] = signal_exclusions.get(key, 0) + 1
            else:
                keep.append(index)
                feature_rows.append(features)
            if (index + 1) % 50 == 0 or index + 1 == len(tasks):
                print(f"HCRD extraction {index + 1}/{len(tasks)}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown()
    exclusion_counts.update({f"signal:{key}": value for key, value in signal_exclusions.items()})
    merged = merged.iloc[keep].reset_index(drop=True)
    if not feature_rows:
        raise RuntimeError("no eligible TargetedMSQC rows")

    arrays: dict[str, NDArray[np.float32]] = {
        name: np.stack([row[name] for row in feature_rows]).astype(np.float32)
        for name in BASE_REPRESENTATIONS
    }
    arrays["qc52"] = merged[qc_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "features.npz", **arrays)
    rows = merged[[*KEY, "Status", "Notes"]].rename(columns={"Status": "status", "Notes": "notes"})
    rows["label_ok"] = rows["status"].str.casefold().eq("ok").astype(np.uint8)
    if rows["label_ok"].nunique() != 2:
        raise RuntimeError("eligible data lack one expert class")
    rows.to_csv(output_dir / "rows.csv", index=False)
    (output_dir / "qc_feature_names.txt").write_text(
        "\n".join(qc_columns) + "\n", encoding="utf-8"
    )
    metadata = {
        "protocol": PROTOCOL,
        "frozen_source_commit": SOURCE_COMMIT,
        "source_commit": commit,
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "annotated_count": int(labels.shape[0]),
        "eligible_count": int(rows.shape[0]),
        "ok_count": int(rows["label_ok"].sum()),
        "flag_count": int((1 - rows["label_ok"]).sum()),
        "run_count": int(rows["FileName"].nunique()),
        "peptide_count": int(rows["PeptideModifiedSequence"].nunique()),
        "exclusion_counts": exclusion_counts,
        "qc_feature_count": len(qc_columns),
        "base_feature_widths": {name: int(value.shape[1]) for name, value in arrays.items()},
        "windows": {
            "short": "[start,end]",
            "context": "[start-(end-start),end+(end-start)] clipped to trace",
        },
        "isotope_order": ["light", "heavy"],
        "workers": workers,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def _model():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
    )


def _model_arrays(cache: Any) -> dict[str, NDArray[np.float32]]:
    qc = cache["qc52"]
    return {
        "qc52": qc,
        "qc52_raw": np.c_[qc, cache["raw64"]],
        "qc52_hcrd1": np.c_[qc, cache["hcrd_1"]],
        "qc52_hcrd8": np.c_[qc, cache["hcrd_8"]],
        "qc52_geometry": np.c_[qc, cache["hcrd_geometry"]],
        "qc52_area": np.c_[qc, cache["area_only"]],
    }


def _cross_validated_scores(
    arrays: dict[str, NDArray[np.float32]],
    labels: NDArray[np.uint8],
    groups: NDArray[Any],
) -> tuple[dict[str, NDArray[np.float64]], list[Any]]:
    values = sorted(np.unique(groups).tolist(), key=str)
    scores = {name: np.full(labels.size, np.nan) for name in arrays}
    for fold, held_out in enumerate(values, start=1):
        test = groups == held_out
        train = ~test
        if np.unique(labels[train]).size != 2 or np.unique(labels[test]).size != 2:
            raise RuntimeError(f"fold {held_out!r} lacks one class")
        for name, features in arrays.items():
            estimator = _model()
            estimator.fit(features[train], labels[train])
            scores[name][test] = estimator.predict_proba(features[test])[:, 1]
        print(f"fold {fold}/{len(values)} held out {held_out}", flush=True)
    if any(not np.all(np.isfinite(value)) for value in scores.values()):
        raise RuntimeError("cross-validation left nonfinite predictions")
    return scores, values


def _metrics(labels: NDArray[np.uint8], scores: NDArray[np.float64]) -> dict[str, float]:
    return {
        "average_precision_ok": float(average_precision_score(labels, scores)),
        "average_precision_flag": float(average_precision_score(1 - labels, 1.0 - scores)),
        "roc_auc_ok": float(roc_auc_score(labels, scores)),
    }


def _weighted_ap_preparation(scores: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]) + 1]
    return order, starts


def _weighted_ap(
    labels: NDArray[np.uint8],
    weights: NDArray[np.float64],
    order: NDArray[np.int64],
    starts: NDArray[np.int64],
) -> float:
    sorted_weight = weights[order]
    sorted_positive = sorted_weight * labels[order]
    group_weight = np.add.reduceat(sorted_weight, starts)
    group_positive = np.add.reduceat(sorted_positive, starts)
    total_positive = float(np.sum(group_positive))
    if total_positive <= 0.0:
        return np.nan
    cumulative_weight = np.cumsum(group_weight)
    cumulative_positive = np.cumsum(group_positive)
    precision = np.divide(
        cumulative_positive,
        cumulative_weight,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_weight > 0.0,
    )
    return float(np.sum(group_positive * precision) / total_positive)


def _two_way_cluster_bootstrap(
    labels: NDArray[np.uint8],
    scores: dict[str, NDArray[np.float64]],
    run_groups: NDArray[Any],
    peptide_groups: NDArray[Any],
    replicates: int,
) -> dict[str, NDArray[np.float64]]:
    _, run_inverse = np.unique(run_groups, return_inverse=True)
    _, peptide_inverse = np.unique(peptide_groups, return_inverse=True)
    run_count = int(run_inverse.max()) + 1
    peptide_count = int(peptide_inverse.max()) + 1
    preparations = {name: _weighted_ap_preparation(value) for name, value in scores.items()}
    output = {name: np.empty(replicates) for name in scores}
    rng = np.random.default_rng(SEED)
    for replicate in range(replicates):
        run_weights = rng.multinomial(run_count, np.full(run_count, 1.0 / run_count))
        peptide_weights = rng.multinomial(
            peptide_count, np.full(peptide_count, 1.0 / peptide_count)
        )
        weights = (run_weights[run_inverse] * peptide_weights[peptide_inverse]).astype(float)
        for name, (order, starts) in preparations.items():
            output[name][replicate] = _weighted_ap(labels, weights, order, starts)
        if (replicate + 1) % 1000 == 0:
            print(f"bootstrap {replicate + 1}/{replicates}", flush=True)
    return output


def _paired_comparison(
    primary: str,
    comparator: str,
    metrics: dict[str, dict[str, float]],
    boot: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    difference = boot[primary] - boot[comparator]
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        raise RuntimeError("all bootstrap contrasts are nonfinite")
    p_value = min(
        1.0,
        2.0
        * min(
            (np.sum(finite <= 0.0) + 1.0) / (finite.size + 1.0),
            (np.sum(finite >= 0.0) + 1.0) / (finite.size + 1.0),
        ),
    )
    return {
        "primary": primary,
        "comparator": comparator,
        "ap_difference": float(
            metrics[primary]["average_precision_ok"]
            - metrics[comparator]["average_precision_ok"]
        ),
        "two_way_cluster_bootstrap_95_ci": np.quantile(finite, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(p_value),
        "finite_bootstrap_replicates": int(finite.size),
    }


def _note_strata(
    rows: pd.DataFrame,
    labels: NDArray[np.uint8],
    scores: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    notes = rows["notes"].fillna("").str.casefold()
    flag = labels == 0
    morphology = flag & notes.str.contains(
        "shoulder|jagged|bimodal|tailing|high background", regex=True
    ).to_numpy()
    ratio = flag & notes.str.contains("inconsistent", regex=False).to_numpy()
    output: dict[str, Any] = {}
    for stratum, subtype in (("morphology", morphology), ("ratio_or_order", ratio)):
        selected = (labels == 1) | subtype
        output[stratum] = {
            "flag_count": int(subtype.sum()),
            "ok_count": int(labels.sum()),
            "flag_positive_ap": {
                name: float(
                    average_precision_score(1 - labels[selected], 1.0 - value[selected])
                )
                for name, value in scores.items()
            },
        }
    return output


def evaluate(feature_dir: Path, output_dir: Path, bootstrap: int) -> None:
    metadata_path = feature_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL or metadata.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("feature cache does not satisfy frozen E3 protocol")
    rows = pd.read_csv(feature_dir / "rows.csv")
    labels = rows["label_ok"].to_numpy(dtype=np.uint8)
    runs = rows["FileName"].astype(str).to_numpy()
    peptides = rows["PeptideModifiedSequence"].astype(str).to_numpy()
    with np.load(feature_dir / "features.npz") as cache:
        arrays = _model_arrays(cache)

    run_scores, run_order = _cross_validated_scores(arrays, labels, runs)
    pooled_metrics = {name: _metrics(labels, value) for name, value in run_scores.items()}
    per_run_metrics = {
        run: {
            name: _metrics(labels[runs == run], value[runs == run])
            for name, value in run_scores.items()
        }
        for run in run_order
    }
    peptide_names = sorted(np.unique(peptides).tolist())
    peptide_fold = {name: index % 5 for index, name in enumerate(peptide_names)}
    peptide_groups = np.asarray([peptide_fold[name] for name in peptides])
    peptide_scores, _ = _cross_validated_scores(arrays, labels, peptide_groups)
    peptide_metrics = {name: _metrics(labels, value) for name, value in peptide_scores.items()}

    boot = _two_way_cluster_bootstrap(labels, run_scores, runs, peptides, bootstrap)
    primary = _paired_comparison("qc52_hcrd8", "qc52", pooled_metrics, boot)
    secondary_comparisons = {
        name: _paired_comparison(name, "qc52", pooled_metrics, boot)
        for name in MODEL_REPRESENTATIONS
        if name not in {"qc52", "qc52_hcrd8"}
    }
    run_differences = {
        run: per_run_metrics[run]["qc52_hcrd8"]["average_precision_ok"]
        - per_run_metrics[run]["qc52"]["average_precision_ok"]
        for run in run_order
    }
    run_wins = int(sum(value > 0.0 for value in run_differences.values()))
    ci_low = float(primary["two_way_cluster_bootstrap_95_ci"][0])
    success_components = {
        "positive_pooled_ap_difference": primary["ap_difference"] > 0.0,
        "positive_cluster_ci_lower": ci_low > 0.0,
        "wins_at_least_three_runs": run_wins >= 3,
        "hcrd8_exceeds_hcrd1": pooled_metrics["qc52_hcrd8"]["average_precision_ok"]
        > pooled_metrics["qc52_hcrd1"]["average_precision_ok"],
    }
    result = {
        "protocol": PROTOCOL,
        "prospective_primary_success": bool(all(success_components.values())),
        "success_components": success_components,
        "primary_comparison": primary,
        "held_out_run_wins": run_wins,
        "held_out_run_ap_differences": run_differences,
        "pooled_run_transfer_metrics": pooled_metrics,
        "per_run_metrics": per_run_metrics,
        "secondary_peptide_group_metrics": peptide_metrics,
        "secondary_comparisons": secondary_comparisons,
        "secondary_note_strata": _note_strata(rows, labels, run_scores),
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": SEED,
        "feature_metadata_sha256": _sha256(metadata_path),
        "feature_metadata": metadata,
        "learner": {
            "imputer": "median; keep_empty_features=True",
            "standardizer": "StandardScaler",
            "class": "LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": SEED,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    prediction_frame = rows[[*KEY, "status", "notes", "label_ok"]].copy()
    for name, value in run_scores.items():
        prediction_frame[f"run_cv_score_{name}"] = value
    for name, value in peptide_scores.items():
        prediction_frame[f"peptide_cv_score_{name}"] = value
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-features")
    extract.add_argument("--repository", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--workers", type=int, default=1)
    score = subparsers.add_parser("evaluate")
    score.add_argument("--feature-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "extract-features":
        if args.workers < 1:
            parser.error("--workers must be positive")
        extract_features(args.repository.resolve(), args.output_dir.resolve(), args.workers)
    else:
        if args.bootstrap < 1:
            parser.error("--bootstrap must be positive")
        evaluate(args.feature_dir.resolve(), args.output_dir.resolve(), args.bootstrap)


if __name__ == "__main__":
    main()
