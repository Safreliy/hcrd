"""Freeze and run the AIOps2018 phase-2 sparse-KPI confirmation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
DATA_REPOSITORY = REPOSITORY_ROOT / "third_party" / "KPI-Anomaly-Detection"
PRELIMINARY = DATA_REPOSITORY / "Preliminary_dataset" / "train.csv"
CONFIRMATION = (
    DATA_REPOSITORY
    / "Finals_dataset"
    / "extracted_train"
    / "phase2_train.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "aiops_kpi_c3"
sys.path.insert(0, str(ROOT / "src"))

PRELIMINARY_SHA256 = "427fff1a9f310e96f8e146a640c69f3045bcdcabA6d52cacd6db91b3a6dea273".lower()
CONFIRMATION_SHA256 = "4807dc9f1f6df31e0688b47734e52b6249ef7680840837a99800bfec6393331d"
REPOSITORY_COMMIT = "d06bda15d511d930cbf4e6a6de14bd94d790f0f2"
MAX_RUN_FRACTION = 0.005
MAX_OCCUPANCY = 0.01
AMPLITUDE_WINDOW = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def longest_run(labels: np.ndarray) -> int:
    changes = np.flatnonzero(np.diff(np.pad(labels.astype(int), (1, 1))))
    return int(np.max(changes[1::2] - changes[::2], initial=0))


def load_groups(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, usecols=["timestamp", "value", "label", "KPI ID"])
    return {
        str(identifier): group.sort_values("timestamp").reset_index(drop=True)
        for identifier, group in frame.groupby("KPI ID", sort=True)
    }


def manifest(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for identifier, group in groups.items():
        labels = group["label"].to_numpy(dtype=int)
        maximum_run = longest_run(labels)
        occupancy = float(np.mean(labels))
        run_fraction = maximum_run / len(labels)
        canonical = group[["timestamp", "value", "label"]].to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8")
        rows.append(
            {
                "kpi_id": identifier,
                "length": int(len(group)),
                "anomaly_points": int(np.sum(labels)),
                "anomaly_occupancy": occupancy,
                "max_run": maximum_run,
                "max_run_fraction": run_fraction,
                "primary_sparse_transient": bool(
                    run_fraction <= MAX_RUN_FRACTION
                    and occupancy <= MAX_OCCUPANCY
                ),
                "canonical_series_sha256": hashlib.sha256(canonical).hexdigest(),
            }
        )
    return pd.DataFrame(rows).sort_values("kpi_id").reset_index(drop=True)


def freeze(output: Path) -> None:
    if sha256(PRELIMINARY) != PRELIMINARY_SHA256:
        raise RuntimeError("preliminary file hash mismatch")
    if sha256(CONFIRMATION) != CONFIRMATION_SHA256:
        raise RuntimeError("phase-2 file hash mismatch")
    if any((output / "scores").glob("*.csv")):
        raise RuntimeError("refusing to freeze after confirmation scores exist")
    preliminary_manifest = manifest(load_groups(PRELIMINARY))
    confirmation_manifest = manifest(load_groups(CONFIRMATION))
    if len(preliminary_manifest) != 26 or int(preliminary_manifest["primary_sparse_transient"].sum()) != 15:
        raise RuntimeError("unexpected preliminary population")
    if len(confirmation_manifest) != 29 or int(confirmation_manifest["primary_sparse_transient"].sum()) != 13:
        raise RuntimeError("unexpected confirmation population")
    output.mkdir(parents=True, exist_ok=True)
    preliminary_manifest.to_csv(output / "preliminary_manifest.csv", index=False)
    confirmation_manifest.to_csv(output / "confirmation_manifest.csv", index=False)
    runner = Path(__file__).resolve()
    protocol = ROOT / "docs" / "aiops_kpi_c3_protocol.md"
    development = ROOT / "results" / "aiops_kpi_development" / "summary.json"
    development_payload = json.loads(development.read_text(encoding="utf-8"))
    if development_payload["selected_candidate"] != "a2_area_sr":
        raise RuntimeError("unexpected development winner")
    payload: dict[str, object] = {
        "status": "frozen_before_any_phase2_score",
        "dataset": "2018 AIOps KPI Anomaly Detection, phase2_train release",
        "dataset_nature": "real production KPIs labelled by domain experts",
        "primary_class": "max_run/length <= 0.005 and anomaly_occupancy <= 0.01",
        "class_threshold_origin": "inherited unchanged from WSD C2 before AIOps labels were inspected",
        "preliminary_development": {
            "all_series": 26,
            "primary_series": 15,
            "candidate_family": "fixed hcrd_temporal_candidate_scores family",
            "selection_metric": "mean per-series AUC-PR",
            "selected_candidate": "HCRD L8-max area-density followed by spectral residual",
            "selected_mean_auc_pr": development_payload["mean_auc_pr"]["a2_area_sr"],
            "direct_hcrd_mean_auc_pr": development_payload["mean_auc_pr"]["a2_direct"],
            "development_only": True,
        },
        "confirmation": {"all_series": 29, "primary_series": 13},
        "primary_endpoint": "paired mean per-series AUC-PR difference: HCRD-area-SR minus raw-signal-SR on 13 primary KPIs",
        "candidate": {
            "hcrd_levels": 8,
            "aggregation": "max",
            "spectral_residual_amplitude_window": AMPLITUDE_WINDOW,
        },
        "primary_comparator": "same spectral-residual operator applied to the raw KPI",
        "secondary": ["all 29 KPIs", "direct HCRD ablation", "raw absolute-deviation ablation", "AUC-ROC"],
        "inference": ["50000-draw paired bootstrap CI", "exact paired sign-flip test on n=13", "paired t-test", "Wilcoxon", "exact sign test"],
        "timing_policy": "no runtime claim",
        "preliminary_sha256": sha256(PRELIMINARY),
        "confirmation_sha256": sha256(CONFIRMATION),
        "repository_commit": REPOSITORY_COMMIT,
        "development_summary_sha256": sha256(development),
        "implementation_sha256": {
            runner.relative_to(ROOT).as_posix(): sha256(runner),
            protocol.relative_to(ROOT).as_posix(): sha256(protocol),
            "src/hcrd/anomaly.py": sha256(ROOT / "src" / "hcrd" / "anomaly.py"),
            "src/hcrd/energy.py": sha256(ROOT / "src" / "hcrd" / "energy.py"),
            "src/hcrd/temporal_anomaly.py": sha256(ROOT / "src" / "hcrd" / "temporal_anomaly.py"),
        },
    }
    atomic_json(output / "frozen_configuration.json", payload)
    print(json.dumps(payload, indent=2))


def evaluate_baseline(output: Path) -> None:
    if not (output / "frozen_configuration.json").exists():
        raise RuntimeError("freeze before baseline evaluation")
    if (output / "scores" / "hcrd.csv").exists():
        raise RuntimeError("refusing baseline evaluation after HCRD")
    from hcrd.temporal_anomaly import spectral_residual_score

    rows: list[dict[str, object]] = []
    for identifier, group in load_groups(CONFIRMATION).items():
        signal = group["value"].to_numpy(dtype=float)
        labels = group["label"].to_numpy(dtype=int)
        raw_sr = spectral_residual_score(signal, amplitude_window=AMPLITUDE_WINDOW)
        raw_abs = np.abs(signal - np.median(signal))
        rows.append(
            {
                "kpi_id": identifier,
                "raw_sr_auc_pr": float(average_precision_score(labels, raw_sr)),
                "raw_sr_auc_roc": float(roc_auc_score(labels, raw_sr)),
                "raw_abs_auc_pr": float(average_precision_score(labels, raw_abs)),
                "raw_abs_auc_roc": float(roc_auc_score(labels, raw_abs)),
            }
        )
        print(f"raw baseline: {identifier}: ok", flush=True)
    path = output / "scores" / "baselines.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("kpi_id").to_csv(path, index=False)
    atomic_json(
        output / "baseline_complete_before_hcrd.json",
        {
            "status": "raw_SR_comparator_complete_before_HCRD",
            "series": len(rows),
            "baseline_metrics_sha256": sha256(path),
        },
    )


def evaluate_hcrd(output: Path) -> None:
    if not (output / "baseline_complete_before_hcrd.json").exists():
        raise RuntimeError("raw SR baseline must be complete before HCRD")
    from hcrd import hcrd_area_anomaly_score
    from hcrd.temporal_anomaly import spectral_residual_score

    rows: list[dict[str, object]] = []
    for identifier, group in load_groups(CONFIRMATION).items():
        signal = group["value"].to_numpy(dtype=float)
        labels = group["label"].to_numpy(dtype=int)
        direct = hcrd_area_anomaly_score(signal, max_levels=8, aggregation="max")
        area_sr = spectral_residual_score(direct, amplitude_window=AMPLITUDE_WINDOW)
        rows.append(
            {
                "kpi_id": identifier,
                "hcrd_area_sr_auc_pr": float(average_precision_score(labels, area_sr)),
                "hcrd_area_sr_auc_roc": float(roc_auc_score(labels, area_sr)),
                "hcrd_direct_auc_pr": float(average_precision_score(labels, direct)),
                "hcrd_direct_auc_roc": float(roc_auc_score(labels, direct)),
            }
        )
        print(f"HCRD: {identifier}: ok", flush=True)
    path = output / "scores" / "hcrd.csv"
    pd.DataFrame(rows).sort_values("kpi_id").to_csv(path, index=False)


def bootstrap_interval(difference: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(50_000, dtype=float)
    for start in range(0, 50_000, 1000):
        indices = rng.integers(0, len(difference), size=(1000, len(difference)))
        means[start : start + 1000] = difference[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def exact_sign_flip_p(difference: np.ndarray) -> float:
    if len(difference) > 20:
        raise ValueError("exact enumeration is restricted to at most 20 pairs")
    observed = abs(float(np.mean(difference)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(difference)):
        statistic = abs(float(np.mean(difference * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
        total += 1
    return exceed / total


def comparison(group: pd.DataFrame, *, primary: bool) -> dict[str, object]:
    difference = group["hcrd_area_sr_auc_pr"].to_numpy() - group["raw_sr_auc_pr"].to_numpy()
    low, high = bootstrap_interval(difference, 20260825 + len(group))
    nonzero = difference[np.abs(difference) > 1e-12]
    payload: dict[str, object] = {
        "series": len(group),
        "hcrd_area_sr_mean_auc_pr": float(group["hcrd_area_sr_auc_pr"].mean()),
        "raw_sr_mean_auc_pr": float(group["raw_sr_auc_pr"].mean()),
        "mean_difference": float(difference.mean()),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
        "paired_t_p": float(ttest_1samp(difference, 0.0).pvalue),
        "wilcoxon_p": float(wilcoxon(difference).pvalue),
        "exact_sign_p": float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue),
        "wins": int(np.sum(difference > 1e-12)),
        "ties": int(np.sum(np.abs(difference) <= 1e-12)),
        "losses": int(np.sum(difference < -1e-12)),
    }
    if primary:
        payload["exact_sign_flip_p"] = exact_sign_flip_p(difference)
    return payload


def analyse(output: Path) -> None:
    manifest_frame = pd.read_csv(output / "confirmation_manifest.csv")
    baselines = pd.read_csv(output / "scores" / "baselines.csv")
    hcrd = pd.read_csv(output / "scores" / "hcrd.csv")
    merged = manifest_frame.merge(baselines, on="kpi_id", validate="1:1").merge(
        hcrd, on="kpi_id", validate="1:1"
    )
    merged.to_csv(output / "all_metrics.csv", index=False)
    primary = merged[merged["primary_sparse_transient"]]
    payload = {
        "status": "complete",
        "primary": comparison(primary, primary=True),
        "all_phase2": comparison(merged, primary=False),
        "ablations_primary_mean_auc_pr": {
            "hcrd_direct": float(primary["hcrd_direct_auc_pr"].mean()),
            "raw_absolute_deviation": float(primary["raw_abs_auc_pr"].mean()),
        },
        "interpretation_boundary": "independent phase release and inherited class threshold; only 13 primary KPIs and one domain-aligned comparator",
    }
    atomic_json(output / "summary.json", payload)
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["freeze", "baseline", "hcrd", "analyse", "all-after-freeze"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.phase == "freeze":
        freeze(output)
    elif args.phase == "baseline":
        evaluate_baseline(output)
    elif args.phase == "hcrd":
        evaluate_hcrd(output)
    elif args.phase == "analyse":
        analyse(output)
    else:
        evaluate_baseline(output)
        evaluate_hcrd(output)
        analyse(output)


if __name__ == "__main__":
    main()
