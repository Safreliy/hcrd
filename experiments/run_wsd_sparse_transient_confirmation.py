"""Prospective WSD confirmation for sparse local transients.

The runner deliberately separates baseline execution, comparator freezing, and
HCRD execution.  It is resumable at the method/file level and records failures
instead of silently dropping series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, ttest_1samp, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
THIRD_PARTY = REPOSITORY_ROOT / "third_party"
TSB_REPOSITORY = THIRD_PARTY / "TSB-AD"
DATA_ROOT = THIRD_PARTY / "TSB-AD-U-data" / "TSB-AD-U"
FILE_LIST_ROOT = TSB_REPOSITORY / "Datasets" / "File_List"
DEFAULT_OUTPUT = ROOT / "results" / "wsd_sparse_transient_c2"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TSB_REPOSITORY))

from hcrd.tsad_metrics import vus_pr_roc  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

ARCHIVE_SHA256 = "0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961"
TUNING_LIST_SHA256 = "7bf24d2ef834bb39ddd1c8c2b02c177339921dce694354e258ed2b77d8d5cd1c"
EVALUATION_LIST_SHA256 = "6f4e1d4ddbde195f9687f2c5a951b51faff54dda4b658d85f0c07e3e2879615a"
TSB_AD_COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"

PRIMARY_MAX_RUN_FRACTION = 0.005
PRIMARY_OCCUPANCY = 0.01
HCRD_CANDIDATE = "hcrd_L8_max"
PRIMARY_METRIC = "vus_pr"
BASELINES: dict[str, dict[str, object]] = {
    "MMPAD": {"n_neighbor": 5, "n_job": 1},
    "KShapeAD": {"periodicity": 1},
    "SAND": {"periodicity": 1},
    "Sub_PCA": {"periodicity": 1, "n_components": None, "n_jobs": 1},
    "MatrixProfile": {"periodicity": 1, "n_jobs": 1},
    "SR": {"periodicity": 1},
    "POLY": {"periodicity": 1, "power": 4, "n_jobs": 1},
    "Sub_IForest": {"periodicity": 1, "n_estimators": 150, "n_jobs": 1},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_sha256(score: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(score, dtype="<f8").tobytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def longest_run(labels: np.ndarray) -> int:
    padded = np.pad(labels.astype(int), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return int(np.max(changes[1::2] - changes[::2], initial=0))


def event_count(labels: np.ndarray) -> int:
    return int(np.sum(np.diff(np.pad(labels.astype(int), (1, 0))) == 1))


def input_integrity() -> dict[str, str]:
    return {
        "archive_sha256": sha256(THIRD_PARTY / "TSB-AD-U.zip"),
        "tuning_list_sha256": sha256(FILE_LIST_ROOT / "TSB-AD-U-Tuning.csv"),
        "evaluation_list_sha256": sha256(FILE_LIST_ROOT / "TSB-AD-U-Eva.csv"),
    }


def assert_input_integrity() -> None:
    expected = {
        "archive_sha256": ARCHIVE_SHA256,
        "tuning_list_sha256": TUNING_LIST_SHA256,
        "evaluation_list_sha256": EVALUATION_LIST_SHA256,
    }
    observed = input_integrity()
    if observed != expected:
        raise RuntimeError(f"input integrity mismatch: {observed}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=TSB_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != TSB_AD_COMMIT:
        raise RuntimeError(f"unexpected TSB-AD commit: {commit}")


def population_manifest() -> pd.DataFrame:
    used: set[str] = set()
    for name in ["TSB-AD-U-Tuning.csv", "TSB-AD-U-Eva.csv"]:
        used.update(pd.read_csv(FILE_LIST_ROOT / name).iloc[:, 0].astype(str))
    rows: list[dict[str, object]] = []
    for path in sorted(DATA_ROOT.glob("*_WSD_*.csv")):
        if path.name in used:
            continue
        frame = pd.read_csv(path).dropna()
        signal = frame["Data"].to_numpy(dtype=float)
        labels = frame["Label"].to_numpy(dtype=int)
        maximum_run = longest_run(labels)
        anomaly_points = int(np.sum(labels))
        identifier = re.search(r"_id_(\d+)_", path.name)
        if identifier is None:
            raise RuntimeError(f"cannot parse WSD id from {path.name}")
        run_fraction = maximum_run / signal.size
        occupancy = anomaly_points / signal.size
        rows.append(
            {
                "file": path.name,
                "wsd_id": int(identifier.group(1)),
                "length": int(signal.size),
                "train_index": int(path.stem.split("_")[-3]),
                "anomaly_points": anomaly_points,
                "anomaly_events": event_count(labels),
                "max_run": maximum_run,
                "max_run_fraction": run_fraction,
                "anomaly_occupancy": occupancy,
                "primary_sparse_transient": bool(
                    run_fraction <= PRIMARY_MAX_RUN_FRACTION
                    and occupancy <= PRIMARY_OCCUPANCY
                ),
                "file_sha256": sha256(path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    if len(manifest) != 86 or int(manifest["primary_sparse_transient"].sum()) != 71:
        raise RuntimeError(
            "unexpected WSD population counts: "
            f"{len(manifest)} total, "
            f"{int(manifest['primary_sparse_transient'].sum())} primary"
        )
    if manifest["wsd_id"].nunique() != len(manifest):
        raise RuntimeError("WSD identifiers are not independent series identifiers")
    return manifest


def freeze_population(output: Path) -> None:
    assert_input_integrity()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / "metrics").rglob("*.json")):
        raise RuntimeError("refusing to freeze after method metrics exist")
    manifest = population_manifest()
    manifest_path = output / "population_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    protocol = ROOT / "docs" / "wsd_sparse_transient_c2_protocol.md"
    runner = Path(__file__).resolve()
    payload: dict[str, object] = {
        "status": "population_and_methods_frozen_before_any_C2_score",
        "study": "C2 real WSD sparse-local-transient confirmation",
        "population_definition": "all WSD series unused by official TSB-AD-U tuning and evaluation lists",
        "population_series": 86,
        "primary_stratum": {
            "definition": "max_run / length <= 0.005 and anomaly_points / length <= 0.01",
            "series": 71,
            "threshold_selection_disclosure": "chosen after inspecting labels/durations but before executing any C2 method score",
        },
        "secondary_stratum": "all 86 series",
        "primary_metric": "mean per-series exact VUS-PR",
        "secondary_metrics": ["AUC-PR", "VUS-ROC", "AUC-ROC"],
        "baseline_methods": BASELINES,
        "baseline_source": "official TSB-AD wrappers and Optimal_Uni_algo_HP_dict",
        "compatibility_changes": {
            "numpy": "np.Inf alias restored as np.inf for NumPy 2 compatibility",
            "MMPAD": "n_job=1 changes resource scheduling only",
        },
        "unavailable_official_methods": {
            "NORMA": "official source archive is password protected/patent restricted",
            "Series2Graph": "official source archive is password protected/patent restricted",
        },
        "comparator_rule": "strongest mean primary VUS-PR among baseline methods complete on all 71 primary series, frozen before HCRD execution",
        "hcrd_candidate": HCRD_CANDIDATE,
        "hcrd_execution_gate": "comparator_frozen.json must exist and no prior HCRD metric is allowed",
        "inference": {
            "primary": "paired mean difference, 50000-draw percentile bootstrap CI, paired t-test, Wilcoxon, exact sign test",
            "secondary_baselines": "Holm correction across fixed baseline family",
            "effect_modifiers": ["max_run_fraction", "anomaly_occupancy"],
        },
        "timing_policy": "wall time is diagnostic only and is excluded from all claims because the host CPU was concurrently loaded",
        **input_integrity(),
        "tsb_ad_commit": TSB_AD_COMMIT,
        "population_manifest_sha256": sha256(manifest_path),
        "implementation_sha256": {
            runner.relative_to(ROOT).as_posix(): sha256(runner),
            protocol.relative_to(ROOT).as_posix(): sha256(protocol),
            "src/hcrd/anomaly.py": sha256(ROOT / "src" / "hcrd" / "anomaly.py"),
            "src/hcrd/energy.py": sha256(ROOT / "src" / "hcrd" / "energy.py"),
            "src/hcrd/tsad_metrics.py": sha256(ROOT / "src" / "hcrd" / "tsad_metrics.py"),
        },
    }
    atomic_json(output / "frozen_population.json", payload)
    print(json.dumps(payload, indent=2))


def load_signal(filename: str) -> tuple[np.ndarray, np.ndarray, int]:
    frame = pd.read_csv(DATA_ROOT / filename).dropna()
    signal = frame["Data"].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy(dtype=int)
    train_index = int(Path(filename).stem.split("_")[-3])
    return signal, labels, train_index


def normalise_score(score: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(score, dtype=float).ravel()
    if values.size != length or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid score shape/values: {values.shape}, expected {length}")
    if np.ptp(values) == 0.0:
        return np.zeros_like(values)
    return MinMaxScaler(feature_range=(0, 1)).fit_transform(values[:, None]).ravel()


def metrics_for_score(
    filename: str, method: str, score: np.ndarray, seconds: float
) -> dict[str, object]:
    signal, labels, _ = load_signal(filename)
    values = normalise_score(score, signal.size)
    window = int(find_length_rank(signal.reshape(-1, 1), rank=1))
    vus_pr, vus_roc = vus_pr_roc(
        labels, values, max_buffer=window, threshold_count=250
    )
    return {
        "status": "ok",
        "method": method,
        "file": filename,
        "length": int(signal.size),
        "window": window,
        "auc_pr": float(average_precision_score(labels, values)),
        "auc_roc": float(roc_auc_score(labels, values)),
        "vus_pr": float(vus_pr),
        "vus_roc": float(vus_roc),
        "score_sha256_float64_le": score_sha256(values),
        "diagnostic_wall_seconds_excluded_from_claims": seconds,
    }


def run_worker(method: str, filename: str, result_path: Path) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    np.random.seed(2024)
    random.seed(2024)
    # Compatibility shim for the pinned TSB-AD implementation under NumPy 2.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]
    started = time.perf_counter()
    try:
        signal, _, train_index = load_signal(filename)
        data = signal[:, None]
        if method == "HCRD":
            from hcrd import hcrd_area_anomaly_score

            score = hcrd_area_anomaly_score(
                signal, max_levels=8, aggregation="max"
            )
            parameters: dict[str, object] = {
                "max_levels": 8,
                "aggregation": "max",
            }
        else:
            from TSB_AD.model_wrapper import (
                run_Semisupervise_AD,
                run_Unsupervise_AD,
            )

            parameters = dict(BASELINES[method])
            if method == "SAND":
                score = run_Semisupervise_AD(
                    method, data[:train_index], data, **parameters
                )
            else:
                score = run_Unsupervise_AD(method, data, **parameters)
            if isinstance(score, str):
                raise RuntimeError(score)
        payload = metrics_for_score(
            filename, method, np.asarray(score), time.perf_counter() - started
        )
        payload["parameters"] = parameters
        atomic_json(result_path, payload)
    except Exception as error:  # retained as an auditable failure
        atomic_json(
            result_path,
            {
                "status": "failed",
                "method": method,
                "file": filename,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "diagnostic_wall_seconds_excluded_from_claims": time.perf_counter()
                - started,
            },
        )
        raise


def result_path(output: Path, method: str, filename: str) -> Path:
    return output / "metrics" / method / f"{filename}.json"


def launch_one(output: Path, method: str, filename: str) -> tuple[str, bool, str]:
    path = result_path(output, method, filename)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            return filename, True, "cached"
        path.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "worker",
        "--output-dir",
        str(output),
        "--method",
        method,
        "--file",
        filename,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    message = (completed.stdout + "\n" + completed.stderr)[-4000:]
    return filename, completed.returncode == 0, message


def execute_method(output: Path, method: str, jobs: int) -> None:
    frozen = output / "frozen_population.json"
    if not frozen.exists():
        raise RuntimeError("run --phase freeze before any method")
    if method == "HCRD" and not (output / "comparator_frozen.json").exists():
        raise RuntimeError("refusing HCRD: comparator has not been frozen")
    if method != "HCRD" and method not in BASELINES:
        raise ValueError(f"unknown baseline method: {method}")
    manifest = pd.read_csv(output / "population_manifest.csv")
    names = manifest["file"].astype(str).tolist()
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(launch_one, output, method, name): name for name in names
        }
        for future in as_completed(futures):
            name, ok, message = future.result()
            print(f"{method}: {name}: {'ok' if ok else 'FAILED'}", flush=True)
            if not ok:
                failures.append((name, message))
    if failures:
        detail = "\n\n".join(f"{name}:\n{message}" for name, message in failures)
        raise RuntimeError(f"{method} failed on {len(failures)} series\n{detail}")


def read_method_metrics(output: Path, method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((output / "metrics" / method).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    return pd.DataFrame(rows)


def freeze_comparator(output: Path) -> None:
    if (output / "metrics" / "HCRD").exists():
        if any((output / "metrics" / "HCRD").glob("*.json")):
            raise RuntimeError("refusing comparator selection after HCRD execution")
    manifest = pd.read_csv(output / "population_manifest.csv")
    primary = set(
        manifest.loc[manifest["primary_sparse_transient"], "file"].astype(str)
    )
    means: dict[str, float] = {}
    completeness: dict[str, dict[str, object]] = {}
    for method in BASELINES:
        metrics = read_method_metrics(output, method)
        good = metrics[metrics.get("status", "") == "ok"] if len(metrics) else metrics
        available = set(good.get("file", pd.Series(dtype=str)).astype(str))
        complete = primary <= available
        values = good[good.get("file", pd.Series(dtype=str)).isin(primary)]
        completeness[method] = {
            "primary_ok": int(len(values)),
            "primary_required": len(primary),
            "eligible": complete,
        }
        if complete:
            means[method] = float(values[PRIMARY_METRIC].mean())
    if len(means) < 4:
        raise RuntimeError(f"fewer than four complete baseline methods: {means}")
    comparator = max(means, key=means.get)
    payload = {
        "status": "primary_comparator_frozen_before_HCRD_execution",
        "primary_comparator": comparator,
        "selection_metric": "mean per-series VUS-PR on the fixed 71-series primary stratum",
        "eligible_baseline_means": means,
        "baseline_completeness": completeness,
        "selection_rule": "largest eligible primary mean",
        "frozen_population_sha256": sha256(output / "frozen_population.json"),
    }
    atomic_json(output / "comparator_frozen.json", payload)
    print(json.dumps(payload, indent=2))


def bootstrap_interval(
    difference: np.ndarray, *, draws: int = 50_000, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = rng.integers(
            0, difference.size, size=(stop - start, difference.size)
        )
        means[start:stop] = difference[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison(
    merged: pd.DataFrame, baseline: str, *, seed: int
) -> dict[str, object]:
    difference = merged["HCRD_vus_pr"].to_numpy() - merged[f"{baseline}_vus_pr"].to_numpy()
    low, high = bootstrap_interval(difference, seed=seed)
    nonzero = difference[np.abs(difference) > 1e-12]
    signed = binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue
    try:
        wilcoxon_p = float(wilcoxon(difference, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    sd = float(np.std(difference, ddof=1))
    return {
        "baseline": baseline,
        "series": int(len(difference)),
        "hcrd_mean_vus_pr": float(merged["HCRD_vus_pr"].mean()),
        "baseline_mean_vus_pr": float(merged[f"{baseline}_vus_pr"].mean()),
        "mean_difference": float(np.mean(difference)),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
        "paired_t_p": float(ttest_1samp(difference, 0.0).pvalue),
        "wilcoxon_p": wilcoxon_p,
        "exact_sign_p": float(signed),
        "cohen_dz": float(np.mean(difference) / sd) if sd > 0 else float("inf"),
        "wins": int(np.sum(difference > 1e-12)),
        "ties": int(np.sum(np.abs(difference) <= 1e-12)),
        "losses": int(np.sum(difference < -1e-12)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def analyse(output: Path) -> None:
    comparator_payload = json.loads(
        (output / "comparator_frozen.json").read_text(encoding="utf-8")
    )
    comparator = str(comparator_payload["primary_comparator"])
    manifest = pd.read_csv(output / "population_manifest.csv")
    merged = manifest.copy()
    for method in [*BASELINES, "HCRD"]:
        metrics = read_method_metrics(output, method)
        good = metrics[metrics["status"] == "ok"]
        if len(good) != len(manifest):
            raise RuntimeError(f"{method} has {len(good)}/{len(manifest)} successful series")
        columns = ["file", "auc_pr", "auc_roc", "vus_pr", "vus_roc"]
        renamed = good[columns].rename(
            columns={name: f"{method}_{name}" for name in columns if name != "file"}
        )
        merged = merged.merge(renamed, on="file", how="left", validate="1:1")
    merged.to_csv(output / "all_metrics.csv", index=False)

    comparisons: list[dict[str, object]] = []
    for stratum_name, group in [
        ("primary_sparse_transient", merged[merged["primary_sparse_transient"]]),
        ("all_WSD", merged),
        ("outside_primary", merged[~merged["primary_sparse_transient"]]),
    ]:
        for index, method in enumerate(BASELINES):
            item = paired_comparison(group, method, seed=20260825 + index + 100 * len(stratum_name))
            item["stratum"] = stratum_name
            item["is_frozen_primary_comparison"] = bool(
                stratum_name == "primary_sparse_transient" and method == comparator
            )
            comparisons.append(item)
    table = pd.DataFrame(comparisons)
    for stratum in table["stratum"].unique():
        mask = table["stratum"] == stratum
        table.loc[mask, "holm_paired_t_p"] = holm_adjust(
            table.loc[mask, "paired_t_p"].astype(float).tolist()
        )
        table.loc[mask, "holm_wilcoxon_p"] = holm_adjust(
            table.loc[mask, "wilcoxon_p"].astype(float).tolist()
        )
        table.loc[mask, "holm_exact_sign_p"] = holm_adjust(
            table.loc[mask, "exact_sign_p"].astype(float).tolist()
        )
    table.to_csv(output / "paired_comparisons.csv", index=False)

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
    pd.DataFrame(modifiers).to_csv(output / "effect_modifiers.csv", index=False)

    primary_row = table[
        (table["stratum"] == "primary_sparse_transient")
        & (table["baseline"] == comparator)
    ].iloc[0]
    payload = {
        "status": "complete",
        "primary_comparator": comparator,
        "primary_series": int(len(primary)),
        "full_series": int(len(merged)),
        "primary_result": primary_row.to_dict(),
        "timing_claim": "none; diagnostic wall time excluded",
        "files": {
            "all_metrics": "all_metrics.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "effect_modifiers": "effect_modifiers.csv",
        },
    }
    atomic_json(output / "summary.json", payload)
    print(json.dumps(payload, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["freeze", "worker", "method", "baselines", "freeze-comparator", "hcrd", "analyse"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=[*BASELINES, "HCRD"])
    parser.add_argument("--file")
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if args.phase == "freeze":
        freeze_population(output)
    elif args.phase == "worker":
        if args.method is None or args.file is None:
            raise ValueError("worker requires --method and --file")
        run_worker(args.method, args.file, result_path(output, args.method, args.file))
    elif args.phase == "method":
        if args.method is None:
            raise ValueError("method phase requires --method")
        execute_method(output, args.method, args.jobs)
    elif args.phase == "baselines":
        for method in BASELINES:
            execute_method(output, method, args.jobs)
    elif args.phase == "freeze-comparator":
        freeze_comparator(output)
    elif args.phase == "hcrd":
        execute_method(output, "HCRD", args.jobs)
    elif args.phase == "analyse":
        analyse(output)


if __name__ == "__main__":
    main()
