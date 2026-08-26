"""Run the frozen A1 HCRD area detector on the sealed TSB-AD-U evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "third_party" / "TSB-AD"))

from hcrd import hcrd_area_anomaly_score, vus_pr_roc  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_candidate(candidate: str) -> tuple[int | None, str]:
    parts = candidate.split("_")
    if len(parts) != 3 or parts[0] != "hcrd":
        raise ValueError(f"invalid frozen candidate: {candidate}")
    depth = None if parts[1] == "full" else int(parts[1].removeprefix("L"))
    return depth, parts[2]


def _process_file(task: tuple[str, str, int | None, str]) -> dict[str, object]:
    data_root_text, filename, depth, aggregation = task
    frame = pd.read_csv(Path(data_root_text) / filename).dropna()
    signal = frame["Data"].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy(dtype=int)
    window = int(find_length_rank(signal.reshape(-1, 1), rank=1))
    started = time.perf_counter()
    score = hcrd_area_anomaly_score(
        signal, max_levels=depth, aggregation=aggregation
    )
    run_seconds = time.perf_counter() - started
    vus_pr, vus_roc = vus_pr_roc(
        labels, score, max_buffer=window, threshold_count=250
    )
    parts = filename.split("_")
    return {
        "file": filename,
        "source": parts[1],
        "domain": parts[4],
        "length": int(signal.size),
        "anomaly_points": int(np.sum(labels)),
        "window": window,
        "run_seconds": run_seconds,
        "auc_pr": float(average_precision_score(labels, score)),
        "auc_roc": float(roc_auc_score(labels, score)),
        "vus_pr": float(vus_pr),
        "vus_roc": float(vus_roc),
    }


def _bootstrap_mean_difference(
    difference: np.ndarray, *, draws: int = 20_000, seed: int = 20240825
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    batch = 1_000
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        indices = rng.integers(0, difference.size, size=(stop - start, difference.size))
        means[start:stop] = np.mean(difference[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _load_baselines() -> pd.DataFrame:
    benchmark = (
        REPOSITORY_ROOT
        / "third_party"
        / "TSB-AD"
        / "benchmark_exp"
    )
    original = pd.read_csv(
        benchmark / "benchmark_eval_results" / "uni_mergedTable_VUS-PR.csv"
    )
    keep = [
        "file",
        "Sub-PCA",
        "POLY",
        "KShapeAD",
        "Series2Graph",
        "MatrixProfile",
        "point_anomaly",
        "seq_anomaly",
    ]
    merged = original[keep].copy()

    stream = pd.read_csv(benchmark / "leaderboard_results" / "Uni_StreamVAE.csv")
    merged = merged.merge(stream[["file", "StreamVAE"]], on="file", how="left")
    mmpad = pd.read_csv(benchmark / "leaderboard_results" / "Uni_MMPAD.csv")
    merged = merged.merge(
        mmpad[["file", "VUS-PR", "Time"]].rename(
            columns={"VUS-PR": "MMPAD", "Time": "MMPAD_seconds"}
        ),
        on="file",
        how="left",
    )
    ts_pulse = pd.read_csv(benchmark / "leaderboard_results" / "Uni_TSPulse.csv")
    merged = merged.merge(
        ts_pulse[["file", "TSPulse (ZS)", "TSPulse (FT)"]],
        on="file",
        how="left",
    )
    maft = pd.read_csv(benchmark / "leaderboard_results" / "Uni_TimeRCD_MAFT.csv")
    merged = merged.merge(
        maft[["filename", "VUS-PR"]].rename(
            columns={"filename": "file", "VUS-PR": "Time-RCD+MAFT (FT)"}
        ),
        on="file",
        how="left",
    )
    return merged


def _comparison_table(merged: pd.DataFrame) -> list[dict[str, object]]:
    methods = [
        "Sub-PCA",
        "MMPAD",
        "StreamVAE",
        "POLY",
        "KShapeAD",
        "Series2Graph",
        "MatrixProfile",
        "TSPulse (ZS)",
        "TSPulse (FT)",
        "Time-RCD+MAFT (FT)",
    ]
    output: list[dict[str, object]] = []
    for index, method in enumerate(methods):
        valid = merged[["vus_pr", method]].dropna()
        difference = valid["vus_pr"].to_numpy() - valid[method].to_numpy()
        low, high = _bootstrap_mean_difference(
            difference, seed=20240825 + index
        )
        output.append(
            {
                "baseline": method,
                "paired_series": int(len(valid)),
                "hcrd_mean_vus_pr": float(valid["vus_pr"].mean()),
                "baseline_mean_vus_pr": float(valid[method].mean()),
                "mean_difference": float(np.mean(difference)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "wins": int(np.sum(difference > 1e-12)),
                "ties": int(np.sum(np.abs(difference) <= 1e-12)),
                "losses": int(np.sum(difference < -1e-12)),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "TSB-AD-U-data" / "TSB-AD-U",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "third_party"
            / "TSB-AD"
            / "Datasets"
            / "File_List"
            / "TSB-AD-U-Eva.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "tsb_ad_a1"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen_path = args.output_dir / "frozen_configuration.json"
    a2_frozen_path = ROOT / "results" / "tsb_ad_a2" / "frozen_configuration.json"
    a3_frozen_path = ROOT / "results" / "tsb_ad_a3" / "frozen_configuration.json"
    if not frozen_path.exists():
        raise RuntimeError("refusing evaluation: frozen_configuration.json is absent")
    if not a2_frozen_path.exists() or not a3_frozen_path.exists():
        raise RuntimeError("refusing evaluation: A2/A3 development is not frozen")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_A1_evaluation_execution":
        raise RuntimeError("refusing evaluation: unexpected frozen status")
    candidate = str(frozen["selected_candidate"])
    depth, aggregation = _parse_candidate(candidate)

    names = pd.read_csv(args.file_list)["file_name"].astype(str).tolist()
    if args.limit is not None:
        names = names[: args.limit]
    tasks = [
        (str(args.data_root.resolve()), name, depth, aggregation) for name in names
    ]
    completed: list[dict[str, object]] = []
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.jobs == 1:
            for task in tasks:
                item = _process_file(task)
                completed.append(item)
                print(f"completed {item['file']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(_process_file, task): task[1] for task in tasks}
                for future in as_completed(futures):
                    item = future.result()
                    completed.append(item)
                    print(f"completed {item['file']}", flush=True)

    metrics = pd.DataFrame(completed).sort_values("file")
    suffix = "evaluation_smoke" if args.limit is not None else "evaluation"
    metrics.to_csv(args.output_dir / f"{suffix}_metrics.csv.gz", index=False)
    if args.limit is not None:
        print(metrics.to_string(index=False), flush=True)
        return

    merged = metrics.merge(_load_baselines(), on="file", how="left", validate="1:1")
    comparisons = _comparison_table(merged)
    strata: list[dict[str, object]] = []
    for column in ["source", "domain", "point_anomaly", "seq_anomaly"]:
        for value, group in merged.groupby(column):
            strata.append(
                {
                    "stratum": column,
                    "value": str(value),
                    "series": int(len(group)),
                    "mean_vus_pr": float(group["vus_pr"].mean()),
                    "mean_sub_pca": float(group["Sub-PCA"].mean()),
                    "difference_vs_sub_pca": float(
                        (group["vus_pr"] - group["Sub-PCA"]).mean()
                    ),
                }
            )
    summary = {
        "status": "A1_evaluation_executed_after_A1_A2_A3_freeze",
        "candidate": candidate,
        "a2_selected_candidate": json.loads(
            a2_frozen_path.read_text(encoding="utf-8")
        )["selected_candidate"],
        "a3_selected_candidate": json.loads(
            a3_frozen_path.read_text(encoding="utf-8")
        )["selected_candidate"],
        "series": int(len(metrics)),
        "mean_vus_pr": float(metrics["vus_pr"].mean()),
        "median_vus_pr": float(metrics["vus_pr"].median()),
        "mean_auc_pr": float(metrics["auc_pr"].mean()),
        "mean_auc_roc": float(metrics["auc_roc"].mean()),
        "total_detector_seconds": float(metrics["run_seconds"].sum()),
        "wall_seconds": time.perf_counter() - started,
        "median_detector_milliseconds_per_10k_samples": float(
            np.median(metrics["run_seconds"] / metrics["length"] * 10_000 * 1_000)
        ),
        "evaluation_file_list_sha256": _sha256(args.file_list),
        "frozen_configuration_sha256": _sha256(frozen_path),
        "a2_frozen_configuration_sha256": _sha256(a2_frozen_path),
        "a3_frozen_configuration_sha256": _sha256(a3_frozen_path),
        "comparisons": comparisons,
        "strata": strata,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(comparisons).to_csv(
        args.output_dir / "evaluation_comparisons.csv", index=False
    )
    pd.DataFrame(strata).to_csv(args.output_dir / "evaluation_strata.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
