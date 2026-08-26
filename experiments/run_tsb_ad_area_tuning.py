"""Tune the preregistered HCRD area-spectrum detector on TSB-AD-U tuning."""

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

from hcrd import (  # noqa: E402
    aggregate_area_density,
    multiscale_area_density,
    vus_pr_roc,
)
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402

AGGREGATIONS = ("total", "sum", "max", "l2", "transport")
DEPTHS: tuple[int | None, ...] = (4, 8, None)
SCREENED_HCRD_CANDIDATES = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_name(depth: int | None, aggregation: str) -> str:
    label = "full" if depth is None else f"L{depth}"
    return f"hcrd_{label}_{aggregation}"


def _parse_candidate(candidate: str) -> tuple[int | None, str]:
    if candidate == "raw_abs_median_deviation":
        return 0, "raw"
    parts = candidate.split("_")
    if len(parts) != 3 or parts[0] != "hcrd":
        raise ValueError(f"invalid candidate: {candidate}")
    depth = None if parts[1] == "full" else int(parts[1].removeprefix("L"))
    return depth, parts[2]


def _load_series(data_root_text: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(Path(data_root_text) / filename).dropna()
    return (
        frame["Data"].to_numpy(dtype=float),
        frame["Label"].to_numpy(dtype=int),
    )


def _process_screening_file(task: tuple[str, str]) -> dict[str, object]:
    data_root_text, filename = task
    signal, labels = _load_series(data_root_text, filename)
    started = time.perf_counter()
    densities = multiscale_area_density(signal, max_levels=None)
    decomposition_seconds = time.perf_counter() - started
    rows: list[dict[str, object]] = []

    raw = np.abs(signal - np.median(signal))
    rows.append(
        {
            "file": filename,
            "candidate": "raw_abs_median_deviation",
            "requested_depth": 0,
            "actual_depth": int(densities.shape[0]),
            "aggregation": "raw",
            "auc_pr": float(average_precision_score(labels, raw)),
            "auc_roc": float(roc_auc_score(labels, raw)),
        }
    )
    for requested_depth in DEPTHS:
        used = densities if requested_depth is None else densities[:requested_depth]
        for aggregation in AGGREGATIONS:
            score = aggregate_area_density(used, aggregation=aggregation)
            rows.append(
                {
                    "file": filename,
                    "candidate": _candidate_name(requested_depth, aggregation),
                    "requested_depth": (
                        "full" if requested_depth is None else requested_depth
                    ),
                    "actual_depth": int(used.shape[0]),
                    "aggregation": aggregation,
                    "auc_pr": float(average_precision_score(labels, score)),
                    "auc_roc": float(roc_auc_score(labels, score)),
                }
            )
    return {
        "file": filename,
        "length": int(signal.size),
        "anomaly_points": int(np.sum(labels)),
        "decomposition_seconds": decomposition_seconds,
        "rows": rows,
    }


def _process_vus_file(
    task: tuple[str, str, tuple[str, ...]],
) -> list[dict[str, object]]:
    data_root_text, filename, candidates = task
    signal, labels = _load_series(data_root_text, filename)
    window = int(find_length_rank(signal.reshape(-1, 1), rank=1))
    needed_depths = [_parse_candidate(candidate)[0] for candidate in candidates]
    finite_depths = [depth for depth in needed_depths if isinstance(depth, int)]
    full_needed = any(depth is None for depth in needed_depths)
    max_levels = None if full_needed else max(finite_depths, default=1)
    densities = multiscale_area_density(signal, max_levels=max_levels)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        depth, aggregation = _parse_candidate(candidate)
        if aggregation == "raw":
            score = np.abs(signal - np.median(signal))
        else:
            used = densities if depth is None else densities[:depth]
            score = aggregate_area_density(used, aggregation=aggregation)
        vus_pr, vus_roc = vus_pr_roc(
            labels, score, max_buffer=window, threshold_count=250
        )
        rows.append(
            {
                "file": filename,
                "candidate": candidate,
                "window": window,
                "vus_pr": float(vus_pr),
                "vus_roc": float(vus_roc),
            }
        )
    return rows


def _screening_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("candidate", as_index=False)
        .agg(
            series=("file", "count"),
            mean_auc_pr=("auc_pr", "mean"),
            median_auc_pr=("auc_pr", "median"),
            mean_auc_roc=("auc_roc", "mean"),
        )
        .sort_values(["mean_auc_pr", "median_auc_pr"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _exact_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("candidate", as_index=False)
        .agg(
            series=("file", "count"),
            mean_vus_pr=("vus_pr", "mean"),
            median_vus_pr=("vus_pr", "median"),
            mean_vus_roc=("vus_roc", "mean"),
            mean_auc_pr=("auc_pr", "mean"),
            mean_auc_roc=("auc_roc", "mean"),
        )
        .sort_values(["mean_vus_pr", "median_vus_pr"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _tie_break_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    parsed = output["candidate"].str.extract(
        r"^hcrd_(?P<depth>L\d+|full)_(?P<aggregation>.+)$"
    )
    output["depth_order"] = parsed["depth"].map(
        lambda value: 10**9 if value == "full" else int(value[1:])
    )
    aggregation_order = {name: index for index, name in enumerate(AGGREGATIONS)}
    output["aggregation_order"] = parsed["aggregation"].map(aggregation_order)
    return output


def _screen_candidates(summary: pd.DataFrame) -> list[str]:
    eligible = summary[summary["candidate"].str.startswith("hcrd_")]
    eligible = _tie_break_columns(eligible).sort_values(
        ["mean_auc_pr", "median_auc_pr", "depth_order", "aggregation_order"],
        ascending=[False, False, True, True],
    )
    return eligible.head(SCREENED_HCRD_CANDIDATES)["candidate"].tolist()


def _select_candidate(summary: pd.DataFrame) -> str:
    eligible = summary[summary["candidate"].str.startswith("hcrd_")]
    eligible = _tie_break_columns(eligible).sort_values(
        ["mean_vus_pr", "median_vus_pr", "depth_order", "aggregation_order"],
        ascending=[False, False, True, True],
    )
    return str(eligible.iloc[0]["candidate"])


def _run_parallel(function, tasks, jobs: int, label: str):
    completed = []
    if jobs == 1:
        for task in tasks:
            completed.append(function(task))
            print(f"{label} {task[1]}", flush=True)
        return completed
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(function, task): task[1] for task in tasks}
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"{label} {futures[future]}", flush=True)
    return completed


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
            / "TSB-AD-U-Tuning.csv"
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
    names = pd.read_csv(args.file_list)["file_name"].astype(str).tolist()
    if args.limit is not None:
        names = names[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = str(args.data_root.resolve())
    started = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        screening_items = _run_parallel(
            _process_screening_file,
            [(data_root, name) for name in names],
            args.jobs,
            "screened",
        )
    screening_rows = [row for item in screening_items for row in item["rows"]]
    screening = pd.DataFrame(screening_rows).sort_values(["file", "candidate"])
    screening_summary = _screening_summary(screening)
    screened = _screen_candidates(screening_summary)
    exact_candidates = tuple(screened + ["raw_abs_median_deviation"])
    suffix = "smoke" if args.limit is not None else "tuning"
    screening.to_csv(
        args.output_dir / f"{suffix}_screening_metrics.csv.gz", index=False
    )
    screening_summary.to_csv(
        args.output_dir / f"{suffix}_screening_summary.csv", index=False
    )
    timing = pd.DataFrame(
        [
            {key: value for key, value in item.items() if key != "rows"}
            for item in screening_items
        ]
    ).sort_values("file")
    timing.to_csv(args.output_dir / f"{suffix}_timing.csv", index=False)
    screening_record = {
        "screening_metric": "mean per-series AUC-PR",
        "screened_hcrd_candidates": screened,
        "exact_vus_reference": "raw_abs_median_deviation",
        "screened_before_exact_vus_execution": True,
    }
    (args.output_dir / f"{suffix}_screened_candidates.json").write_text(
        json.dumps(screening_record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(screening_record, indent=2), flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vus_items = _run_parallel(
            _process_vus_file,
            [(data_root, name, exact_candidates) for name in names],
            args.jobs,
            "exact-vus",
        )
    vus = pd.DataFrame([row for rows in vus_items for row in rows])
    metrics = vus.merge(
        screening[["file", "candidate", "auc_pr", "auc_roc"]],
        on=["file", "candidate"],
        how="left",
        validate="1:1",
    ).sort_values(["file", "candidate"])
    exact_summary = _exact_summary(metrics)
    metrics.to_csv(args.output_dir / f"{suffix}_metrics.csv.gz", index=False)
    exact_summary.to_csv(args.output_dir / f"{suffix}_summary.csv", index=False)

    if args.limit is None:
        selected = _select_candidate(exact_summary)
        implementation_files = [
            ROOT / "src" / "hcrd" / "anomaly.py",
            ROOT / "src" / "hcrd" / "energy.py",
            ROOT / "src" / "hcrd" / "tsad_metrics.py",
            Path(__file__).resolve(),
            ROOT / "docs" / "tsb_ad_a1_protocol.md",
        ]
        frozen = {
            "status": "frozen_before_A1_evaluation_execution",
            "selected_candidate": selected,
            "screening": screening_record,
            "selection_metric": "mean per-series exact VUS-PR over official tuning split",
            "tuning_series": len(names),
            "file_list": str(args.file_list.resolve()),
            "file_list_sha256": _sha256(args.file_list),
            "data_archive_sha256": _sha256(
                REPOSITORY_ROOT / "third_party" / "TSB-AD-U.zip"
            ),
            "tsb_ad_code_commit": "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48",
            "implementation_sha256": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in implementation_files
            },
            "wall_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "frozen_configuration.json").write_text(
            json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(frozen, indent=2), flush=True)
    print(exact_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
