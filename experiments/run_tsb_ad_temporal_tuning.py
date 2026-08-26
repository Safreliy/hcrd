"""Tune the preregistered A2 temporal HCRD candidates on TSB-AD-U."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "third_party" / "TSB-AD"))

from hcrd import hcrd_temporal_candidate_scores, vus_pr_roc  # noqa: E402
from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_file(task: tuple[str, str]) -> dict[str, object]:
    data_root_text, filename = task
    frame = pd.read_csv(Path(data_root_text) / filename).dropna()
    signal = frame["Data"].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy(dtype=int)
    window = int(find_length_rank(signal.reshape(-1, 1), rank=1))
    started = time.perf_counter()
    candidates = hcrd_temporal_candidate_scores(signal)
    feature_seconds = time.perf_counter() - started
    rows = []
    for candidate, score in candidates.items():
        vus_pr, vus_roc = vus_pr_roc(
            labels, score, max_buffer=window, threshold_count=250
        )
        rows.append(
            {
                "file": filename,
                "candidate": candidate,
                "window": window,
                "vus_pr": vus_pr,
                "vus_roc": vus_roc,
                "auc_pr": float(average_precision_score(labels, score)),
                "auc_roc": float(roc_auc_score(labels, score)),
            }
        )
    return {
        "file": filename,
        "length": int(signal.size),
        "anomaly_points": int(labels.sum()),
        "feature_seconds": feature_seconds,
        "rows": rows,
    }


def _summarise(frame: pd.DataFrame) -> pd.DataFrame:
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
        .sort_values(
            ["mean_vus_pr", "median_vus_pr", "candidate"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


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
        "--output-dir", type=Path, default=ROOT / "results" / "tsb_ad_a2"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a1_freeze = ROOT / "results" / "tsb_ad_a1" / "frozen_configuration.json"
    if not a1_freeze.exists():
        raise RuntimeError("A1 must be frozen before A2 tuning")
    names = pd.read_csv(args.file_list)["file_name"].astype(str).tolist()
    if args.limit is not None:
        names = names[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(args.data_root.resolve()), name) for name in names]
    completed = []
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.jobs == 1:
            for task in tasks:
                completed.append(_process_file(task))
                print(f"completed {task[1]}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(_process_file, task): task[1] for task in tasks}
                for future in as_completed(futures):
                    completed.append(future.result())
                    print(f"completed {futures[future]}", flush=True)
    rows = [row for item in completed for row in item["rows"]]
    metrics = pd.DataFrame(rows).sort_values(["file", "candidate"])
    timing = pd.DataFrame(
        [
            {key: value for key, value in item.items() if key != "rows"}
            for item in completed
        ]
    ).sort_values("file")
    summary = _summarise(metrics)
    suffix = "smoke" if args.limit is not None else "tuning"
    metrics.to_csv(args.output_dir / f"{suffix}_metrics.csv.gz", index=False)
    timing.to_csv(args.output_dir / f"{suffix}_timing.csv", index=False)
    summary.to_csv(args.output_dir / f"{suffix}_summary.csv", index=False)

    if args.limit is None:
        selected = str(summary.iloc[0]["candidate"])
        implementation_files = [
            ROOT / "src" / "hcrd" / "temporal_anomaly.py",
            ROOT / "src" / "hcrd" / "anomaly.py",
            ROOT / "src" / "hcrd" / "energy.py",
            ROOT / "src" / "hcrd" / "tsad_metrics.py",
            Path(__file__).resolve(),
            ROOT / "docs" / "tsb_ad_a2_protocol.md",
        ]
        frozen = {
            "status": "frozen_before_A1_A2_evaluation_execution",
            "selected_candidate": selected,
            "selection_metric": "mean per-series exact VUS-PR over official tuning split",
            "tuning_series": len(names),
            "file_list_sha256": _sha256(args.file_list),
            "data_archive_sha256": _sha256(
                REPOSITORY_ROOT / "third_party" / "TSB-AD-U.zip"
            ),
            "a1_frozen_configuration_sha256": _sha256(a1_freeze),
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
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

