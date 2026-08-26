"""Clean single-process runtime measurement for the frozen A1 detector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from hcrd import hcrd_area_anomaly_score  # noqa: E402


def _measure(task: tuple[str, str]) -> dict[str, object]:
    data_root_text, filename = task
    read_started = time.perf_counter()
    signal = pd.read_csv(Path(data_root_text) / filename, usecols=["Data"]).dropna()[
        "Data"
    ].to_numpy(dtype=float)
    read_seconds = time.perf_counter() - read_started
    detector_started = time.perf_counter()
    score = hcrd_area_anomaly_score(signal, max_levels=8, aggregation="max")
    detector_seconds = time.perf_counter() - detector_started
    return {
        "file": filename,
        "length": int(signal.size),
        "read_seconds": read_seconds,
        "detector_seconds": detector_seconds,
        "score_checksum": float(np.sum(score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    data_root = REPOSITORY_ROOT / "third_party" / "TSB-AD-U-data" / "TSB-AD-U"
    file_list = (
        REPOSITORY_ROOT
        / "third_party"
        / "TSB-AD"
        / "Datasets"
        / "File_List"
        / "TSB-AD-U-Eva.csv"
    )
    output_dir = ROOT / "results" / "tsb_ad_a1"
    names = pd.read_csv(file_list)["file_name"].astype(str).tolist()
    hcrd_area_anomaly_score(np.sin(np.linspace(0.0, 20.0, 10_000)))

    tasks = [(str(data_root.resolve()), filename) for filename in names]
    wall_started = time.perf_counter()
    if args.jobs == 1:
        rows = [_measure(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_measure, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    frame = pd.DataFrame(rows)
    label = "single_process" if args.jobs == 1 else f"{args.jobs}_processes"
    frame.to_csv(output_dir / f"runtime_{label}.csv", index=False)
    summary = {
        "protocol": (
            f"{args.jobs} process(es), warmed parent interpreter, "
            "data read outside detector timer"
        ),
        "series": len(frame),
        "samples": int(frame["length"].sum()),
        "detector_seconds": float(frame["detector_seconds"].sum()),
        "read_seconds": float(frame["read_seconds"].sum()),
        "wall_seconds": time.perf_counter() - wall_started,
        "median_detector_milliseconds_per_10k_samples": float(
            np.median(
                frame["detector_seconds"] / frame["length"] * 10_000 * 1_000
            )
        ),
        "aggregate_detector_milliseconds_per_10k_samples": float(
            frame["detector_seconds"].sum()
            / frame["length"].sum()
            * 10_000
            * 1_000
        ),
    }
    (output_dir / f"runtime_{label}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
