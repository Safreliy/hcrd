"""Export MIMIC PERform MAT structs into deterministic per-record NPZ files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw" / "mimic_perform"
OUTPUT = PROJECT / "data" / "processed" / "mimic_perform"
FILES = {
    "train": RAW / "mimic_perform_train_all_data.mat",
    "test": RAW / "mimic_perform_test_all_data.mat",
}


def assignment(rows: list[dict[str, object]]) -> None:
    for group in ("a", "n"):
        eligible = [row for row in rows if row["source_split"] == "train" and row["group"] == group]
        eligible.sort(key=lambda row: hashlib.sha256(str(row["record_id"]).encode()).hexdigest())
        if len(eligible) != 100:
            raise RuntimeError(f"expected 100 training records in group {group}")
        for index, row in enumerate(eligible):
            row["phase"] = "development" if index < 80 else "validation"
    for row in rows:
        if row["source_split"] == "test":
            row["phase"] = "confirmation"


def main() -> None:
    rows: list[dict[str, object]] = []
    for source_split, source_path in FILES.items():
        records = loadmat(
            source_path, simplify_cells=True, variable_names=["data"]
        )["data"]
        split_output = OUTPUT / source_split
        split_output.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records):
            ppg = np.asarray(record["ppg"]["v"], dtype=np.float64)
            ecg = np.asarray(record["ekg"]["v"], dtype=np.float64)
            ppg_fs = float(record["ppg"]["fs"])
            ecg_fs = float(record["ekg"]["fs"])
            metadata = record["fix"]
            record_id = str(metadata["file"])
            group = str(metadata["group"])
            if ppg.shape != ecg.shape or ppg.ndim != 1:
                raise RuntimeError(f"unexpected signal shape for {record_id}")
            if ppg_fs != ecg_fs:
                raise RuntimeError(f"sampling-rate mismatch for {record_id}")
            filename = f"{index:03d}_{record_id}.npz"
            destination = split_output / filename
            if not destination.exists():
                np.savez_compressed(
                    destination,
                    ppg=ppg,
                    ecg=ecg,
                    sampling_frequency=ppg_fs,
                    group=group,
                    subject_id=str(metadata["subj_id"]),
                    record_id=record_id,
                )
            rows.append(
                {
                    "source_split": source_split,
                    "source_index": index,
                    "record_id": record_id,
                    "subject_id": str(metadata["subj_id"]),
                    "group": group,
                    "samples": int(ppg.size),
                    "sampling_frequency": ppg_fs,
                    "file": str(destination.relative_to(PROJECT)).replace("\\", "/"),
                }
            )
            print(f"[{source_split} {index + 1:03d}/{len(records):03d}] {record_id}", flush=True)
    assignment(rows)
    rows.sort(key=lambda row: (str(row["source_split"]), int(row["source_index"])))
    manifest = {
        "dataset": "MIMIC PERform Training and Testing Datasets",
        "assignment_rule": (
            "Within each adult/neonate training stratum, sort records by "
            "SHA256(record_id); first 80 development, last 20 validation. "
            "Official testing records are locked confirmation."
        ),
        "counts": {
            phase: sum(row["phase"] == phase for row in rows)
            for phase in ("development", "validation", "confirmation")
        },
        "records": rows,
    }
    destination = PROJECT / "data" / "manifests" / "mimic_perform_records.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

