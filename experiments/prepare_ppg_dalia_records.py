"""Export PPG-DaLiA activity records to compact, reproducible NumPy files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT / "data" / "manifests" / "ppg_dalia_manifest.json"
OUTPUT = PROJECT / "data" / "processed" / "ppg_dalia"
DESTINATION_MANIFEST = PROJECT / "data" / "manifests" / "ppg_dalia_records.json"
SOURCE_MANIFEST_SHA256 = "889ca6bd4961cb1f31b1a4e651990565584693264254041581f96b47a068609a"
FOLDS = (
    ("S9", "S13", "S5"),
    ("S1", "S10", "S3"),
    ("S11", "S8", "S2"),
    ("S6", "S4", "S12"),
    ("S7", "S15", "S14"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source manifest changed after the protocol was frozen")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    subject_to_fold = {
        subject: fold for fold, subjects in enumerate(FOLDS) for subject in subjects
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for file_item in source["files"]:
        activity = str(file_item["activity"])
        source_path = PROJECT / str(file_item["file"])
        data = np.atleast_1d(
            loadmat(source_path, squeeze_me=True, struct_as_record=False)["data"]
        )
        for record in data:
            subject = str(record.fix.subj_name)
            ppg = np.asarray(record.ppg.v, dtype=np.float64).reshape(-1)
            ppg_fs = float(record.ppg.fs)
            ecg_fs = float(record.ecg.fs)
            ecg_rpeaks_one_based = np.asarray(
                record.ecg.rpeaks, dtype=np.int64
            ).reshape(-1)
            reference_times = (ecg_rpeaks_one_based - 1) / ecg_fs
            reference = np.unique(np.round(reference_times * ppg_fs).astype(np.int64))
            reference = reference[(reference >= 0) & (reference < ppg.size)]
            destination = OUTPUT / f"{subject}_{activity}.npz"
            np.savez_compressed(
                destination,
                ppg=ppg,
                sampling_frequency=ppg_fs,
                reference=reference,
                subject=subject,
                activity=activity,
            )
            row = {
                "key": f"{subject}_{activity}",
                "subject": subject,
                "activity": activity,
                "outer_fold": subject_to_fold[subject],
                "samples": int(ppg.size),
                "sampling_frequency": ppg_fs,
                "reference_beats": int(reference.size),
                "file": str(destination.relative_to(PROJECT)).replace("\\", "/"),
                "sha256": sha256(destination),
            }
            rows.append(row)
            print(
                f"{row['key']}: {row['samples']} samples, "
                f"{row['reference_beats']} reference beats",
                flush=True,
            )
    rows.sort(key=lambda row: (int(str(row["subject"])[1:]), str(row["activity"])))
    manifest = {
        "dataset": "PPG-DaLiA",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "fold_assignment": "SHA256(subject ID), five consecutive folds of three",
        "folds": [list(subjects) for subjects in FOLDS],
        "records": rows,
    }
    DESTINATION_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(rows),
                "subjects": len({str(row["subject"]) for row in rows}),
                "reference_beats": sum(int(row["reference_beats"]) for row in rows),
                "hours": sum(
                    int(row["samples"]) / float(row["sampling_frequency"])
                    for row in rows
                )
                / 3600.0,
                "manifest": str(DESTINATION_MANIFEST.relative_to(PROJECT)).replace(
                    "\\", "/"
                ),
                "sha256": sha256(DESTINATION_MANIFEST),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
