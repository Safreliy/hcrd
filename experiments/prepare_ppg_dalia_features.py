"""Cache full HCRD candidates and labels for all PPG-DaLiA records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from hcrd.ppg import hcrd_candidate_bank, match_event_pairs
from hcrd.ppg_benchmark import align_ecg_to_ppg, flatline_mask


PROJECT = Path(__file__).resolve().parents[1]
RECORD_MANIFEST = PROJECT / "data" / "manifests" / "ppg_dalia_records.json"
OUTPUT = PROJECT / "results" / "ppg_dalia" / "features"
PROTOCOL_SHA256 = "0d75839c5ea850795a29f4b0250c7297336c156b3cce3204586b208c847ed000"
RECORD_MANIFEST_SHA256 = "95d18294f434159f3208c4c080a11993ada5d4745dcd40735067221958bc05c0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(row: dict[str, object], overwrite: bool) -> dict[str, object]:
    key = str(row["key"])
    destination = OUTPUT / f"{key}.npz"
    if destination.exists() and not overwrite:
        with np.load(destination) as cached:
            return {
                "key": key,
                "status": "cached",
                "subject": str(cached["subject"]),
                "activity": str(cached["activity"]),
                "candidates": int(cached["positions"].size),
                "reference_beats": int(cached["aligned_reference_p0"].size),
                "feature_seconds": float(cached["feature_seconds"]),
                "sha256": sha256(destination),
            }
    with np.load(PROJECT / str(row["file"])) as raw:
        ppg = np.asarray(raw["ppg"], dtype=float)
        sampling_frequency = float(raw["sampling_frequency"])
        reference = np.asarray(raw["reference"], dtype=np.int64)
        subject = str(raw["subject"])
        activity = str(raw["activity"])
    started = time.perf_counter()
    bank = hcrd_candidate_bank(ppg, sampling_frequency)
    feature_seconds = time.perf_counter() - started
    ppg_invalid = flatline_mask(ppg, sampling_frequency)
    provisional = find_peaks(
        bank.conditioned_signal,
        distance=int(round(sampling_frequency * 60.0 / 300.0)),
        prominence=0.5,
    )[0]
    provisional = provisional[~ppg_invalid[provisional]]
    aligned_reference, lags = align_ecg_to_ppg(
        reference, provisional, sampling_frequency
    )
    in_bounds = (aligned_reference >= 0) & (aligned_reference < ppg.size)
    aligned_reference = aligned_reference[in_bounds]
    lags = lags[in_bounds]
    valid_reference = ~ppg_invalid[aligned_reference]
    aligned_reference = aligned_reference[valid_reference]
    lags = lags[valid_reference]
    keep = ~ppg_invalid[bank.positions]
    positions = bank.positions[keep]
    geometry = bank.geometry[keep]
    morphology = bank.morphology[keep]
    pairs = match_event_pairs(
        aligned_reference,
        positions,
        int(round(0.15 * sampling_frequency)),
    )
    labels = np.zeros(positions.size, dtype=np.int8)
    if pairs.size:
        labels[pairs[:, 1]] = 1
    np.savez_compressed(
        destination,
        positions=positions,
        geometry=geometry,
        morphology=morphology,
        geometry_names=np.asarray(bank.geometry_names),
        morphology_names=np.asarray(bank.morphology_names),
        labels=labels,
        unaligned_reference=reference,
        aligned_reference_p0=aligned_reference,
        alignment_lags_p0=lags,
        ppg_invalid_mask=ppg_invalid,
        sampling_frequency=sampling_frequency,
        sample_count=ppg.size,
        subject=subject,
        activity=activity,
        outer_fold=int(row["outer_fold"]),
        feature_seconds=feature_seconds,
    )
    return {
        "key": key,
        "status": "computed",
        "subject": subject,
        "activity": activity,
        "candidates": int(positions.size),
        "reference_beats": int(aligned_reference.size),
        "feature_seconds": feature_seconds,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if sha256(RECORD_MANIFEST) != RECORD_MANIFEST_SHA256:
        raise RuntimeError("record manifest changed after the protocol was frozen")
    rows = json.loads(RECORD_MANIFEST.read_text(encoding="utf-8"))["records"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    items = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract, row, args.overwrite): row for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            items.append(item)
            print(
                f"[{completed:03d}/{len(rows):03d}] {item['key']}: "
                f"{item['candidates']} candidates, {item['reference_beats']} refs",
                flush=True,
            )
    items.sort(key=lambda item: str(item["key"]))
    manifest = {
        "protocol_sha256": PROTOCOL_SHA256,
        "record_manifest_sha256": RECORD_MANIFEST_SHA256,
        "workers": args.workers,
        "records": len(items),
        "candidates": sum(int(item["candidates"]) for item in items),
        "reference_beats": sum(int(item["reference_beats"]) for item in items),
        "serial_feature_seconds": sum(float(item["feature_seconds"]) for item in items),
        "items": items,
    }
    destination = OUTPUT / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                **{key: value for key, value in manifest.items() if key != "items"},
                "manifest_sha256": sha256(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
