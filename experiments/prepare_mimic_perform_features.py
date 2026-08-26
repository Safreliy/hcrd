"""Cache HCRD candidates and consensus ECG references for MIMIC PERform."""

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
from hcrd.ppg_benchmark import (
    align_ecg_to_ppg,
    consensus_ecg_reference,
    flatline_mask,
)


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data" / "manifests" / "mimic_perform_records.json"
OUTPUT = PROJECT / "results" / "mimic_perform" / "features"
PROTOCOL_SHA256 = "510e47b365ef2e284ebd7adb8d5ecec4bd41ed2b4492a6c7ca6531e4cb76a511"
MANIFEST_SHA256 = "8d2511c3ce018ed5094b8a03fcaf95b652cb5cec051822e0cea68a17a675f517"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(row: dict[str, object]) -> str:
    return f"{row['source_split']}_{Path(str(row['file'])).stem}"


def _extract(row: dict[str, object], overwrite: bool) -> dict[str, object]:
    key = cache_key(row)
    destination = OUTPUT / f"{key}.npz"
    if destination.exists() and not overwrite:
        with np.load(destination) as cached:
            return {
                "key": key,
                "status": "cached",
                "group": str(cached["group"]),
                "candidates": int(cached["positions"].size),
                "reference_beats": int(cached["aligned_reference_p0"].size),
                "agreement_fraction": float(cached["agreement_fraction"]),
                "feature_seconds": float(cached["feature_seconds"]),
                "reference_seconds": float(cached["reference_seconds"]),
                "sha256": sha256(destination),
            }
    source = PROJECT / str(row["file"])
    with np.load(source) as raw:
        ppg = np.asarray(raw["ppg"], dtype=float)
        ecg = np.asarray(raw["ecg"], dtype=float)
        sampling_frequency = float(raw["sampling_frequency"])
        group = str(raw["group"])
    started = time.perf_counter()
    reference = consensus_ecg_reference(ecg, sampling_frequency)
    reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    bank = hcrd_candidate_bank(ppg, sampling_frequency)
    feature_seconds = time.perf_counter() - started
    ppg_invalid = flatline_mask(ppg, sampling_frequency)
    provisional_p0 = find_peaks(
        bank.conditioned_signal,
        distance=int(round(sampling_frequency * 60.0 / 300.0)),
        prominence=0.5,
    )[0]
    provisional_p0 = provisional_p0[~ppg_invalid[provisional_p0]]
    aligned_reference, alignment_lags = align_ecg_to_ppg(
        reference.beats, provisional_p0, sampling_frequency
    )
    valid_reference = (aligned_reference >= 0) & (aligned_reference < ppg.size)
    aligned_reference = aligned_reference[valid_reference]
    alignment_lags = alignment_lags[valid_reference]
    keep_reference = ~ppg_invalid[aligned_reference]
    aligned_reference = aligned_reference[keep_reference]
    alignment_lags = alignment_lags[keep_reference]
    keep_candidates = ~ppg_invalid[bank.positions]
    positions = bank.positions[keep_candidates]
    geometry = bank.geometry[keep_candidates]
    morphology = bank.morphology[keep_candidates]
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
        unaligned_reference=reference.beats,
        aligned_reference_p0=aligned_reference,
        alignment_lags_p0=alignment_lags,
        reference_quality_mask=reference.quality_mask,
        ppg_invalid_mask=ppg_invalid,
        xqrs_beats=reference.xqrs_beats,
        neurokit_beats=reference.neurokit_beats,
        agreement_fraction=reference.agreement_fraction,
        sampling_frequency=sampling_frequency,
        sample_count=ppg.size,
        group=group,
        feature_seconds=feature_seconds,
        reference_seconds=reference_seconds,
    )
    return {
        "key": key,
        "status": "computed",
        "group": group,
        "candidates": int(positions.size),
        "reference_beats": int(aligned_reference.size),
        "agreement_fraction": reference.agreement_fraction,
        "feature_seconds": feature_seconds,
        "reference_seconds": reference_seconds,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("development", "validation", "confirmation"), required=True
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("record manifest changed after split assignment")
    if args.phase == "confirmation" and not (
        PROJECT / "results" / "mimic_perform" / "frozen_confirmation_rule.json"
    ).exists():
        raise RuntimeError("official Testing references remain locked until validation")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [row for row in manifest["records"] if row["phase"] == args.phase]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract, row, args.overwrite): row for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            records.append(result)
            print(
                f"[{completed:03d}/{len(rows):03d}] {result['key']}: "
                f"{result['candidates']} candidates, {result['reference_beats']} refs, "
                f"agreement={result['agreement_fraction']:.3f}",
                flush=True,
            )
    records.sort(key=lambda item: str(item["key"]))
    summary = {
        "phase": args.phase,
        "protocol_sha256": PROTOCOL_SHA256,
        "record_manifest_sha256": MANIFEST_SHA256,
        "workers": args.workers,
        "records": len(records),
        "candidates": sum(int(item["candidates"]) for item in records),
        "reference_beats": sum(int(item["reference_beats"]) for item in records),
        "median_agreement_fraction": float(
            np.median([float(item["agreement_fraction"]) for item in records])
        ),
        "serial_feature_seconds": sum(float(item["feature_seconds"]) for item in records),
        "serial_reference_seconds": sum(
            float(item["reference_seconds"]) for item in records
        ),
        "items": records,
    }
    destination = OUTPUT / f"manifest_{args.phase}.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
