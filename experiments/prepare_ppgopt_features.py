"""Cache label-free multilevel HCRD candidate banks for PPGopt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hcrd.ppg import hcrd_candidate_bank, iter_ppgopt_keys, load_ppgopt_recording


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "raw" / "ppgopt"
OUTPUT = PROJECT / "results" / "ppgopt" / "features"
PHASE_SUBJECTS = {
    "development": (1, 2, 3),
    "validation": (4, 5),
    "confirmation": (6, 7),
    "all": tuple(range(1, 8)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(key: tuple[int, str, int], overwrite: bool) -> dict[str, object]:
    subject, activity, trial = key
    record_key = f"s{subject}_{activity}{trial}"
    destination = OUTPUT / f"{record_key}.npz"
    if destination.exists() and not overwrite:
        with np.load(destination) as cached:
            return {
                "key": record_key,
                "status": "cached",
                "samples": int(cached["sample_count"]),
                "candidates": int(cached["positions"].size),
                "seconds": float(cached["extraction_seconds"]),
                "sha256": sha256(destination),
            }
    recording = load_ppgopt_recording(
        DATA_ROOT,
        subject,
        activity,
        trial,
        load_annotations=False,
    )
    started = time.perf_counter()
    bank = hcrd_candidate_bank(recording.signal, recording.sampling_frequency)
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        destination,
        positions=bank.positions,
        geometry=bank.geometry,
        morphology=bank.morphology,
        geometry_names=np.asarray(bank.geometry_names),
        morphology_names=np.asarray(bank.morphology_names),
        sampling_frequency=recording.sampling_frequency,
        sample_count=recording.signal.size,
        extraction_seconds=elapsed,
    )
    return {
        "key": record_key,
        "status": "computed",
        "samples": int(recording.signal.size),
        "candidates": int(bank.positions.size),
        "seconds": elapsed,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASE_SUBJECTS, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subjects = PHASE_SUBJECTS[args.phase]
    keys = [item for item in iter_ppgopt_keys() if item[0] in subjects]
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract, key, args.overwrite): key for key in keys}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            records.append(result)
            print(
                f"[{completed:02d}/{len(keys):02d}] {result['key']}: "
                f"{result['candidates']} candidates, {result['seconds']:.2f}s "
                f"({result['status']})",
                flush=True,
            )
    records.sort(key=lambda item: str(item["key"]))
    summary = {
        "phase": args.phase,
        "subjects": subjects,
        "workers": args.workers,
        "protocol_sha256": "1e33526deb7e3f24f426c8a2d92506c203e17bcbdee56200b2bde11388cbc782",
        "recordings": len(records),
        "samples": sum(int(item["samples"]) for item in records),
        "candidates": sum(int(item["candidates"]) for item in records),
        "serial_extraction_seconds": sum(float(item["seconds"]) for item in records),
        "records": records,
    }
    manifest = OUTPUT / f"manifest_{args.phase}.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
