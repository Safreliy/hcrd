"""Download the official MIMIC PERform train/test benchmark from Zenodo."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "data" / "raw" / "mimic_perform"
RECORD_ID = "6950488"
FILES = {
    "mimic_perform_train_all_data.mat": {
        "bytes": 133919123,
        "md5": "1003ac99d4a4c01fbbe8c54e3f981e43",
    },
    "mimic_perform_test_all_data.mat": {
        "bytes": 133992514,
        "md5": "4cf66ff255ab369433ae2d733444edf6",
    },
}


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, expected in FILES.items():
        url = f"https://zenodo.org/api/records/{RECORD_ID}/files/{filename}/content"
        destination = OUTPUT / filename
        if not destination.exists():
            request = urllib.request.Request(
                url, headers={"User-Agent": "hcrd-research/0.1 (reproducibility download)"}
            )
            with urllib.request.urlopen(request) as response, destination.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        observed_md5 = file_hash(destination, "md5")
        if destination.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"size mismatch for {filename}")
        if observed_md5 != expected["md5"]:
            raise RuntimeError(f"MD5 mismatch for {filename}")
        records.append(
            {
                "file": filename,
                "url": url,
                "bytes": destination.stat().st_size,
                "md5": observed_md5,
                "sha256": file_hash(destination, "sha256"),
            }
        )
        print(f"verified {filename}", flush=True)
    manifest = {
        "dataset": "MIMIC PERform Training and Testing Datasets",
        "zenodo_record": RECORD_ID,
        "source": "https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_testing/",
        "license": "CC BY 4.0",
        "records": records,
    }
    destination = PROJECT / "data" / "manifests" / "mimic_perform_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
