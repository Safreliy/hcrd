"""Download the exact CWRU records used by the preregistered R1 experiment."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "data" / "raw" / "cwru"
BASE_URL = "https://engineering.case.edu/sites/default/files"

# 12 kHz drive-end records, ordered by motor load 0, 1, 2, and 3 hp.
RECORDS = {
    "normal": [97, 98, 99, 100],
    "inner_007": [105, 106, 107, 108],
    "ball_007": [118, 119, 120, 121],
    "outer_007_6oclock": [130, 131, 132, 133],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for label, record_ids in RECORDS.items():
        for load, record_id in enumerate(record_ids):
            url = f"{BASE_URL}/{record_id}.mat"
            destination = OUTPUT / f"{record_id}.mat"
            if not destination.exists():
                print(f"downloading {url}")
                request = urllib.request.Request(
                    url, headers={"User-Agent": "HCRD reproducibility study"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    destination.write_bytes(response.read())
            manifest.append(
                {
                    "record_id": record_id,
                    "label": label,
                    "load_hp": load,
                    "url": url,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"records": len(manifest), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
