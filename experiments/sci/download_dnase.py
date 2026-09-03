"""Download and verify the public R DNase dataset."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "data/external/dnase"
URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/"
    "Rdatasets/master/csv/datasets/DNase.csv"
)
OFFICIAL_DOCUMENTATION = (
    "https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/DNase.html"
)
EXPECTED_SHA256 = "d9af548405c6772dfbc7caf9998544f27fc133b7dbaa4340ae198787da98d5f4"


def main() -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "shapecontrast/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"DNase checksum changed: {digest}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT_DIR / "DNase.csv"
    data_path.write_bytes(payload)
    manifest = {
        "dataset": "DNase from the R datasets package",
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "mirror_url": URL,
        "official_documentation": OFFICIAL_DOCUMENTATION,
        "sha256": digest,
        "bytes": len(payload),
        "license_note": "Distributed by R as part of the datasets package.",
    }
    (OUTPUT_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
