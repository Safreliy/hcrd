"""Download the official activity-wise PPG-DaLiA MATLAB release from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "data" / "raw" / "ppg_dalia"
MANIFEST = PROJECT / "data" / "manifests" / "ppg_dalia_manifest.json"
ZENODO_RECORD = "12793711"
API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activities",
        nargs="*",
        help="Optional activity names such as walking or table_soccer; default: all.",
    )
    args = parser.parse_args()
    with urllib.request.urlopen(API) as response:
        record = json.load(response)
    requested = set(args.activities or [])
    files = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for item in record["files"]:
        name = str(item["key"])
        activity = name.removeprefix("ppg_dalia_").removesuffix("_data.mat")
        if requested and activity not in requested:
            continue
        destination = OUTPUT / name
        expected_md5 = str(item["checksum"]).removeprefix("md5:")
        if not destination.exists() or digest(destination, "md5") != expected_md5:
            print(f"downloading {name} ({int(item['size']) / 1e6:.1f} MB)", flush=True)
            urllib.request.urlretrieve(str(item["links"]["self"]), destination)
        actual_md5 = digest(destination, "md5")
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {destination}")
        files.append(
            {
                "activity": activity,
                "file": str(destination.relative_to(PROJECT)).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "md5": actual_md5,
                "sha256": digest(destination, "sha256"),
                "source": str(item["links"]["self"]),
            }
        )
    manifest = {
        "dataset": "PPG-DaLiA data in MATLAB format",
        "zenodo_record": ZENODO_RECORD,
        "doi": record["doi"],
        "license": "CC BY 4.0",
        "files": sorted(files, key=lambda item: str(item["activity"])),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
