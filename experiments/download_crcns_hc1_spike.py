"""Download the fixed CRCNS HC-1 roots for the SpikeForest spike study."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "third_party" / "crcns_hc1"
BASE_URL = "https://crcnsarchive.s3.amazonaws.com/hc-1/live/Data"

# Fifteen smallest public HC-1 archives (as listed on 2026-08-25) that contain
# at least one recording in the official SpikeForest PAIRED_CRCNS_HC1 index.
SESSION_ROOTS = (
    "d13521",
    "d5331",
    "d13921",
    "d14531",
    "d15121",
    "d7211",
    "d7111",
    "d5611",
    "d13711",
    "d18811",
    "d18712",
    "d7212",
    "d6111",
    "d14921",
    "d12821",
)


def split_for_root(root: str) -> str:
    """Deterministic session-level split fixed before waveform inspection."""

    first_byte = hashlib.sha256(root.encode("utf-8")).digest()[0]
    return "development" if first_byte % 2 == 0 else "confirmation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(root: str) -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / f"{root}.zip"
    if not archive.exists():
        url = f"{BASE_URL}/{root}.zip"
        temporary = archive.with_suffix(".zip.partial")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        temporary.replace(archive)
    extract_dir = OUTPUT / root
    marker = extract_dir / ".extracted"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_dir)
        marker.write_text("complete\n", encoding="utf-8")
    return {
        "root": root,
        "split": split_for_root(root),
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["development", "confirmation", "all"],
        default="development",
    )
    parser.add_argument("--root", choices=SESSION_ROOTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [args.root] if args.root else list(SESSION_ROOTS)
    if args.split != "all":
        selected = [root for root in selected if split_for_root(root) == args.split]
    rows = []
    for root in selected:
        print(f"downloading/extracting {root} ({split_for_root(root)})", flush=True)
        rows.append(download(root))
    manifest = OUTPUT / f"download_manifest_{args.split}.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
