#!/usr/bin/env python3
"""Resume, verify and selectively extract the official TARDIS E6 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/tardis_e6_manifest.json"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0
    if existing > expected_size:
        raise ValueError(f"existing file is too large: {existing} > {expected_size}")
    if existing == expected_size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "hcrd-e6-reproducer/1"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        if existing and response.status != 206:
            raise RuntimeError("server ignored resume Range; keep the partial file and retry")
        mode = "ab" if existing else "xb"
        with target.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                existing += len(chunk)
                print(f"downloaded {existing}/{expected_size} bytes", flush=True)
    if target.stat().st_size != expected_size:
        raise ValueError(f"download size mismatch: {target.stat().st_size}")


def _seven_zip() -> str:
    command = shutil.which("7z")
    if command:
        return command
    windows = Path(r"C:\Program Files\7-Zip\7z.exe")
    if windows.exists():
        return str(windows)
    raise RuntimeError("7-Zip is required for selective RAR extraction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/external/data_zenodo.rar",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=ROOT / "data/external/tardis_extracted",
    )
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _download(
        manifest["archive_url"],
        args.archive,
        int(manifest["archive_size_bytes"]),
    )
    observed = _md5(args.archive)
    if observed != manifest["archive_md5"]:
        raise ValueError(f"archive MD5 mismatch: {observed}")
    print(f"verified MD5 {observed}", flush=True)
    if args.extract:
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        command = [
            _seven_zip(),
            "x",
            "-y",
            f"-o{args.extract_dir}",
            str(args.archive),
            *manifest["selected_archive_paths"],
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
