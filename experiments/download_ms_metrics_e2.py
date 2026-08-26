#!/usr/bin/env python3
"""Download, verify, and extract the frozen E2 LC--MS sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/ms_metrics_e2_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    start = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        if start and response.status != 206:
            raise RuntimeError(f"server did not honor resume range for {url}")
        mode = "ab" if start else "wb"
        with temporary.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def _seven_zip() -> str:
    candidates = (
        shutil.which("7zz"),
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "7-Zip is required because the official MESOSCOPE archive uses Deflate64"
    )


def _clone_and_verify(root: Path, manifest: dict) -> None:
    if not root.exists():
        subprocess.run(
            ["git", "clone", manifest["source_repository"], str(root)], check=True
        )
    expected = manifest["source_repository_commit"]
    observed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected:
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", expected], check=True
        )
    observed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected:
        raise RuntimeError(f"repository commit {observed}; expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--third-party-dir", type=Path, default=ROOT / "third_party/MS_metrics"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    destination = args.third_party_dir.resolve()
    _clone_and_verify(destination, manifest)
    archives = destination / "raw_archives"
    archives.mkdir(parents=True, exist_ok=True)
    raw_root = destination / "raw_mzml"
    raw_root.mkdir(parents=True, exist_ok=True)
    extractor = _seven_zip() if not args.skip_extract else None
    for dataset in manifest["datasets"]:
        archive = archives / dataset["archive_file"]
        if not archive.exists() and args.skip_download:
            raise FileNotFoundError(archive)
        if not archive.exists():
            print(f"downloading {dataset['name']} from {dataset['archive_url']}")
            _download(dataset["archive_url"], archive)
        if archive.stat().st_size != dataset["archive_bytes"]:
            raise RuntimeError(f"byte count mismatch for {archive}")
        observed = _sha256(archive)
        if observed != dataset["archive_sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {archive}: {observed}")
        labels = destination / dataset["labels_file"]
        if _sha256(labels) != dataset["labels_sha256"]:
            raise RuntimeError(f"label hash mismatch for {labels}")
        extracted = raw_root / dataset["name"]
        if not args.skip_extract:
            extracted.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [extractor, "x", str(archive), f"-o{extracted}", "-aoa", "-bso0", "-bsp0"],
                check=True,
            )
        files = sorted(extracted.glob("*.mzML"))
        if files and len(files) != dataset["mzml_count"]:
            raise RuntimeError(
                f"{dataset['name']}: expected {dataset['mzml_count']} mzML, found {len(files)}"
            )
        print(f"verified {dataset['name']}: {archive.stat().st_size} bytes, {len(files)} mzML")


if __name__ == "__main__":
    main()
