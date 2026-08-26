"""Download the official TSB-AD-U archive and pin the benchmark code."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://www.thedatum.org/datasets/TSB-AD-U.zip"
ARCHIVE_SHA256 = "0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961"
CODE_URL = "https://github.com/TheDatumOrg/TSB-AD.git"
CODE_COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(root: Path) -> None:
    archive = root / "TSB-AD-U.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        print(f"downloading {ARCHIVE_URL}")
        with urllib.request.urlopen(ARCHIVE_URL) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    observed = sha256(archive)
    if observed != ARCHIVE_SHA256:
        raise RuntimeError(
            f"TSB-AD-U archive hash mismatch: expected {ARCHIVE_SHA256}, got {observed}"
        )

    destination = root / "TSB-AD-U-data"
    expected = destination / "TSB-AD-U"
    if not expected.exists():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    files = list(expected.glob("*.csv"))
    if len(files) != 870:
        raise RuntimeError(f"expected 870 TSB-AD-U CSV files, found {len(files)}")
    print(f"verified {archive} and {len(files)} extracted series")


def clone_code(root: Path) -> None:
    destination = root / "TSB-AD"
    if not destination.exists():
        subprocess.run(
            ["git", "clone", CODE_URL, str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", CODE_COMMIT],
            check=True,
        )
    observed = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != CODE_COMMIT:
        raise RuntimeError(
            f"TSB-AD code is at {observed}; expected pinned commit {CODE_COMMIT}"
        )
    print(f"verified benchmark code commit {observed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--third-party-root",
        type=Path,
        default=ROOT.parent / "third_party",
        help="destination used by the frozen experiment scripts",
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.third_party_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_data:
        download_archive(root)
    if not args.skip_code:
        clone_code(root)


if __name__ == "__main__":
    main()
