"""Download and verify the official TSB-UAD subsets needed for C1/D1."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://www.thedatum.org/datasets/TSB-UAD-Public.zip"
ARCHIVE_SHA256 = "ff4aa83a5a111835d410d962152e8dbebcda1039b778bae45b6b9c3f46dd49a1"
CODE_URL = "https://github.com/TheDatumOrg/TSB-UAD.git"
CODE_COMMIT = "313f0fdeba14292b9db4e1aa94c74a983a25de31"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_public_subsets(root: Path) -> None:
    archive = root / "TSB-UAD-Public.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        print(f"downloading {ARCHIVE_URL}")
        with urllib.request.urlopen(ARCHIVE_URL) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    observed = sha256(archive)
    if observed != ARCHIVE_SHA256:
        raise RuntimeError(
            f"TSB-UAD archive hash mismatch: expected {ARCHIVE_SHA256}, got {observed}"
        )

    expected_counts = {"YAHOO": 367, "KDD21": 250}
    with zipfile.ZipFile(archive) as bundle:
        for dataset, expected_count in expected_counts.items():
            destination = root / "TSB-UAD-Public-data" / dataset
            destination.mkdir(parents=True, exist_ok=True)
            members = [
                name
                for name in bundle.namelist()
                if name.startswith(f"TSB-UAD-Public/{dataset}/")
                and name.endswith(".out")
            ]
            for name in members:
                target = destination / Path(name).name
                if not target.exists():
                    with bundle.open(name) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            files = list(destination.glob("*.out"))
            if len(files) != expected_count:
                raise RuntimeError(
                    f"expected {expected_count} {dataset} series, found {len(files)}"
                )
            print(f"verified {len(files)} {dataset} series")
    print(f"verified archive {archive}")


def clone_code(root: Path) -> None:
    destination = root / "TSB-UAD"
    if not destination.exists():
        subprocess.run(["git", "clone", CODE_URL, str(destination)], check=True)
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
            f"TSB-UAD code is at {observed}; expected pinned commit {CODE_COMMIT}"
        )
    print(f"verified benchmark code commit {observed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--third-party-root",
        type=Path,
        default=ROOT.parent / "third_party",
        help="destination used by the frozen C1 experiment",
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.third_party_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_data:
        download_public_subsets(root)
    if not args.skip_code:
        clone_code(root)


if __name__ == "__main__":
    main()
