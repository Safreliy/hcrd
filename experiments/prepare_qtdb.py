"""Create the locked QTDB split and download only required record files."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "qtdb"
BASE_URL = "https://physionet.org/files/qtdb/1.0.0"
SALT = "hcrd-r2-20260824:"
EXTENSIONS = ("hea", "dat", "q1c", "pu0", "pu1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(records: list[str]) -> dict[str, object]:
    ranked = sorted(
        records,
        key=lambda name: hashlib.sha256(f"{SALT}{name}".encode()).hexdigest(),
    )
    return {
        "source": f"{BASE_URL}/RECORDS",
        "salt": SALT,
        "algorithm": "sort by SHA-256(salt + record_name)",
        "pilot": ranked[:25],
        "confirmation_locked": ranked[25:],
    }


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        choices=("none", "pilot", "confirmation_locked", "all"),
        default="none",
    )
    args = parser.parse_args()
    records_path = DATA / "RECORDS"
    records = [line.strip() for line in records_path.read_text().splitlines() if line.strip()]
    if len(records) != 105 or len(set(records)) != 105:
        raise RuntimeError("expected 105 unique QTDB records")
    split = _split(records)
    (DATA / "record_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    if args.partition == "none":
        print(json.dumps({key: len(value) for key, value in split.items() if isinstance(value, list)}))
        return
    selected = records if args.partition == "all" else split[args.partition]
    targets = [(record, extension) for record in selected for extension in EXTENSIONS]

    def fetch(target: tuple[str, str]) -> dict[str, object]:
        record, extension = target
        name = f"{record}.{extension}"
        destination = DATA / name
        _download(f"{BASE_URL}/{name}", destination)
        return {
            "record": record,
            "extension": extension,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(fetch, targets))
    manifest_path = DATA / f"download_manifest_{args.partition}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(selected), "files": len(manifest)}))


if __name__ == "__main__":
    main()
