"""Verify every file recorded in ``release_manifest.json``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in payload["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size: {entry['path']}")
            continue
        if digest(path) != entry["sha256"]:
            failures.append(f"sha256: {entry['path']}")
    if failures:
        raise SystemExit("release verification failed:\n" + "\n".join(failures))
    print(f"verified {len(payload['files'])} release files")


if __name__ == "__main__":
    main()
