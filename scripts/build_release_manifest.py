"""Create deterministic hashes for a public HCRD release snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_manifest.json"
INCLUDED_ROOTS = (
    "src",
    "tests",
    "experiments",
    "docs",
    "theory",
    "results",
    "reports",
    "paper",
    "figures",
    "data/manifests",
    "data/qtdb",
    "examples",
    ".github",
)
INCLUDED_FILES = (
    "README.md",
    "REPRODUCIBILITY.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-CONTENT.md",
    "pyproject.toml",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included_paths() -> list[Path]:
    paths = [ROOT / name for name in INCLUDED_FILES]
    for name in INCLUDED_ROOTS:
        paths.extend(
            path
            for path in (ROOT / name).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
            and path.suffix != ".pyc"
            and path.suffix.lower() not in {".npy", ".npz", ".mzml", ".mzxml"}
            and path.suffix not in {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out"}
            and path.relative_to(ROOT).as_posix() != "paper/main.pdf"
            and not path.relative_to(ROOT).as_posix().startswith("paper/qa")
        )
    paths.extend(
        (
            ROOT / "scripts" / "build_release_manifest.py",
            ROOT / "scripts" / "verify_release.py",
            ROOT / "data" / "README.md",
            ROOT / ".gitignore",
        )
    )
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    entries = []
    for path in included_paths():
        if not path.exists():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    payload = {
        "schema": 1,
        "release": "0.1.0",
        "author": "Saveliy Baturin",
        "algorithm": "SHA-256",
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(entries)} files")


if __name__ == "__main__":
    main()
