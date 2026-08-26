"""Download the public PPG-DaLiA-derived peak annotation benchmark."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "data" / "raw" / "ppgopt"
BASE = "https://www.eti.uni-siegen.de/ubicomp/home/datasets/embc21/"
ARCHIVES = {
    "ppgopt_annot.zip": BASE + "ppgopt_annot.zip",
    "ppgopt_viridis.zip": BASE + "ppgopt_viridis.zip",
    # Original PPG/accelerometer recordings used by the annotation benchmark.
    "PMC6971339_supplementary.zip": (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6971339/"
        "supplementaryFiles"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in ARCHIVES.items():
        archive = OUTPUT / filename
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        digest = sha256(archive)
        archive_stem = archive.name.removesuffix(".zip")
        extract_path = OUTPUT / archive_stem
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            names = [item.filename for item in members]
            for name in names:
                destination = (extract_path / name).resolve()
                if (
                    extract_path.resolve() not in destination.parents
                    and destination != extract_path.resolve()
                ):
                    raise RuntimeError(f"unsafe archive member: {name}")
            handle.extractall(extract_path)
        nested_records = []
        for nested in sorted(extract_path.glob("*.zip")):
            nested_path = extract_path / nested.stem
            nested_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(nested) as handle:
                nested_members = handle.infolist()
                for item in nested_members:
                    destination = (nested_path / item.filename).resolve()
                    if (
                        nested_path.resolve() not in destination.parents
                        and destination != nested_path.resolve()
                    ):
                        raise RuntimeError(
                            f"unsafe nested archive member: {item.filename}"
                        )
                handle.extractall(nested_path)
            nested_records.append(
                {
                    "file": nested.name,
                    "bytes": nested.stat().st_size,
                    "sha256": sha256(nested),
                    "members": len(nested_members),
                }
            )
        records.append(
            {
                "file": filename,
                "url": url,
                "bytes": archive.stat().st_size,
                "sha256": digest,
                "members": len(members),
                "nested_archives": nested_records,
            }
        )
    manifest = {
        "dataset": "Optimal Preprocessing of Raw Reflective PPG (PPGopt)",
        "source": "https://www.eti.uni-siegen.de/ubicomp/home/datasets/embc21/",
        "records": records,
    }
    manifest_path = PROJECT / "data" / "manifests" / "ppgopt_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
