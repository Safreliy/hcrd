"""Build and validate the source archive intended for arXiv upload."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
OUT = HERE / "arxiv_source"
ZIP_PATH = HERE / "hcrd_arxiv_source.zip"

SOURCES = (
    Path("main.tex"),
    Path("references.bib"),
    Path("generated/e2_refit_sensitivity_table.tex"),
    Path("figures/method_overview.pdf"),
    Path("figures/recovery_phase_diagram.pdf"),
    Path("figures/approximate_join_phase.pdf"),
    Path("figures/lcms_evidence.pdf"),
    Path("figures/curvature_visibility_failure.pdf"),
)


def validate_archive_names(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        if "\\" in name:
            raise ValueError(f"Windows separator in archive path: {name!r}")
        if path.is_absolute() or re.match(r"^[A-Za-z]:", name):
            raise ValueError(f"Absolute or drive-qualified archive path: {name!r}")
        if ".." in path.parts:
            raise ValueError(f"Parent traversal in archive path: {name!r}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for relative in SOURCES:
        source = HERE / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ],
        cwd=OUT,
        check=True,
    )

    archive_files = list(SOURCES) + [Path("main.bbl")]
    archive_names = [path.as_posix() for path in archive_files]
    validate_archive_names(archive_names)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, arcname in zip(archive_files, archive_names, strict=True):
            archive.write(OUT / relative, arcname=arcname)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        stored_names = archive.namelist()
        validate_archive_names(stored_names)
        if stored_names != archive_names:
            raise ValueError("Archive member order or names changed unexpectedly")
        bad_members = archive.testzip()
        if bad_members is not None:
            raise ValueError(f"CRC failure in archive member: {bad_members}")

    print(f"Built and compile-tested {ZIP_PATH}")
    print("Archive paths are relative POSIX paths:")
    for name in archive_names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
