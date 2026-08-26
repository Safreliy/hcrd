"""Build a flat, compile-ready Elsevier source package.

Editorial Manager does not preserve directory structures in LaTeX uploads.
This script rewrites only local paths, copies the required inputs into one
directory, verifies the flat manuscript with latexmk, and creates a ZIP file.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OUT = HERE / "submission_source"
ZIP_PATH = HERE / "hcrd_signal_processing_source.zip"

FIGURES = (
    "method_overview.pdf",
    "recovery_phase_diagram.pdf",
    "approximate_join_phase.pdf",
    "lcms_evidence.pdf",
)


def flat_main_source() -> str:
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    replacements = {
        r"\graphicspath{{../}}": r"\graphicspath{{./}}",
        "figures/": "",
        "../generated/e2_refit_sensitivity_table.tex": "e2_refit_sensitivity_table.tex",
        "generated/e2_refit_sensitivity_table.tex": "e2_refit_sensitivity_table.tex",
        r"\bibliography{../references}": r"\bibliography{references}",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return "\\def\\SPsubmission{1}\n" + source


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    (OUT / "manuscript.tex").write_text(flat_main_source(), encoding="utf-8")
    shutil.copy2(PAPER / "references.bib", OUT / "references.bib")
    shutil.copy2(
        PAPER / "generated" / "e2_refit_sensitivity_table.tex",
        OUT / "e2_refit_sensitivity_table.tex",
    )
    for name in FIGURES:
        shutil.copy2(PAPER / "figures" / name, OUT / name)

    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "manuscript.tex",
        ],
        cwd=OUT,
        check=True,
    )

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.suffix.lower() in {".tex", ".bib", ".pdf"}:
                archive.write(path, arcname=path.name)

    print(f"Built {OUT}")
    print(f"Built {ZIP_PATH}")


if __name__ == "__main__":
    main()
