# SCI manuscript for Statistics

[![SCI archive DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22338783.svg)](https://doi.org/10.5281/zenodo.22338783)

The code and numerical results are archived as `sci-v1.0.0` at the
version-specific DOI above (Git commit `2144b48ab2d21fdeb05c6dd9c40b4a9dbe93b28a`).
The current manuscript includes subsequent editorial clarifications; the DOI
identifies the computational archive, not a snapshot of this revised PDF.

The authoritative source is `manuscript.tex`, prepared for **Statistics:
A Journal of Theoretical and Applied Statistics**. It uses Taylor & Francis'
`interact` class and NLM numerical references (`tfnlm.bst`). Both publisher
files and `references.bib` are included locally. See `JOURNAL_PREPARATION.md`
for the checked instructions and `TEMPLATE_PROVENANCE.md` for attribution.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
python check_submission.py --compiled
```

The checker verifies the abstract, keywords, word count, figure/table callouts,
all bibliographic citations, graphics, and the final compilation log. It uses
Python's standard library and TeXcount (included in TeX Live and MiKTeX).
For a self-contained submission ZIP, including the PDF, source, bibliography,
publisher files, and exactly the four graphics used by the manuscript:

```powershell
python check_submission.py --compiled --zip ../../artifacts/sci_statistics_submission.zip
```

The packaging command forces a fresh manuscript compilation before checking
and archiving the files. `--compiled` alone only inspects existing outputs;
it is not a freshness check. The ZIP is a generated, ignored artifact. It does not contain the full research
repository or any temporary build logs. Run the build first, then create it.
The same build commands work after extracting it into an empty directory.
Before submission, build in a new extraction directory, compare the PDF text
with the packaged PDF, and inspect the rebuilt pages visually.

The separate `cover_letter.tex` / `cover_letter.pdf` is intended for the
editorial cover-letter field, not as part of the manuscript source ZIP.
Build it with `latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex`.
Before using it, confirm current submission status and any declarations
requested by the submission portal; see `JOURNAL_PREPARATION.md`.

If the text changes, run the following command and update `word_count.tex`
with its `Sum count` before rebuilding. The manuscript excludes its own word
count declaration from this calculation.

```powershell
texcount -inc -total '-sum=1,1,1,0,0,0,0' manuscript.tex
```

Use `latexmk -c manuscript.tex` to remove auxiliary files after verification.

The frozen numerical sources for every table and figure are stored under
`../../results/sci/` and `../../results/hct/`.  The prose-only planning spine
is retained as `manuscript_plain_english.md`; the LaTeX manuscript is the
authoritative paper source.

Regenerate the executable method overview from this repository root with:

```powershell
python experiments/sci/generate_method_overview.py
```

The command writes both vector PDF and 300 dpi PNG versions to `figures/`.

The manuscript uses existing vector PDFs for the overview, benchmark, and
LIDAR figures. The DNase PNG is placed at 0.75 of the 34-pica text width,
giving approximately 306 effective dpi without resampling or changing data.
