# SIMODS submission build

This directory contains the SIMODS wrapper, submission PDF, and cover letter.
Download the current SIAM journal LaTeX template from the official SIAM author
resources and place `siamonline250211.cls` and `siamplain.bst` in this directory.
The SIAM template files are intentionally not redistributed separately here.

From this directory, build the manuscript with:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The wrapper reads `../main.tex`, figures from `../figures`, generated tables
from `../generated`, and the bibliography from `../references.bib`.
