# Signal Processing submission package

Build the venue-specific manuscript and optional graphical abstract from this
folder:

```bash
latexmk -pdf -jobname=hcrd_signal_processing manuscript.tex
python generate_graphical_abstract.py
python build_submission_source.py
```

The manuscript body is shared with `../main.tex`; the wrapper selects Elsevier's
single-column review layout, journal front matter, numerical references, line
numbers, and six keywords. Highlights, the optional graphical abstract,
cover-letter draft, and the final upload checklist are separate files in this
directory.

The last command creates a flat, compile-tested `submission_source/` directory
and `hcrd_signal_processing_source.zip`. This avoids nested paths, which
Editorial Manager does not process in LaTeX source uploads.
