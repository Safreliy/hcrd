# SCI manuscript

The paper source is `manuscript.tex`.  It is a standalone LaTeX article and
uses the local `references.bib` file.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

Remove auxiliary build files with:

```powershell
latexmk -c manuscript.tex
```

The frozen numerical sources for every table and figure are stored under
`../../results/sci/` and `../../results/hct/`.  The prose-only planning spine
is retained as `manuscript_plain_english.md`; the LaTeX manuscript is the
authoritative paper source.
