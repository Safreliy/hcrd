# Publisher template provenance

Retrieved on 2026-09-05 from Taylor & Francis:

- Bundle: <https://files.taylorandfrancis.com/InteractNLMLaTeX.zip>
- Bundle SHA-256: `56fe0625442d5ab78f672d530e8a598030efba389bd4e2e656a74e49a1c5fb4b`
- `interact.cls`: publisher class v1.05, 2017-07-31.
- `tfnlm.bst`: publisher NLM bibliography style, revision 2016-08-10.
- Publisher-authored template and documentation:
  <https://www.overleaf.com/latex/templates/taylor-and-francis-latex-template-for-authors-interact-layout-plus-nlm-reference-style/bngwgqnxcxrp>
- Current NLM guide: <https://files.taylorandfrancis.com/tf_NLM.pdf>
  (version 2.2, 2023-05-30).

The two vendor files are retained without code changes; line endings and
trailing whitespace have been normalized. Standard packages distributed in the bundle are
provided by the user's TeX distribution instead of vendoring old versions.
The template on Overleaf is attributed to Taylor & Francis under CC BY 4.0.
The `tfnlm.bst` header separately permits redistribution under LPPL version 1
or later and retains its original copyright and license notice. These vendor
files are not relicensed under the repository's MIT license.

The manuscript adapts the publisher's front matter and numerical natbib setup.
The publisher's `tfnlm.bst` does not emit BibTeX `doi` fields, so the bibliography
retains those fields and also supplies DOI links through its supported `note`
field. Explicit author names plus numerical citations replace `\citet`, because
this style does not emit natbib author labels. Bibliographic data remain
editable in `references.bib`.
