# Statistics submission preparation

Checked on 2026-09-05 against the journal's Instructions for Authors, updated
2026-06-22:
<https://www.tandfonline.com/action/authorSubmission?journalCode=gsta20&show=instructions>.
This is **Statistics**, journal code `gsta20`, not Journal of Applied Statistics.

## Format and contents

The journal accepts format-free initial submissions, including LaTeX and PDF,
and allows figures and tables within the text. It does not currently mandate
a journal-specific LaTeX class. This manuscript voluntarily uses the publisher's
Interact + NLM template, consistent with the journal's linked reference style.
The publisher applies the final journal layout after acceptance.

- Original article in English, with one named author, affiliation, and
  corresponding-author email. Single-anonymous review does not require an
  anonymized manuscript.
- Unstructured abstract: 150 words (hyphenated terms and numeric ranges each
  counted as one word). Five keywords.
- A declared TeXcount word count, including appendices, text, headings, and
  captions; excluding references, mathematical expressions, and the count
  declaration itself. There is no overall journal word limit.
- Numerical citations in order of first appearance, NLM bibliography,
  abbreviated journal names, and DOI links where already available.
- References before appendices. All four figures and five tables have explicit
  numbered callouts in the body, before their floats and in numerical order.
  The previously missing DNase figure callout has been added.
- Three vector figures; the colour DNase raster has approximately 306 effective
  dpi at its typeset width. No image was upsampled. Tables remain editable LaTeX.
- Funding, Disclosure of interest, CRediT contribution statement, Declaration
  of generative AI use, Notes on contributor, and Data availability statement.
  The author confirmed sole authorship and no funding on 2026-09-05.
- Existing mathematical claims, experimental results, and uncertainty
  qualifications have not been strengthened by this formatting revision.

The generic structure section also lists end-placed tables and figures. We use
the explicitly permitted inline layout under Format-Free Submission to keep
the mathematical and empirical discussion readable during peer review.

## Before the author submits

- Confirm the short biographical note and current affiliation/email; add an
  ORCID if available. No ORCID, degree, employer, or funding source was invented.
- Review the existing generative-AI disclosure and responsibility statement.
- The repository hyperlink is supplied for data and code. A permanent archival
  DOI is encouraged by the journal but is not mandatory under its Basic Data
  Sharing Policy; no DOI deposit has been made as part of this formatting task.
- Decide whether to request open access or paid colour in print during the
  submission process. No paid option has been selected.
- Upload the PDF and editable source package through the journal's Submission
  Portal. This preparation task does not submit the paper to the journal.

## Reproducibility and verification

See `README.md` for build, word-count, and packaging commands.
`check_submission.py --compiled` checks every float callout, all bibliography
entries, required sections, the abstract length, keywords, effective raster
resolution, and unresolved references or overfull boxes in the final log.
Visual PDF inspection is still required; a passing script is not a visual audit.
Scientific frozen artifacts are checked separately from the repository root
with `python scripts/verify_sci_artifact.py`.
