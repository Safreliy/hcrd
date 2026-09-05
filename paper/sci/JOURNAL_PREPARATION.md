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
  ORCID if available. The biography uses the author's stated master's degree
  in applied mathematics and role as Head of Machine Learning at Postgres
  Professional. The author confirmed that this research was conducted
  independently, so the manuscript affiliation remains Independent Researcher.
  Employment is biographical context, not an institutional affiliation for
  this work. No ORCID, university, or funding source was invented.
- Review the existing generative-AI disclosure and responsibility statement.
- Review `cover_letter.pdf`. The author confirmed that the manuscript is not
  under consideration elsewhere; the letter now includes this statement.
  Answer any prior-publication/preprint questions accurately in the portal.
  No unconfirmed unpublished-work declaration has been added.
- The author archived software release `sci-v1.0.0` in Zenodo:
  <https://doi.org/10.5281/zenodo.22338783>. This version-specific DOI is now
  the primary code/data link in the manuscript and cover letter; GitHub
  remains the development repository. The release tag resolves to
  `2144b48ab2d21fdeb05c6dd9c40b4a9dbe93b28a`. It preserves the SCI numerical
  study, not the later editorial clarifications in the submission PDF.
  The Zenodo record retains the repository's historical HCRD archive title.
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
The `--zip` option forces a fresh compilation before creating the archive.
An independent build from a new extraction directory remains the final check
for self-contained sources.
Scientific frozen artifacts are checked separately from the repository root
with `python scripts/verify_sci_artifact.py`.

## Final scientific wording audit

- E33, E36, and E38 all use fixed equally spaced points or deterministic
  Beta(4,8) quantiles. No experiment was rerun to correct this description.
- SCI certifies mean-contrast signs, not convexity or concavity of a whole
  support. The introduction, explanation, caption, schematic labels, and
  prose-only spine now use the same interpretation.
- The schematic's final interval uses the same coordinate scale and bounds
  as the preceding one-sided constraints. Lobes remain geometric examples,
  not a converse to the chord-sign implication or a proof of coverage.
- Wide outputs are attributed to SCI's contrast family and calibration, not
  to a lower bound for every valid inference procedure.
- Affine cancellation states both the zeroth and first weighted moments.
- The abstract's runtime claim concerns contrast evaluation, not end-to-end
  inference. Numerical results and theorem statements are unchanged.
