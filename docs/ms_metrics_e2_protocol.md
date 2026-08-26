# E2: cross-dataset HILIC mass-feature quality transfer

Protocol frozen on 2026-08-25 after the E1 result and after inspecting only the
published repository schema, class counts and ZIP member names, but before
downloading or inspecting any raw Falkor or MESOSCOPE waveform.

## Question

Does a multilevel HCRD representation improve an interpretable classifier of
expert-approved chromatographic mass features when the classifier is trained
on one real HILIC LC--MS study and transferred without refitting to another?
This is an independent external replication of the EIC mechanism, not a
re-analysis of Mueller et al.'s E1 data.

## Sources and fixed populations

Use the public data and manual labels from Kumler, Hazelton and Ingalls (2023),
"Picky with peakpicking", BMC Bioinformatics, DOI
`10.1186/s12859-023-05533-4`, repository commit recorded by the runner.

- Falkor: Metabolomics Workbench study ST002788, positive-mode ZIP; 41 mzML
  files; `made_data_FT2040/classified_feats.csv`.
- MESOSCOPE: study ST002789, positive-mode ZIP; 168 mzML files;
  `made_data_MS3000/classified_feats.csv`.

Primary binary labels are `Good` (1) and `Bad` (0).  `Ambiguous` and `Stans
only` are excluded before fitting or scoring.  Both transfer directions are
fixed endpoints: Falkor -> MESOSCOPE and MESOSCOPE -> Falkor.  No random
within-dataset train/test split is a primary result.

## Raw EIC extraction

For each manually classified mass feature, use its published global
`min_mz`, `max_mz`, `min_rt`, and `max_rt` bounding box.  For every mzML file,
retain MS1 points whose retention time and m/z lie in that closed box.  At a
scan with multiple retained m/z points, sum their intensities.  As in the
source paper, absent scans are not replaced by zero; known points are connected
linearly.  Duplicate retention times are aggregated by median.  A file-level
EIC with fewer than 8 distinct time points is marked unavailable.  Raw source
ZIP and extracted cache hashes are recorded and neither is redistributed.

This fixed global-box extraction is applied identically to both datasets.  It
is intentionally called the `global-window qscore` rather than claimed to be
bit-for-bit identical to the authors' per-detected-peak qscore.  On Falkor,
where the repository includes the authors' extracted metrics, report the
correlation between our qscore and their published `med_cor`/`med_SNR` as a
fidelity audit.

## Frozen per-file representations

For every available file-level EIC compute the same single-window feature bank
as E1:

- `RAW`: 64 normalized samples and 11 amplitude/area/shape scalars;
- `DOMAIN`: RAW plus Gaussian/LoG, white-top-hat and prominence summaries at
  scales 1, 2, 4 and 8;
- `HCRD-1`: RAW plus the first signed detail and first-level structure bank;
- `HCRD-8`: RAW plus all signed detail channels, final trend and the full
  eight-level structure hierarchy;
- `HCRD-GEOMETRY` and `AREA-ONLY`: the same ablations as E1.

Also compute the source paper's qscore formula on the same EIC: the maximum
correlation with beta densities `(alpha, beta)=(2.5,5),(3,5),(4,5),(5,5)` and
their residual-based SNR.  Nonfinite qscore values are treated as missing.

At the mass-feature level aggregate each per-file vector elementwise by median,
90th percentile and maximum across available files, then append the available
file fraction.  This produces a fixed representation independent of the number
of files in a study.  The source-style baseline contains median qscore
correlation and median qscore SNR only.  `DOMAIN+Q`, `HCRD-1+Q`, `HCRD-8+Q`,
`HCRD-GEOMETRY+Q`, and `AREA-ONLY+Q` append those two qscore values.  Thus the
primary comparison measures the incremental value of the HCRD hierarchy over
the strongest simple shape prior rather than replacing it.

## Frozen learner

Every representation uses the same pipeline:

`StandardScaler()` followed by
`LogisticRegression(C=1, penalty="l2", solver="liblinear",
class_weight="balanced", max_iter=5000, random_state=20260825)`.

Fit on all unambiguous labels of the source dataset.  Apply without calibration,
threshold selection or refitting to the other dataset.  Probability 0.5 is the
fixed classification threshold.

## Endpoints and success rule

Primary metric is average precision (AP); ROC AUC is secondary.  At threshold
0.5 also report MCC, balanced accuracy, F1, false discovery rate and good
features found (recall).  For each transfer direction use 10,000 paired feature
bootstrap replicates (seed 20260825) for AP differences.  Holm-correct the two
directional primary p-values.

Pre-specified E2 success requires in both transfer directions:

1. `HCRD-8+Q` AP exceeds qscore-only AP;
2. the paired 95% bootstrap lower bound is positive after reporting both raw
   and Holm-adjusted p-values;
3. `HCRD-8+Q` AP exceeds `HCRD-1+Q` AP.

`DOMAIN+Q` is an additional strong non-neural comparator.  All ablations and
negative directions are retained.  Before any E2 score was computed, the
reporting code was also fixed to give paired feature-bootstrap intervals for
`HCRD-8+Q - DOMAIN+Q` and `HCRD-8+Q - HCRD-1+Q`, with Holm correction across
the two directions for each secondary contrast.  These secondary intervals do
not alter the primary success rule.  Runtime is descriptive only.

## Interpretation

Success supports transfer of multilevel convex-lobe geometry for mass-feature
quality assessment across these two HILIC studies.  It does not establish full
untargeted peak discovery, compound identification, performance on other
chromatographic modes or superiority to neural image classifiers.
