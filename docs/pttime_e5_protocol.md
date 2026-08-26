# E5: conditional external verification on Pttime

Protocol frozen on 2026-08-25 after the successful E2 result and after
inspecting the published repository schema, annotation rule, class counts and
archive byte count, but before downloading the Pttime archive or inspecting
any Pttime chromatogram.  The representation, learner and endpoints below must
not be changed after target waveform inspection.

## Question and selection boundary

Does the E2-selected multilevel HCRD representation identify residual bad mass
features inside an independently acquired HILIC LC--MS dataset that the source
authors' two-variable quality model had already scored above 0.9?

Pttime is **not** a fully labelled external cohort.  Kumler, Hazelton and
Ingalls labelled only the 400 of 7,781 mass features selected by their model.
Those 400 contain 348 `Good`, 17 `Bad` and 35 `Ambiguous` labels.  Consequently:

- E5 is a conditional verification/reranking experiment on the selected set;
- it cannot estimate recall, calibration or FDR for all Pttime features;
- it cannot be combined with E2 as if Pttime were a random labelled cohort;
- the source paper reports that the annotator knew the features were expected
  to be good, so possible experimenter bias is retained as a limitation.

This restricted question is nevertheless practical: after a high-specificity
filter has proposed a compact list for downstream analysis, can HCRD move the
remaining false positives toward the top of a manual-review queue?

## Frozen sources and population

- Labels and feature boxes: `wkumler/MS_metrics`, commit
  `491deaf1d5f27f9d276e58acb4c1dfca2a2e21b9`, file
  `made_data_Pttime/classified_feats.csv`.
- Raw data: Metabolomics Workbench study ST002077, archive
  `https://www.metabolomicsworkbench.org/studydownload/ST002077_Pttime.zip`;
  HTTP `Content-Length` observed before download: 2,916,229,074 bytes.
- Training data and cached E2 representations: fully labelled Falkor
  (ST002788) and MESOSCOPE (ST002789), with hashes already recorded by E2.

The target evaluation includes `Good` and `Bad` only.  `Ambiguous` is excluded.
No Pttime label is used for representation choice, fitting, scaling,
hyperparameter selection, calibration or threshold selection.

### Pre-extraction source-schema correction (2026-08-25)

After the archive download, but before extracting any chromatogram or computing
any HCRD feature or target metric, archive inspection revealed 52 positive-mode
and 52 negative-mode mzML files.  The source Pttime peak-picking pipeline and
filename parser use the `..._POS_MSMS-v2_...` files; the published feature boxes
and manual labels therefore belong to that positive-mode analysis.  E5 uses
exactly those 52 POS files (basename contains `_POS_`) and excludes the 52 NEG
files.  This is a source-schema correction, not an outcome-dependent protocol
change: retaining both modes would apply positive-mode feature boxes to an
incompatible acquisition.  No target waveform had been inspected and no E5
score had been computed when this correction was recorded.

## Frozen extraction and representations

For each labelled Pttime feature and every one of the 52 POS raw files, extract an EIC from the
published closed global `(min_mz, max_mz, min_rt, max_rt)` box.  Sum multiple
intensities within a scan, aggregate duplicate retention times by median,
linearly connect observed points and mark EICs with fewer than eight distinct
times unavailable.  Apply the same normalization, 64-sample grid and feature
bank as E2.

The frozen representations are exactly E2's `qscore`, `DOMAIN+Q`, `HCRD-1+Q`,
`HCRD-8+Q`, `HCRD-GEOMETRY+Q` and `AREA-ONLY+Q`.  File-level features are
aggregated by median, 90th percentile and maximum, with availability appended.
HCRD-8 remains the pre-specified representation because it was selected before
Pttime waveform inspection and passed both E2 transfer directions.

## Frozen learner

For every representation, pool all unambiguous Falkor and MESOSCOPE labels and
fit the same E2 pipeline:

`StandardScaler()` followed by
`LogisticRegression(C=1, penalty="l2", solver="liblinear",
class_weight="balanced", max_iter=5000, random_state=20260825)`.

Apply the resulting model to Pttime without refitting.  The model output is a
good-feature probability; the residual-bad review score is `1 - p_good`.

## Frozen endpoints and success rule

The primary metric is bad-class average precision (AP-bad), because the
operational objective is to prioritize the 17 residual bad features for
review.  ROC AUC is secondary.  Also report the number of bad features in the
top 17 and in the top 5% of the review ranking, but do not use either as a
success criterion.

Use 10,000 paired, class-stratified target-feature bootstrap replicates: sample
17 bad and 348 good features with replacement within class on every replicate.
Report percentile 95% intervals and two-sided bootstrap p-values for all fixed
contrasts.

Pre-specified E5 success requires both:

1. `HCRD-8+Q` AP-bad exceeds qscore-only AP-bad with a positive 95% bootstrap
   lower bound;
2. `HCRD-8+Q` AP-bad exceeds `HCRD-1+Q` AP-bad.

`DOMAIN+Q`, geometry and area are secondary fixed ablations.  Every result is
retained regardless of direction.  There is one primary external target, so no
across-direction multiplicity correction is needed.

## Interpretation

If successful, E5 supports conditional transfer of multilevel HCRD morphology
to residual-error triage after a high-specificity HILIC feature-quality filter.
It does not establish performance on the full Pttime feature population,
unbiased external FDR, full peak discovery, other chromatographic modes or
superiority to neural peak pickers.
