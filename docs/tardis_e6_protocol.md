# E6: frozen external targeted-LC--MS transfer

**Lock time:** 2026-08-25, before any row of the released manual-label tables
was inspected and before any TARDIS chromatogram was reconstructed.

## Question

Does the compact multilevel HCRD representation learned on the fully labelled
Falkor and MESOSCOPE studies transfer, without target fitting, to manually
reviewed targeted LC--MS panels from a different laboratory, sample matrix and
processing workflow?

E6 is a no-refit external-transfer experiment.  It is not a new search over
HCRD levels, classifiers, windows or endpoints.

## Frozen sources

- Vangeenderhuysen et al. (2025), TARDIS, Analytical Chemistry,
  DOI `10.1021/acs.analchem.5c00567`.
- Publication-analysis repository:
  `https://github.com/UGent-LIMET/tardis_publication_code`.
- Input-data record: Zenodo DOI `10.5281/zenodo.14548033`, archive
  `data_zenodo.rar`, MD5 `38ddb2822551b1d281b57d610eb56986` and released size
  5,669,337,245 bytes.
- Primary cohort: FAME saliva.  Fixed replication order: ENVIRONAGE urine,
  FGFP metabolomics, FGFP lipids.

The primary cohort was chosen from published study-level metadata, before row
inspection, because the source paper documents complete manual review of its
target panel in representative QC runs.  Replication cohorts will be included
whenever they pass the waveform-availability gate below; no cohort may be
included or excluded based on HCRD performance.

The extracted label-table bytes are pinned as follows:

| file | bytes | SHA-256 |
|---|---:|---|
| `fame_feat_labeled.csv` | 63,013 | `5436b248730ec12d37f5729ebaf00c152fc30d55abd69f8bd8c794cf330e0e8e` |
| `environage_labeled.csv` | 51,961 | `943bc33e35761e2052baed9d64db6fe1522a55d33a7d5603e9d7e5735b83db5e` |
| `metabo_fgfp_labeled.csv` | 73,634 | `72f6fa23c9c87f1b2a0b97f4d17c98ce3f5d795be16aee4ff6c6fca2cc0399b2` |
| `fgfp_lipids_labeled.csv` | 17,054 | `fe43b1df1bf3815af59ca741d097392e494f9d70b7fd4ac0e9ad70f593b3738d` |

## Availability gate

A cohort is evaluable only if the released material supplies all of:

1. a one-to-one mapping from every manual rating to target identifier, m/z,
   expected retention time and polarity;
2. raw mzML/mzXML QC files or numerical EIC arrays from which the rated
   chromatograms can be reconstructed;
3. at least eight distinct scans in an EIC and at least 20 unambiguous targets
   in each of the `Good` and `Bad` classes.

Failure of this gate is reported as a data-availability no-go, not repaired by
using TARDIS summary metrics as substitutes for HCRD waveforms.  `Ambiguous`
targets are excluded from the primary binary endpoint before any score is
computed and retained only for the ordered-label sensitivity analysis.

### Pre-score availability amendment (2026-08-25)

Archive inspection after the protocol lock, but before any HCRD feature or
model score was computed, established that `scale_analysis/files700` contains
700 positive-polarity mzXML files, including 119 filenames marked as QC.  The
archive supplies negative-polarity diagnostic PNGs and summary tables, but no
negative raw files or numerical EIC arrays.  Consequently the full FAME panel
fails item 2 of the waveform gate, whereas its positive-polarity stratum is
evaluable.  E6 therefore uses all 119 released positive QC files and all rated
positive targets satisfying the remaining gate criteria.  Negative targets
are reported as unavailable, never approximated from plots, and not excluded
on the basis of an HCRD result.  The fixed representation, learner, endpoints,
success rule and bootstrap seed below are unchanged.

## Frozen waveform representation

For raw files, an EIC is the sum of MS1 intensity within 10 ppm of the released
target m/z and within a 60-second window centred on the released expected RT.
Polarity must match the released target.  No border, smoothing parameter or
window may be adjusted after labels or scores are viewed.  If the release
contains the exact numerical EICs used for manual review, those arrays take
priority over reconstruction; this availability branch is determined without
using ratings.

The per-EIC representations and target-level aggregation are exactly those
locked in LCMS-E2:

- qscore;
- DOMAIN+Q;
- HCRD-1+Q;
- HCRD-8+Q;
- the existing geometry-only and area-only sensitivity representations;
- across available QC files: coordinatewise median, 0.9 quantile, maximum,
  availability fraction and median qscore.

All non-finite values are replaced exactly as in E2.  There is no target-cohort
feature selection, scaling, imputation choice or representation change.

## Frozen learner

For each representation, fit the existing E2 logistic pipeline once on the
pooled unambiguous Falkor and MESOSCOPE targets: `StandardScaler` followed by
balanced L2 logistic regression with `C=1`, `liblinear`, seed 20240229 and at
most 5,000 iterations.  Apply it to every evaluable TARDIS cohort without
refitting or recalibration.  A model output is the probability of `Good`; the
prespecified residual-triage score is `1 - P(Good)`.

## Endpoints and success rule

The primary endpoint is target-level average precision for `Bad`.  On FAME,
HCRD-8+Q succeeds only if both paired differences have strictly positive
percentile 95% bootstrap intervals:

1. HCRD-8+Q minus qscore;
2. HCRD-8+Q minus HCRD-1+Q.

Use 20,000 multinomial paired bootstrap replicates over targets with seed
20260825.  The two requirements are co-primary and must both pass.  The frozen
secondary non-inferiority comparison is HCRD-8+Q minus DOMAIN+Q with margin
-0.02.  Report AP, ROC AUC, top-decile Bad enrichment, all paired intervals and
two-sided bootstrap p-values for every cohort whether favourable or not.

Replications are interpreted in their fixed order.  A fixed-effect inverse-
variance summary across evaluable replication cohorts is descriptive; no
failed cohort is removed.  The ordered-label sensitivity endpoint is Spearman
correlation between residual-triage score and `Good < Ambiguous < Bad`.

## Interpretation

Passing E6 supports no-refit transfer for quality triage of the released
target panels.  It does not estimate untargeted feature-detection recall, prove
population-wide FDR control, validate data-dependent tuning, or replace a raw
chromatogram benchmark if the availability gate fails.
