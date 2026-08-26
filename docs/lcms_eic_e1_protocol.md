# E1: expert-labelled LC--MS peak-shape confirmation

Protocol frozen on 2026-08-25 before downloading or inspecting the EIC arrays
or their classification tables.

## Scientific question

Does the multilevel HCRD hierarchy add out-of-sample information for deciding
whether a real extracted-ion chromatogram (EIC) contains a genuine
chromatographic peak?  The intended object is a compact, predominantly
one-signed lobe above a locally changing baseline.  This is a mechanistic test
of HCRD, not another generic anomaly-detection benchmark.

## External data and labels

Use version 3 of Mueller et al.'s 2020 dataset, Zenodo DOI
`10.5281/zenodo.3756211`: 255,000 EICs formed by 5,000 candidate features in 51
real LC--MS samples.  The primary labels are
`Classification_before_cleanup.csv`, because they are the experts' shape-only
judgements and do not include the later intensity/scan-length rules.

Primary binary endpoint:

- definite peak (the expert cell was left empty) -> 1;
- definite non-peak (expert code 1) -> 0;
- inconclusive (expert code 0) -> excluded from the primary binary endpoint.

The post-cleanup table is secondary only.  The raw data and labels are not
redistributed by the HCRD repository; the download script records the source
DOI, archive MD5 and local SHA-256.

## Double group holdout

The same candidate feature occurs in many samples and every sample contains
many candidates.  A random EIC split would therefore leak both chemical-feature
and sample context.  We use a conservative Cartesian holdout.

Normalize every sample name and peak ID with Unicode NFKC, surrounding
whitespace removal and lower-casing.  Compute

`SHA256("hcrd-e1-v1|axis|normalized_identifier")[0]`,

where `axis` is `sample` or `peak`.  Assign each axis independently:

- byte 0--153: train;
- byte 154--204: validation;
- byte 205--255: confirmation.

An EIC is used only when its sample and peak assignments agree.  Cross-block
cells are unused.  Thus training, validation and confirmation share neither a
sample nor a candidate peak.  If an identifier cannot be matched unambiguously
between the EIC object and the classification table, that EIC is excluded and
reported.

Confirmation labels must not be loaded by the development command.  The
strongest comparator and the probability threshold are written to a frozen
selection JSON before the explicit confirmation command is allowed to run.

## Frozen representations

The dataset supplies the same EIC in the two windows shown to the experts
(approximately +/-7 s and +/-14 s).  Compute every frozen representation below
independently on both windows and concatenate the two vectors, so all methods
receive the same information as the annotator.  All signals are ordered by
retention time.  Duplicate times are aggregated by their median intensity.
Non-finite pairs are removed.  An EIC is excluded if either window has fewer
than 8 distinct time points.  For fixed-length channels, interpolate on 64
equally spaced positions over each observed window.

`RAW64` (strong representation baseline):

- 64 robustly normalized intensity samples;
- log amplitude and log positive trapezoid area before normalization;
- center of mass, peak location, half-height width, left/right width ratio;
- total variation, second-difference variation, zero fraction;
- maximum SciPy prominence and number of local maxima.

`DOMAIN` (non-neural signal-processing baseline) adds, at fixed scales
1, 2, 4 and 8 samples:

- Gaussian/Laplacian-of-Gaussian response summaries;
- white top-hat response summaries;
- peak-prominence and width summaries.

`HCRD-1` adds only level 1 to `RAW64`.  `HCRD-8` adds the complete hierarchy up
to eight levels to `RAW64`.  Each available HCRD level contributes its
resampled signed detail plus structure count, knot fraction, positive and
negative polygon area, triangle area, quadratic energy, maximum amplitude,
area concentration, signed-area balance, duration quantiles and the location,
duration, sign, amplitude, polygon area and shape factor of the four largest
structures.  Missing levels are zero-padded and accompanied by an availability
flag.  A final trend channel and hierarchy depth are included.  HCRD is fit to
the original, nonuniform retention-time coordinates; signal scale is retained
in the scalar area/amplitude features.

`HCRD-GEOMETRY` contains the HCRD-8 scalar structure bank but not the resampled
raw or detail channels.  `AREA-ONLY` contains just the per-level positive,
negative and total polygon areas, triangle areas and their cross-level changes.
These two are ablations, not eligible comparator families.

## Frozen learner and selection

Use the same non-neural learner for every representation:

`HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250,
max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
class_weight="balanced", random_state=20260825, early_stopping=False)`.

No HCRD or learner hyperparameter is chosen on confirmation.  On validation,
select the strongest eligible non-HCRD comparator from `RAW64` and `DOMAIN` by
average precision (AP), with ROC AUC then lexical name as deterministic ties.
Choose one probability threshold per fitted representation by maximum
validation MCC, breaking ties toward the larger threshold.  Freeze model,
comparator name, threshold, data hashes, feature schema and package versions in
`selection_frozen.json` before confirmation.

## Endpoints and inference

Primary endpoint: confirmation AP difference, `HCRD-8 - frozen comparator`.
The pre-specified success criterion requires all of:

1. positive AP difference;
2. a positive 95% two-way cluster-bootstrap confidence lower bound, resampling
   confirmation samples and peak IDs independently (10,000 replicates,
   seed 20260825);
3. HCRD-8 AP greater than HCRD-1 AP, establishing a benefit from multiple
   levels rather than merely a one-level wrapper.

Secondary endpoints are ROC AUC, MCC, balanced accuracy and F1 at the frozen
threshold, plus AP for HCRD-GEOMETRY and AREA-ONLY.  Pairwise secondary tests
against the comparator use Holm correction.  Report prevalence, exclusions,
sample/peak group counts, point estimates, cluster intervals and all negative
results.  Runtime is descriptive only and is not a scientific endpoint.

## Interpretation

A success supports expert-labelled EIC peak-shape classification and the
mechanistic class of isolated, predominantly one-signed convex lobes.  It does
not by itself establish superiority for generic anomaly detection, ECG QRS
detection, oscillatory spikes, full LC--MS feature finding or downstream
compound identification.
