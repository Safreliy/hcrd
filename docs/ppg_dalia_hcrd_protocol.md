# Frozen PPG-DaLiA HCRD benchmark protocol

Status: frozen before HCRD feature extraction or any local detector outcome.

Author: Saveliy Baturin, Independent Researcher

## Driving question

Can the full HCRD component trajectory provide a practically useful,
accelerometer-free representation for detecting pulse beats in wrist PPG under
daily-life motion artifacts? The primary unproved claim is that learned
multilevel geometry generalizes to unseen people better than a tuned
single-scale peak detector. Polygon/triangle area and quadratic energy are
secondary coordinates, not the representation itself.

## Data and immutable outer evaluation

Use the activity-wise MATLAB release of PPG-DaLiA from Zenodo record 12793711
(derived from the CC BY 4.0 UCI dataset). It contains 15 subjects and eight
activities: car driving, cycling, lunch break, sitting, stair climbing, table
soccer, walking, and working. The ECG `rpeaks` supplied with the release are the
reference annotations. Convert each one-based ECG index to time and then to the
64 Hz PPG sample grid.

Sort subject IDs by SHA-256 and form five outer folds of three subjects:

1. S9, S13, S5
2. S1, S10, S3
3. S11, S8, S2
4. S6, S4, S12
5. S7, S15, S14

For outer fold `f`, its subjects are test, fold `(f+1) mod 5` is validation,
and the remaining nine subjects are development. Each subject is test exactly
once. No subject identity, activity label, ECG, or accelerometer value is an
input feature.

## Signal, references, and candidate labels

Use the frozen PPGopt conditioning and candidate extractor: zero-phase
fourth-order 0.5--15 Hz Butterworth bandpass, robust MAD scaling, 30 s HCRD
windows with 2 s halos, eight levels, and all 140 geometry features. The seven
local morphology variables remain a separately declared optional block. Mark
non-finite or flat-envelope runs longer than 200 ms invalid.

As in PPG-beats, compensate ECG-to-PPG pulse-transit delay separately for every
evaluated detector in consecutive blocks of 300 ECG beats. Search lags from
-10 to +10 s in 20 ms increments, maximizing reference beats within 150 ms;
ties choose the smallest absolute lag. Training labels use a provisional
`find_peaks` detector with 300 bpm maximum and prominence 0.5. Each aligned ECG
beat labels at most one nearest HCRD candidate within 150 ms.

Apply 200 ms non-maximum suppression to every HCRD model. Exclude a record only
if the provisional alignment leaves fewer than three reference beats; report
all exclusions.

## Fixed representations and learners

Model capacities transfer unchanged from PPGopt/MIMIC; no capacity search is
allowed on PPG-DaLiA:

1. mass-only diagnostic HGB: 39 area/energy coordinates, learning rate 0.1,
   31 leaves, 200 iterations, L2=1;
2. geometry logistic: all 140 HCRD coordinates, standardized, C=0.1;
3. geometry HGB: all 140 HCRD coordinates, learning rate 0.1, 31 leaves,
   200 iterations, L2=1;
4. hybrid HGB: 140 HCRD plus seven local morphology coordinates, learning
   rate 0.05, 15 leaves, 200 iterations, L2=1.

All classifiers use balanced class weights. For each outer fold, fit candidates
from its nine development subjects. On its three validation subjects select
only an event probability threshold in `{0.05, 0.10, ..., 0.95}` by median
subject-activity F1, then micro-F1, precision, and higher threshold. The
mass-only control is not eligible to become primary. Among eligible models,
select the outer-fold primary by the same criteria, breaking ties for
geometry-only and then logistic regression. This selection is performed before
scoring that fold's test subjects.

## Local and literature baselines

- P0: `find_peaks` on the identical conditioned signal; validation chooses
  maximum bpm in `{160, 180, 200, 240, 300}` and prominence in
  `{0.1, 0.2, 0.35, 0.5, 0.75, 1.0}`.
- P1: deterministic HCRD persistence with minimum supporting-level count in
  `{1,...,5}`, selected on validation.
- HeartPy 1.2.7 with 30--240 bpm bounds and no activity-specific tuning.
- Published PPG-beats activity results, including MSPTD/qppg and MSPTDfast v2,
  are contextual until their GPL MATLAB implementations are run through the
  same local reference and exact matcher.

## Metrics and success rule

The primary metric is median exact one-to-one F1 across all available held-out
subject-activity records pooled from the five outer folds. Report IQR,
micro-precision/recall/F1, timing error, false positives per minute, the median
for every activity, and a macro median over the three motion-intensive
activities walking, stair climbing, and table soccer. Also report the
PPG-beats-compatible nearest-beat metric as a sensitivity analysis.

A strong practical result requires the cross-fitted HCRD primary to:

1. exceed the strongest local baseline in overall median and micro-F1;
2. improve at least five of eight activity medians;
3. improve the motion-intensive three-activity macro median;
4. beat the mass-only control materially;
5. have a positive 95% subject-bootstrap interval versus the strongest local
   baseline.

Failure is a domain limitation. No outer-test-derived parameter change is
confirmatory.

## Pre-feature source-completeness amendment D1

Exporting the immutable Zenodo files, before HCRD extraction or detector
scoring, showed that the activity-wise release contains 117 rather than 120
subject-activity records: S6 has no lunch-break, walking, or working record.
These records are absent from the source files rather than removed by a quality
rule. Evaluate all 117 available records. The exported source contains 137,104
manual ECG R peaks over 26.2244 hours.

## Pre-outcome alignment-conformance amendment D2

Inspection of the current official PPG-beats MATLAB source before any local
PPG-DaLiA detector score showed that its nominal 300-beat alignment blocks
expand the last block to the end of the record rather than creating a shorter
remainder block. The local implementation and a regression test were updated
to reproduce that behavior exactly before regenerating candidate labels.
