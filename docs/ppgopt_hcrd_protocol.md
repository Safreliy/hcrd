# Frozen PPGopt protocol: multilevel HCRD pulse detection

Status: frozen before any HCRD-vs-label evaluation on this dataset.

Author: Saveliy Baturin, Independent Researcher

## Question and scope

This experiment tests the class for which HCRD's geometry is structurally
aligned with the target: detecting repeated rounded pulse events and their
fiducial locations in raw reflective-mode PPG. It does **not** claim that HCRD
is a universal time-series representation.

The primary hypothesis is unproved:

> A pulse supported by several nested signed chord structures can be separated
> from noise and motion-induced local maxima more reliably than by a
> single-scale amplitude or moving-average rule.

The experiment must retain a pure structural ablation, a learned structural
model, and a conventional non-HCRD baseline. Area is one feature family; it is
never used as a replacement for the decomposition.

## Data and immutable subject split

- Raw data: Biagetti et al. MAXREFDES100# PPG/ACC dataset, 400 Hz.
- Labels and artifact intervals: Wolling et al. PPGopt annotation release.
- Development subjects: S1, S2, S3.
- Validation subjects: S4, S5.
- Locked confirmation subjects: S6, S7.
- The confirmation labels must not be scored until the candidate extractor,
  model family, hyperparameter grid, threshold rule, matching code, and all
  ablations are frozen from development and validation data.

The activity labels (rest, squat, step) are used only for stratified reporting,
not as classifier inputs. The unusable S5/squat3 recording and any recording
without expert peaks are excluded. Samples inside annotated artifact intervals
are masked. Expert peaks inside the same intervals are not scoring targets.

## Signal conditioning

The primary input is the raw PPG channel. Following the source benchmark, use
a zero-phase fourth-order Butterworth bandpass of 0.5--15 Hz. No accelerometer
channel is used in the primary experiment. This makes the comparison stricter
and keeps the claimed mechanism morphological. ACC-assisted rejection may be
reported later as a separately labelled extension.

Robust scaling subtracts the median and divides by `1.4826 * MAD`; if MAD is
zero, standard deviation is used. Conditioning is fit independently within
each recording and uses no labels.

## Model P0: conventional peak baseline

Definition: local maxima of the conditioned signal are selected with
`scipy.signal.find_peaks`. The development-only grid contains refractory
distances corresponding to 200, 180, 160, and 140 bpm and prominence
thresholds of 0.1, 0.2, 0.35, 0.5, 0.75, and 1.0 robust standard deviations.
The parameter pair maximizing micro-F1 on S1--S3 is frozen and reported on S4,
S5, then S6--S7. HeartPy with its documented defaults and the 30--200 bpm
range is a second conventional baseline.

Demonstration: an isolated clean pulse should yield one local maximum.

Tunable parameter: prominence threshold.

Expected observation: P0 should perform strongly at rest but produce surplus
candidates during motion.

False intuition exposed: a visually prominent maximum is not necessarily a
physiological pulse.

## Model P1: deterministic multilevel HCRD persistence

Definition: split the conditioned recording into 30 s windows with a 2 s halo.
Compute sparse HCRD to at most eight levels in each window. Retain positive
structures with durations from 50 ms to 2 s, map each structure maximum to the
nearest conditioned-signal maximum within 50 ms, and cluster mapped positions
within 80 ms. A cluster is an event candidate. Its deterministic score is the
number of distinct consecutive levels supporting it, followed by normalized
summed amplitude as a tie-breaker. Non-maximum suppression enforces a 300 ms
refractory interval. Development data select one minimum-persistence value
from 1--5; no other threshold is tuned.

Demonstration: a genuine rounded pulse is expected to survive in nested chord
structures while a narrow high-frequency perturbation is expected to disappear
after the first levels.

Tunable parameter: minimum number of supporting HCRD levels.

Expected observation: persistence should improve precision over single-level
HCRD without destroying recall at rest.

False intuition exposed: a large polygonal mass alone proves that an event is a
pulse. Baseline wander and motion can also have large area.

## Model P2: learned full HCRD structure bank

Definition: use exactly the candidates from P1 before persistence thresholding.
For every candidate and each of eight levels, record:

- support count and sign;
- left/right boundaries, duration, and candidate position within the support;
- residual amplitude and amplitude relative to the level and recording;
- exact polygon area, signed area, quadratic energy, triangle surrogate, and
  shape factor;
- distances between the candidate and the structure peak/midpoint;
- corresponding negative-structure context on both sides.

Cross-level features record persistence, longest consecutive run, first and
last supporting level, support-position dispersion, boundary dispersion,
duration growth, amplitude/area/energy decay, total signed and unsigned mass,
and the level at which each maximum occurs. A small raw-morphology block
(prominence, left/right slope, width, and local curvature) is included only for
the declared hybrid ablation.

Three classifiers are compared on identical candidates:

1. HCRD geometry only: class-balanced logistic regression.
2. HCRD geometry only: histogram gradient boosting.
3. Hybrid: histogram gradient boosting with the raw-morphology block.

Each expert peak is assigned to at most one nearest candidate within 250 ms;
that candidate is positive and all remaining candidates are negative. Model
selection is grouped by recording and subject. S1--S3 select a small frozen
grid; S4--S5 select the probability threshold. The primary model is the best
validation micro-F1 model, with ties resolved in favor of geometry-only and
then the simpler classifier.

Demonstration: P2 should learn that a pulse is a *trajectory through the HCRD
hierarchy*, not a single triangle.

Tunable parameter: classifier probability threshold after the classifier grid
has been selected on development subjects.

Expected observation: geometry-only P2 should exceed P1; the hybrid should be
competitive with or better than P0 under squat and step motion.

False intuition exposed: exact reconstruction implies discriminative
sufficiency. It does not; the event-selection map can still discard or confuse
relevant structures.

## Matching and metrics

Matching is one-to-one and maximizes the number of pairs within +/-250 ms;
ties minimize total absolute timing error. A prediction in a masked artifact
interval is ignored, as is a target in that interval. Primary metric is
micro-averaged event F1 over recordings. Report precision, recall, mean and
median absolute timing error, false positives per minute, throughput, and 95%
subject-level bootstrap confidence intervals. Report macro-F1 by subject and
by activity as secondary metrics.

Published all-data optima (Karlen F1 0.958 and van Gent/HeartPy F1 0.970) were
jointly optimized over the entire benchmark and are context, not held-out
comparators. Our direct held-out comparison is against P0 and HeartPy run by
this repository on the same subject split and scoring implementation.

## Success and stop rules

The main practical claim requires all of the following on locked S6--S7:

1. higher micro-F1 than both locally run conventional baselines;
2. no activity stratum worse by more than 0.01 F1;
3. a subject-level bootstrap interval for the F1 improvement whose lower bound
   is above zero, or replication on a second public PPG dataset;
4. a geometry-only ablation that materially exceeds deterministic P1, showing
   that the structure bank, not only generic boosting, carries the gain.

If these conditions fail, the result is reported as a falsification or a
trade-off. No post-confirmation parameter change may be called confirmatory.

## Pre-outcome data-format amendment A1

The original protocol file had SHA-256
`f7619437297e9b46a2206249a476332c8680f4fe05940beda0078bf61a2dfdce`.
During label-free feature extraction, S2/squat3 was found to contain 39 isolated
non-finite sensor values at regular one-second positions. Before any aggregate
development result or parameter selection, deterministic linear interpolation
of non-finite samples was added to signal conditioning. This amendment changes
only missing-value handling; all candidate, split, model, and metric rules above
remain frozen.

## Pre-outcome implementation amendment A2

Before aggregate development scoring, the small classifier grid and its tie
rules were made explicit. Logistic regression uses standardized features,
balanced class weights, `max_iter=2000`, and `C` in `{0.1, 1, 10}`. Histogram
gradient boosting uses balanced class weights, `max_iter=200`, learning rate in
`{0.05, 0.1}`, maximum leaf nodes in `{15, 31}`, and L2 regularization fixed to
1. Hyperparameters are selected by pooled leave-one-development-subject-out
average precision, a threshold-free criterion; ties choose the smaller model.
The final probability threshold is selected on S4--S5 from
`{0.05, 0.10, ..., 0.95}` by event micro-F1, with higher precision and then the
higher threshold as tie-breakers. Event NMS remains fixed at 300 ms.

## Pre-validation data-format amendment A3

Label-free extraction revealed that every S4 MAT file has a third, identically
zero column, whereas all other subjects have only time and PPG columns. The
loader now accepts such zero padding and continues to use column 0 as time and
column 1 as PPG, as specified by the source data description. Any additional
nonzero column remains an error. No validation label or validation score had
been read when this amendment was made; development features are unaffected.
