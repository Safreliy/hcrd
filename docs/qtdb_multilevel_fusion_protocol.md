# QTDB multilevel HCRD fusion protocol M1

Status: fixed before running M1.

Date fixed: 2026-08-24.

This is a **post-R2 development experiment**, not a fresh confirmation. The
80 evaluation records were previously used to evaluate the frozen one-level
wrapper, although no multilevel features or learned fusion results have yet
been inspected. Any positive M1 result therefore requires confirmation on an
independent dataset before it can support a superiority claim.

## Question

Given the same expert QRS fiducial available to the original HCRD wrapper, do
nested HCRD levels contain boundary information beyond (a) a constant QRS
width learned from pilot records and (b) level 1 alone?

## Split and information boundary

- Training: the 25 records in `data/qtdb/record_split.json::pilot`.
- Evaluation: the 80 records in `confirmation_locked`.
- No beat from an evaluation record is used to fit preprocessing, models, or
  boundary-width constants.
- Both ECG leads may be used by HCRD. The expert QRS fiducial is supplied to
  every learned method, matching the information given to the earlier HCRD
  wrapper. Distributed `pu0`/`pu1` boundaries remain external single-lead
  comparators.

## Fixed representation

For each lead, a quadratic guide with the already pilot-selected
regularization `lambda=10` is decomposed through at most four HCRD levels.
At every available level, anchor-and-grow candidates are computed at amplitude
ratios 0.1, 0.2, and 0.3. Each candidate contributes onset and offset relative
to the supplied fiducial, width, success, log normalized anchor amplitude, and
log structure count. Missing terminal levels are encoded as missing values.

Two representation ablations use exactly the same learner:

1. `learned_level1`: both leads, level 1, all three growth ratios;
2. `learned_multilevel`: both leads, levels 1--4, all three ratios.

The negative control `pilot_constant` predicts the median pilot onset and
offset relative to the expert fiducial.

## Fixed learner and metric

Separate LightGBM L1 regressors predict onset and offset in milliseconds:
400 trees, learning rate 0.03, 7 leaves, maximum depth 3, minimum 30 samples per
leaf, feature fraction 0.8, L2 regularization 1, seed 20260824. Predictions are
clipped to the fixed 140-ms pre-fiducial and 180-ms post-fiducial windows and
rounded to the record's sampling grid.

Primary loss is the macro-average across evaluation records of mean joint
onset/offset absolute error. The primary M1 contrast is
`learned_multilevel - learned_level1`; secondary contrasts are against
`pilot_constant`, `pu0`, and `pu1`. Uncertainty uses a paired record bootstrap
and an exact sign test. Failure is penalized by 160 ms as in R2.

## Evaluation rule

M1 establishes useful multilevel information only if the 95% record-bootstrap
interval for multilevel minus level 1 lies below zero and multilevel also beats
the pilot constant. Beating an `ecgpuwave` channel would be a promising
development result, not a confirmatory state-of-the-art result. All outcomes,
including a negative result, remain in the public release.
