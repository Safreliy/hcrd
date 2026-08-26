# PPG-DaLiA local-motion augmentation (D3)

Author: Saveliy Baturin, Independent Researcher

Date fixed: 2026-08-25

Status: post-outer-test exploratory development. The frozen PPG-DaLiA outer
tests have already been inspected. Therefore D3 cannot become confirmatory,
regardless of its numerical result; any selected rule must be transferred
without retuning to a different wearable cohort.

## Question

Does synchronized local wrist-motion context improve the full multilevel HCRD
pulse-event representation? This is not a test of acceleration alone. The HCRD
candidate trajectory remains the primary input, and motion coordinates are an
auxiliary quality/context channel.

## Immutable data boundary

- Use the existing 117 PPG-DaLiA activity records and the five outer subject
  folds in `data/manifests/ppg_dalia_records.json`.
- For outer fold `f`, use fold `f` as test, fold `(f+1) mod 5` as validation,
  and the remaining nine subjects for fitting.
- Reuse the PPG conditioning, ECG-to-PPG alignment, 150 ms one-to-one matcher,
  invalid-sample mask, and validation-selected P0 parameters in the existing
  outer-test locks. Do not reselect P0 on an outer test.

## Inputs

At each HCRD candidate, use the complete 140-coordinate, eight-level geometry
trajectory. Evaluate two fixed feature sets:

1. `geometry_motion`: 140 HCRD geometry plus 11 motion coordinates;
2. `hybrid_motion`: the same 140 geometry, seven local PPG morphology
   coordinates, and the same 11 motion coordinates.

The wrist-acceleration signal is the scalar `acc_ppg_site` channel sampled at
32 Hz. For centred windows of 0.5, 1, 2, 4, and 8 seconds, calculate
`log1p(local standard deviation)` and
`log1p(local mean absolute first difference)`. The eleventh coordinate is
`log1p(abs(acceleration - centred 8-second mean))`. Sample all coordinates at
the PPG candidate time by nearest acceleration index. These operations use no
ECG annotation.

Polygon/triangle mass and residual energy remain coordinates inside the full
geometry. The previously published 39-coordinate mass-only model is retained
as an ablation and is not augmented here.

## Learner and selection

For each feature set, fit one `HistGradientBoostingClassifier` with
`learning_rate=0.05`, `max_leaf_nodes=15`, `max_iter=200`,
`l2_regularization=1.0`, balanced classes, and seed 1729. These settings equal
the frozen PPG-DaLiA hybrid learner; no D3 model hyperparameter grid is allowed.

On validation subjects only, choose an event-probability threshold from
`0.05, 0.10, ..., 0.95`. Rank by median record F1, then micro-F1, then
micro-precision, then the larger threshold. Select the D3 primary feature set
by the same ranking. Apply it once to the outer test subjects.

## Endpoints and interpretation

Pool the five cross-fitted outer-test partitions and report median record F1,
micro-precision, micro-recall, micro-F1, motion-intensive macro median F1, all
activity medians, and paired 20,000-replicate subject-bootstrap differences.
Compare D3 with the locked P0 detector, frozen geometry/hybrid HCRD models, and
the mass-only ablation.

An interesting development result requires higher median and micro-F1 than P0.
A positive paired interval would strengthen the signal but would still not
repair the post-test status. Failure is retained. No additional D3 feature,
window, model, gate, or threshold family may be introduced after inspecting
its outer-test outputs.
