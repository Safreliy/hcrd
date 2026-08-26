# A2: temporal analysis and label-free fusion of HCRD area series

Status before execution: **development protocol, A1 evaluation still sealed**.

## Motivation

A1 tuning selected the maximum robust area-density surprise across eight HCRD
levels. Its exact tuning VUS-PR is 0.353779 versus 0.267917 for absolute raw
amplitude deviation, but the two scores fail on different series. A2 tests the
user-proposed next step: treat polygon mass as a time series, analyse its
temporal regularity, and combine complementary label-free evidence.

## Fixed representation and candidates

The base representation is the complete eight-row matrix
`A[l, t] = |d_l(t)|`; the A1 direct score is the largest robust level surprise.
The temporal operator is spectral-residual saliency with a fixed frequency-axis
averaging width of 100, applied either to the direct area series or separately
to all eight level rows. Levelwise saliencies are combined by their empirical
rank maximum or mean.

Three fixed rank fusions combine direct HCRD evidence with raw amplitude using
HCRD weights 0.25, 0.50, and 0.75. Further 0.50/0.50 fusions combine direct,
raw, and levelwise temporal evidence. All ranks and transforms use the observed
series only; labels and file/domain metadata are never inputs.

The twelve candidate names and formulas are generated only by
`hcrd_temporal_candidate_scores`; no hidden dataset-specific branches are
allowed.

## Selection and confirmation

- Development data: the same official 48-series TSB-AD-U tuning split.
- Selection: maximum mean exact 250-threshold VUS-PR; ties use median VUS-PR,
  then lexical candidate name.
- Freeze: write candidate name, all metrics, code hashes, tuning-list hash, and
  the unchanged A1 freeze hash before any A1/A2 evaluation execution.
- Confirmation: execute the frozen A1 and A2 detectors together on all 350
  official evaluation files. A2 is the practical primary; A1 remains the pure
  geometric ablation. No post-evaluation retuning is permitted.

## Interpretation limits

Spectral residual is a generic downstream operator, so an A2 improvement would
show that HCRD produces a useful temporal representation, not that HCRD alone
solves every anomaly type. Conversely, a fusion win would establish
complementarity with amplitude anomalies rather than universal dominance of
convexity geometry.

