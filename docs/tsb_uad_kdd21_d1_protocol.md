# D1: cross-source KDD21/UCR point-anomaly confirmation

Status before HCRD execution: **frozen cross-source extension**.

## Purpose

D1 tests the unchanged `hcrd_L8_max` detector outside the Yahoo source family.
It uses the official KDD21/UCR series in TSB-UAD that can be uniquely matched to
TSB-AD-U but are absent from both the 48-series tuning list and the 350-series
evaluation list. KDD21/UCR is a separate source family, although its anomalies
are benchmark constructions rather than a newly collected deployment cohort.

## Frozen detector and population rule

- detector: eight-level maximum robust positive HCRD area surprise;
- no fitting, label use, threshold, or parameter change;
- match every remaining TSB-AD-U `UCR` series to official TSB-UAD `KDD21`
  files by equal length, first anomaly, exact labels, and signal values within
  `1e-10` absolute tolerance;
- exclude every nonunique content match rather than select by filename;
- primary stratum: every unique match with official `point_anom == 1`;
- abort before scoring if fewer than ten primary series are available;
- primary metric: mean per-series AUC-PR.

## Comparators and success rule

Before HCRD execution, choose as primary comparator the method with the highest
mean published TSB-UAD AUC-PR on the fixed primary population among IForest,
LOF, Matrix Profile, NORMA, IForest1, HBOS, OCSVM, PCA, and POLY. AE, CNN, and
LSTM are learned contextual comparators. The primary success condition is a
positive paired HCRD-minus-primary-comparator mean difference whose 95% file-
bootstrap interval is entirely above zero. Report every method regardless of
outcome; do not discard series after scores are visible.

## Independence boundary

D1 is independent of Yahoo at the source-family level and is unseen by HCRD
tuning/evaluation, but it shares the TSB-UAD/TSB-AD benchmark infrastructure.
It is not evidence from live operations, and a small point-anomaly population
will limit precision. The official published comparator scores are used without
rerunning or retuning their implementations.

The freeze file must hash the external archive, official result table,
exclusion lists, matched manifest, protocol, detector, and runner before the
evaluation mode is permitted to compute HCRD scores.

