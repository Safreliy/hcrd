# C1: held-out Yahoo point-anomaly confirmation

Status before HCRD execution: **frozen confirmation extension**.

## Purpose and independence boundary

The official TSB-AD experiment found a point-anomaly specialization on 49
evaluation series. C1 tests the unchanged detector on additional Yahoo series
that were absent from both the 48-series TSB-AD tuning list and the 350-series
TSB-AD evaluation list. This is a new-series confirmation within the same Yahoo
dataset family, not an independent data source.

## Frozen detector

- configuration: `hcrd_L8_max`;
- score: maximum robust positive surprise over eight exact HCRD area-density
  rows;
- no fitting, label use, source-specific threshold, or parameter change;
- primary metric: mean per-series AUC-PR, because TSB-UAD publishes exact
  per-series AUC-PR for all comparators but not per-series VUS-PR.

## Population construction

1. Start with all 259 Yahoo series in TSB-AD-U.
2. Exclude every filename in `TSB-AD-U-Tuning.csv` or `TSB-AD-U-Eva.csv`.
3. Match the remaining series to the official TSB-UAD Yahoo files by equal
   length, first anomaly, exact labels, and signal values within `1e-10` absolute
   tolerance (observed representation difference at most `5.7e-14`).
4. Exclude non-unique content matches instead of choosing by filename.
5. Use the official TSB-UAD `point_anom == 1` flag as the primary stratum.

The label-free mapping stage yielded 220 unique matched series, including 134
point-anomaly series, before any C1 HCRD score was computed.

## Comparators and evaluation

Published TSB-UAD per-series values are used without rerunning or retuning the
comparators. The primary non-neural comparator is the method with the largest
mean AUC-PR on the fixed point population before HCRD execution: NORMA
(`0.338286`). CNN (`0.905162`) is the learned performance ceiling. IForest,
LOF, Matrix Profile, IForest1, HBOS, OCSVM, PCA, autoencoder, LSTM, and POLY are
secondary descriptive comparisons.

The primary success condition is a positive paired HCRD-minus-NORMA mean
difference with a 95% file-bootstrap interval above zero. CNN is reported even
if HCRD does not beat it. The full 220-series matched population is secondary.
Wins/ties/losses use a `1e-12` tolerance. No series may be dropped after HCRD
scores are visible except for a documented execution failure.

## Fixed external artifacts

- TSB-UAD code commit: `313f0fdeba14292b9db4e1aa94c74a983a25de31`;
- TSB-UAD Public archive SHA-256:
  `ff4aa83a5a111835d410d962152e8dbebcda1039b778bae45b6b9c3f46dd49a1`;
- official AUC-PR table SHA-256:
  `c86cb2cec271a5346e116b00c012376024c5af44897ee1f119d9f8834cfe3534`.

The freeze file and content-matched manifest must be written before the
evaluation mode is permitted to compute HCRD scores.
