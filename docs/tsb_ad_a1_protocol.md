# A1: HCRD area-spectrum anomaly detection protocol

Status before execution: **development protocol, evaluation split sealed**.

## Question

Can the temporal distribution of exact HCRD polygon mass detect anomalous
subsequences in a broad, accepted benchmark better than treating polygon area
as one static feature per record?

## Benchmark and split discipline

- Benchmark: univariate TSB-AD (NeurIPS 2024 Datasets and Benchmarks track).
- Primary metric: mean per-series VUS-PR, using the benchmark's own metric code
  and its rank-1 autocorrelation window.
- Development: the official 48-series `TSB-AD-U-Tuning.csv` split only.
- Confirmation: the official 350-series `TSB-AD-U-Eva.csv` split remains
  uninspected by the A1 detector until one configuration is written to
  `results/tsb_ad_a1/frozen_configuration.json`.
- No dataset- or domain-specific parameters are allowed.

## Representation

For every HCRD level, define the nonnegative area-density series

`a_l(t) = |d_l(t)|`,

where `d_l` is the piecewise-linear HCRD detail. Its integral is exactly the
sum of polygon areas at that level. Thus `a(t) = (a_1(t), ..., a_L(t))` retains
both the hierarchy and the time at which geometric mass occurs.

## Frozen candidate family for tuning

- hierarchy depths: 4, 8, and complete;
- `total`: sum of unnormalised level densities;
- `sum`, `max`, `l2`: three fixed aggregations after per-level, label-free
  median/90th-percentile scaling;
- `transport`: `sum` multiplied by one plus the total-variation flux of the
  normalised scale-mass distribution between adjacent samples.

This gives 15 HCRD candidates. The raw absolute median-deviation score is a
sanity baseline, not eligible to be selected as HCRD.

## Freezing rule

Computational amendment recorded before any aggregate tuning result or A1
evaluation score was available: the reference VUS implementation recreates a
length-`n` threshold vector for every one of 250 thresholds and every tolerance
width. Evaluating all 15 variants this way is unnecessarily prohibitive on the
650,000-sample medical records. We therefore screen all 15 candidates by mean
per-series AUC-PR, retain five, and write their identities to disk before exact
VUS execution. The raw sanity baseline is added as a sixth reference. The final
HCRD choice is still the candidate with the largest arithmetic mean **exact
official VUS-PR** over all 48 tuning series. Break ties by, in order: higher
median VUS-PR, fewer levels, and the simpler aggregation order `total`, `sum`,
`max`, `l2`, `transport`. Record the configuration, data-list SHA-256,
implementation SHA-256, screening results, and exact tuning results before
reading any A1 evaluation score.

The exact phase uses a score-order cumulative-sum implementation of VUS. It was
verified against the benchmark reference implementation on tied scores and
real TSB-AD files to floating-point precision; it changes computational cost,
not the 250 thresholds, buffer surface, or metric value.

## Evaluation criteria

The primary descriptive comparisons are the paired VUS-PR differences against
published Sub-PCA (the strongest original non-pretrained classical baseline),
MMPAD and StreamVAE (newer non-pretrained submissions), on identical evaluation
files whenever their public score tables permit pairing. The current overall
leader, pretrained Time-RCD+MAFT, is also reported as the performance ceiling;
we do not imply that a label-free geometric score and a pretrained model use
the same resources. Every paired comparison uses a file-level bootstrap
confidence interval and wins/ties/losses. POLY, KShapeAD, Series2Graph, and
MatrixProfile are secondary baselines. Results will also be stratified by
TSB-AD point versus sequence anomalies and by application domain, without
retuning, and runtime/memory will be reported to test for a performance-cost
Pareto advantage.

Failure is informative: if the frozen detector does not beat Sub-PCA overall,
we will identify the predeclared anomaly strata in which HCRD helps, but will
not redefine the primary benchmark after seeing confirmation results.
