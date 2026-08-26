# Frozen XJTU-SY remaining-useful-life protocol X1

Status: prospectively fixed before HCRD-energy outcome inspection.
Date fixed: 2026-08-24.

## Why this task

XJTU-SY contains complete run-to-failure vibration measurements for 15
bearings under three operating conditions. Each acquisition has two channels
and 32,768 samples at 25.6 kHz; acquisitions are one minute apart. Unlike the
completed CWRU classification study, the task is not at a performance ceiling
and directly tests whether the evolution of HCRD geometric mass carries
degradation information.

Official source:
<https://biaowang.tech/xjtu-sy-bearing-datasets/>

## Primary question

With an identical temporal regressor and identical train/test folds, does
adding causal multilevel HCRD polygon-energy features reduce normalized RUL
RMSE relative to established time/frequency features alone?

## Split and target

- Fifteen leave-one-bearing-out folds.
- Each test bearing is excluded from fitting the predictor and all learned
  preprocessing.
- Training bearings come from the same operating condition as the held-out
  bearing for direct comparison with the public LOBO benchmark.
- Target is linearly normalized full-life RUL in `[1,0]`.
- Primary metric: macro-average of per-bearing normalized RMSE.
- Secondary metrics: MAE, R-squared, late-life RMSE for RUL <= 0.3, and
  per-bearing paired differences.
- Uncertainty: bearing-level bootstrap confidence interval and a paired
  permutation test over 15 bearings.

## Representations

All feature families see both vibration channels.

1. `standard`: conventional time/frequency statistics.
2. `hcrd-only`: HCRD polygon, quadratic, triangle, sign, concentration,
   duration, and amplitude summaries over six levels.
3. `standard+hcrd`: concatenation of 1 and 2; this is the primary candidate.
4. Negative controls: shuffled HCRD columns within training bearings and a
   parameter-matched random projection of the standard features.

HCRD is applied to a fixed-length block-RMS envelope and a fixed-length
log-power spectrum, not to all 32,768 raw samples. This keeps the representation
fast and assigns a clear meaning to geometry along time and frequency axes.

## Temporal predictor

Primary model: a small past-only feature-sequence LSTM with a history of ten
acquisitions. A gradient-boosted tree with causal lag/difference features is
the non-neural robustness check. Hyperparameters are selected inside each
training fold only, from a small predeclared grid, using bearing-grouped
validation.

The main ablation changes only the input representation. It is not valid to
compare HCRD with a stronger learner against a weaker standard-feature learner.

## Calibration variants

- X1a, benchmark-comparable: per-bearing calibration using the first 20% of
  each trajectory, matching the public Feature-LSTM benchmark. This is a form
  of test-time adaptation and will be labelled as such.
- X1b, online sensitivity analysis: calibration from the first 20
  acquisitions only, independent of eventual lifetime.

The published 0.160 LOBO normalized RMSE is used only as an external reference
for X1a, not as a statistically paired comparator.

The conventional 65-feature extractor is used unmodified from the MIT-licensed
`thfmn/xjtu-sy-bearing` repository at commit
`7d7231c582961741bde629da6731e6c169d88785`. Our wrapper parallelizes independent
bearings and records the upstream revision; it does not change feature formulas.

## Evaluation criteria

- Primary success: `standard+hcrd` improves macro RMSE over `standard`, with a
  bearing-level 95% confidence interval excluding zero.
- Strong external success: X1a macro RMSE is below 0.160 and the result is
  stable across five model seeds.
- A win on a subset of bearings or after unregistered tuning is exploratory.
- Failure, null results, and per-bearing heterogeneity remain in the release.

## Post-X1 development amendment

The first X1 run retained all 288 HCRD coordinates and was negative
(`standard` macro RMSE 0.229; `standard+hcrd` 0.261, seed 9). It is permanently
labelled exploratory and will not be overwritten. X2 tests a compact envelope
representation motivated by the geometric definition: polygon mass,
quadratic energy, concentration, and shape factor at six levels, aggregated by
the mean and maximum across sensor axes (48 coordinates). Spectral HCRD and
redundant amplitude/triangle coordinates are excluded. Because this amendment
was designed after viewing X1, XJTU-SY is now a development dataset; any
positive X2 result requires confirmation on PRONOSTIA or another untouched
run-to-failure dataset.

X3 is a further explicitly post-outcome diagnostic: instead of 48 summaries,
it retains only the exact log polygon mass from envelope levels 2--4 on both
sensor axes (six coordinates). Its purpose is to test the user's original area
hypothesis without allowing auxiliary HCRD statistics to dominate the learner.

## Recorded outcome

The five-seed X3 LightGBM robustness run is negative. Standard features alone
gave macro RMSE `0.1909049`; adding the six exact polygon masses gave
`0.1926870`. The candidate-minus-standard bearing-level difference was
`+0.0017821` (`+0.93%`), with a 95% bearing-cluster bootstrap interval
`[+0.0005647,+0.0031190]`. Only 4/15 bearings improved after averaging over
seeds. The one-seed apparent gain therefore did not reproduce and is not a
claim.

An explicitly exploratory health-indicator analysis found that
`h_env_level_3_log1p_polygon_area` had positive Spearman association with life
progress in all 15 bearings (median `0.743`, range `0.051--0.920`). A standard
low-frequency vertical band-power feature was more strongly monotone overall
(median `0.891`, positive on 14/15). The HCRD observation motivates, but does
not validate, a separately frozen health-indicator or change-detection task on
an untouched run-to-failure dataset.

That follow-up was frozen as PRONOSTIA H1 and failed: the exact selected mass
was positive on 9/17 independent complete trajectories and did not improve
absolute trendability over RMS. See `docs/pronostia_health_indicator_protocol.md`.

## Leakage guards

- No random window split.
- No fitting scaler, feature selector, or hyperparameter on a held-out bearing.
- Temporal windows use present and past acquisitions only.
- No future-derived onset index is an input feature.
- Dataset download hashes, code revision, package versions, and seeds are
  recorded before the confirmatory rerun.
