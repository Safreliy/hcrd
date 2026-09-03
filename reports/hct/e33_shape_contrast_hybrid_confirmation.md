# E33: honest S-shaped inflection UQ and frontier hybrid

Status: **confirmation**.  Nominal level: 95%.  Repetitions: 200 per cell.

E33 uses a fixed multiscale family of sign-valid chord contrasts with analytic Gaussian Bonferroni calibration. `Sshaped` 1.2 is the external point estimator; `ShapeChange` 1.5 with its documented residual-bootstrap interval (1000 resamples) is the UQ comparator.

| signal | design | n | E33 cover | E33 med. width | ShapeChange cover | ShapeChange med. width | Sshaped MAE | projected MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f1_cusp | beta_4_8 | 1000 | 0.965 | 0.0712 | 0.875 | 0.0186 | 0.0017 | 0.0017 |
| f1_cusp | beta_4_8 | 500 | 0.975 | 0.0877 | 0.830 | 0.0326 | 0.0022 | 0.0025 |
| f1_cusp | uniform | 1000 | 0.985 | 0.1109 | 0.000 | 0.0051 | 0.0023 | 0.0023 |
| f1_cusp | uniform | 500 | 0.985 | 0.1737 | 0.000 | 0.0069 | 0.0037 | 0.0039 |
| f2_onset | beta_4_8 | 1000 | 0.990 | 0.7863 | 0.000 | 0.0255 | 0.0821 | 0.0821 |
| f2_onset | beta_4_8 | 500 | 0.985 | 0.8149 | 0.000 | 0.0498 | 0.0972 | 0.0968 |
| f2_onset | uniform | 1000 | 0.985 | 0.6703 | 0.000 | 0.0141 | 0.0982 | 0.1009 |
| f2_onset | uniform | 500 | 0.975 | 0.7325 | 0.000 | 0.0199 | 0.1081 | 0.1124 |
| f3_jump | beta_4_8 | 1000 | 0.965 | 0.0017 | 0.000 | 0.0083 | 0.0000 | 0.0000 |
| f3_jump | beta_4_8 | 500 | 0.995 | 0.0041 | 0.000 | 0.0123 | 0.0002 | 0.0002 |
| f3_jump | uniform | 1000 | 0.985 | 0.0060 | 0.000 | 0.0054 | 0.0007 | 0.0007 |
| f3_jump | uniform | 500 | 0.970 | 0.0100 | 0.015 | 0.0079 | 0.0014 | 0.0014 |
| f4_logistic | beta_4_8 | 1000 | 0.975 | 1.0000 | 0.985 | 0.4699 | 0.1390 | 0.1372 |
| f4_logistic | beta_4_8 | 500 | 0.975 | 1.0000 | 0.985 | 0.5509 | 0.1411 | 0.1409 |
| f4_logistic | uniform | 1000 | 0.975 | 0.9990 | 0.960 | 0.4443 | 0.1645 | 0.1664 |
| f4_logistic | uniform | 500 | 0.975 | 1.0000 | 0.990 | 0.4515 | 0.1554 | 0.1546 |

## Frozen gates

- PASS — `hct_coverage_at_least_0_93_every_cell`
- PASS — `zero_unexplained_empty_sets`
- PASS — `external_fit_success_at_least_0_98_every_paper_cell`
- PASS — `predeclared_large_n_width_limits`
- PASS — `no_projection_increase_on_covered_trials`
- FAIL — `aggregate_projection_does_not_increase_mean_error`
- PASS — `publication_separation_at_least_two_nonsmooth_cells`
- PASS — `weak_logistic_regime_retained`

## Interpretation guardrails

Coverage of `ShapeChange` outside its documented smooth spline regime is a robustness comparison, not a claim that its assumptions are met. The logistic cell is retained as the smooth in-class check. Failures are counted as noncoverage. E33's theorem requires known Gaussian noise scale; this experiment does not establish an unknown-scale result or real-data utility.

The confidence-set theorem does not depend on either comparator. Projection is a hybridization step: on every covered trial it cannot increase absolute error of the external point estimate.

Raw result directory: `C:\Users\Savely\Desktop\NeuralNetsTesting\hcrd_research\results\hct\shape_contrast_hybrid_e33_confirmation`.
