# E2 matched-capacity sensitivity analysis

This supplementary sensitivity analysis was specified on 2026-08-26 after the
primary E2 result and before any matched-control score was computed. It is not
part of the prospective E2 success rule and does not alter the primary E2
inference.

## Question

Does HCRD-8 retain cross-study LC--MS transfer value against a non-HCRD
waveform bank with exactly the same number of per-file and aggregated
variables?

## Fixed control

For each available global-window EIC, start from the same 75 raw variables used
in E2: 64 normalized waveform samples and 11 scalar summaries. Append Gaussian
smooth, first-derivative, and second-derivative waveforms at scales 1, 2, 4,
and 8. For each of these 12 channels append mean, standard deviation, minimum,
maximum, mean absolute value, root-mean-square value, mean absolute first
difference, and maximum absolute value. Append four residual-energy ratios,
four derivative maxima, and normalized spectral entropy. The resulting 948
per-file variables exactly match HCRD-8.

Aggregate by median, 90th percentile, and maximum across files, then append the
available-file fraction and the two independently recomputed global-window
qscore variables. Both representations therefore contain 2,847 variables.

## Evaluation

Use the same unambiguous Falkor and MESOSCOPE populations, both no-refit
directions, standardized L2-regularized logistic learner, and 10,000 paired
target-feature bootstrap replicates as E2. The audit contrast is HCRD-8+Q minus
the matched Gaussian-derivative control. Holm correction is applied across the
two transfer directions.

The interval estimates are conditional on the fitted source model and treat
target mass features as exchangeable units. They do not include source-refit,
compound/adduct-cluster, or file-level uncertainty.
