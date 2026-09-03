# E37 protocol: exact replicate-curve SCI on the DNase assay

## Question

Can SCI report uncertainty about the transition of a real S-shaped assay curve
without pretending that repeated measurements are independent?

## Data fixed before analysis

The public `DNase` dataset in the R `datasets` package contains 176 optical
density measurements from 11 assay runs. Every run has two technical
measurements at each of eight known protein concentrations. The data were
obtained while an ELISA assay for recombinant DNase in rat serum was being
developed.

The source CSV and its SHA-256 checksum are fixed by
`experiments/sci/download_dnase.py`. The official R documentation is
<https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/DNase.html>.

## Analysis fixed before evaluation

1. Average the two technical measurements within each run and concentration.
2. Keep the 11 assay runs as the independent sampling units.
3. Use log2 concentration because the experimental concentrations form a
   multiplicative grid.
4. Build SCI contrasts with all fitting dyadic block sizes and separation
   multipliers 1 and 2.
5. Use the exact replicate-curve Student band at confidence level 95%.
6. Invert certified signs over the observed concentration range.
7. Fit one four-parameter logistic curve to the across-run mean only as a
   descriptive point estimate. It is not treated as a confidence procedure.

No contrast, scale, or endpoint is selected after seeing the result.

## Assumption and estimand

The exact statement treats the 11 run-level curves as independent draws from
one multivariate Gaussian distribution. Dependence and unequal variances
between concentrations inside a run are allowed. The target is the full set
of convex-to-concave transition locations of the mean optical-density curve
on the log2-concentration scale.

## Interpretation rules

- A one-sided or wide SCI set is a valid result. It means the experiment does
  not locate both sides of the transition precisely.
- The logistic point estimate must not be reported as if it had the same
  finite-sample guarantee.
- This example tests the replicate-data extension. It does not establish that
  Gaussian run-level curves are scientifically exact for all ELISA studies.
