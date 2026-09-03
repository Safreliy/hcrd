# Exact SCI inference from replicate curves

## Practical setting

Some experiments measure the same response curve several times. Examples
include assay runs, growth experiments, and repeated dose-response curves.
The measurements at two design points can be strongly related because they
come from the same experimental run. Treating all measurements as independent
would give intervals that are too optimistic.

The replicate version of SCI keeps each run intact. It needs independent runs,
but it allows arbitrary dependence and unequal variance within a run.

## Assumptions

Let `Y_1,...,Y_R` be independent Gaussian vectors with the same distribution

`Y_r ~ N(mu, Sigma)`,

where `Sigma` may be non-diagonal or singular. The vectors are observed on the
same fixed design. Let `a_1,...,a_M` be the fixed SCI contrast vectors and let
`theta_j = a_j^T mu`.

For each contrast, compute its value in every run,

`Z_rj = a_j^T Y_r`,

and let `mean(Z_j)` and `s_j` be the sample mean and sample standard deviation
over runs. Set

`q = t_(R-1)(1 - alpha/(2M))`

and use the bounds

`mean(Z_j) +/- q s_j/sqrt(R)`.

## Theorem R1: simultaneous finite-sample coverage

With probability at least `1-alpha`, all `M` intervals contain their contrast
means at the same time. Therefore, inverting their certified signs gives an
outer confidence set that contains the full identified transition set.

### Proof

Fix one contrast `j`. Across independent runs, `Z_1j,...,Z_Rj` are independent
normal variables with a common mean and variance. If that variance is
positive, the usual Student statistic has exactly a `t_(R-1)` distribution.
Thus the two-sided interval misses `theta_j` with probability
`alpha/M`. If the variance is zero, all scores equal `theta_j` almost surely
and the zero-width interval is exact. The union bound over `M` contrasts gives
simultaneous failure probability at most `alpha`. The deterministic SCI
inversion theorem then gives coverage of the full transition set.

## What the theorem does and does not allow

- It allows correlation between concentrations or time points within a run.
- It allows a different variance at every design point.
- It does not require estimation of the full covariance matrix.
- Runs must be independent and have the same mean and covariance.
- Gaussianity is needed for the exact small-sample Student statement.
- Technical duplicates inside one run are not independent runs. They should be
  averaged first, or modelled separately.
