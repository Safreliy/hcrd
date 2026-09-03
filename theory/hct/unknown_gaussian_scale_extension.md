# Honest unknown-scale extension for shape-contrast inversion

## Construction

Let `Y=f+epsilon` with `epsilon ~ N(0,sigma^2 I_n)` and unknown `sigma`.
Choose, before observing `Y`, any linear nuisance space with design matrix
`X` of rank `r<n`, and let `R=I-P_X` be its orthogonal residual projector.
For a scale-failure budget `eta`, define

$$
\bar\sigma=
\left\{
\frac{\|RY\|_2^2}{\chi^2_{n-r,\eta}}
\right\}^{1/2},
$$

where `chi^2_{nu,eta}` is the lower `eta` quantile of a central chi-square
variable with `nu` degrees of freedom.

## Theorem

**THM E34.1.**  Uniformly over every deterministic mean vector `f` and every
`sigma>0`,

$$
\Pr_{f,\sigma}\{\bar\sigma\ge\sigma\}\ge1-\eta.
$$

If E33's Bonferroni contrast bands are computed with `bar_sigma` and a
remaining error budget `alpha-eta`, their inversion obeys

$$
\Pr_{f,\sigma}\{I_f\subseteq C(Y)\}\ge1-\alpha.
$$

No independence between the scale estimate and the chord contrasts is
required.

## Proof

The residual quadratic form has distribution

$$
\frac{\|RY\|_2^2}{\sigma^2}
\sim\chi^2_{n-r}(\lambda),
\qquad
\lambda=\frac{\|Rf\|_2^2}{\sigma^2}.
$$

A noncentral chi-square variable is stochastically increasing in its
noncentrality parameter.  Its probability of falling below the central lower
`eta` quantile is therefore at most `eta`, proving the scale statement.

Let `E_sigma={bar_sigma>=sigma}`.  On `E_sigma`, every band radius computed
with `bar_sigma` is at least the corresponding known-`sigma` radius.  The
known-scale Gaussian union bound at level `alpha-eta` and a union bound with
`E_sigma^c` give total failure at most `(alpha-eta)+eta=alpha`.  Deterministic
E33 inversion then retains the full admissible-inflection set.

## Fixed block nuisance space

For computation, partition the ordered observations into `K` consecutive
equal-index blocks and use their indicator vectors as columns of `X`.  The
residual sum of squares is simply within-block variation.  A fixed
`K=floor(sqrt(n))` is a candidate compromise:

- `n-K` residual degrees of freedom keep the lower chi-square quantile stable;
- smooth signals vary little inside a block;
- a jump contaminates at most one block when it is not on a boundary;
- arbitrary lack of fit can only inflate `bar_sigma`, so validity is unchanged.

This choice is not yet frozen.  Its efficiency, especially under the jump and
nonuniform design, must be selected on development data and confirmed on fresh
seeds.  The theorem holds for every fixed `K<n`; only informativeness depends
on the choice.

## Pair-block adaptivity for monotone S-shaped means

The development-selected special case uses consecutive pairs (`K=n/2` for
even `n`).  Its residual statistic has the transparent form

$$
\|RY\|_2^2
=\frac12\sum_{j=1}^{n/2}(Y_{2j}-Y_{2j-1})^2,
\qquad \nu=n/2.
$$

**COR E34.2.**  Suppose `sigma>0` is fixed and the deterministic mean vectors
are sampled from monotone functions whose total range is bounded by a constant
`V`.  For the pair-block construction and any fixed `eta in (0,1)`,

$$
\frac{\bar\sigma}{\sigma}\ \xrightarrow{P}\ 1.
$$

Hence replacing known `sigma` by `bar_sigma` does not change the exact-cubic
localization order in THM E33.2 (the allocation from `alpha` to
`alpha-eta` changes only a constant critical value).

To prove the corollary, its noncentrality satisfies

$$
\lambda
=\frac{1}{2\sigma^2}
\sum_{j=1}^{n/2}{f(x_{2j})-f(x_{2j-1})}^2
\le \frac{V^2}{2\sigma^2},
$$

because monotonicity makes all increments have one sign.  Thus
`chi^2_nu(lambda)/nu -> 1` in probability by its mean and variance, while the
fixed lower central quantile obeys `chi^2_(nu,eta)/nu -> 1`.  Their ratio and
square root converge to one.  The same conclusion holds for bounded total
variation, with `V` interpreted as the variation bound.

## Proof DAG

```text
fixed nuisance space
  -> orthogonal residual quadratic form
  -> noncentral chi-square law
  -> lower-tail domination by central chi-square
  -> P(sigma <= upper scale) >= 1-eta

upper scale event + Gaussian contrast union at alpha-eta
  -> all true contrast means covered
  -> E33 deterministic inversion
  -> P(I_f subset C(Y)) >= 1-alpha
```

## Limitations

- Gaussian homoskedastic independence remains essential for the exact
  chi-square pivot.
- A response-selected nuisance space would invalidate the proof unless its
  selection were separately calibrated.
- Severe projection bias makes the interval conservative, never anti-honest.
- Heteroskedastic, dependent and heavy-tailed extensions remain open.
