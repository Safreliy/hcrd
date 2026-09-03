# E34 — honest unknown-noise extension

## Status and theorem

**Status:** development protocol written before seed `20261811` generated a
response.

E33 is exact for a known Gaussian noise scale.  E34 removes that impractical
assumption without estimating derivatives or bounding the unknown mean.  For a
fixed nuisance space of rank `r<n`, residual RSS divided by `sigma^2` is a
noncentral chi-square variable.  Dividing RSS by the lower `eta` quantile of
the central `chi^2_(n-r)` law gives a one-sided upper scale bound uniformly
over every mean vector.  E33 then spends `alpha-eta` on its simultaneous
contrasts and `eta` on scale underestimation.

The proof is in `theory/hct/unknown_gaussian_scale_extension.md`.  Unknown
scale affects efficiency but not the finite-sample coverage theorem.

## Development design

- all four Feng signals and affine/convex/concave controls;
- `n in {100,200,500,1000}`;
- uniform and Beta(4,8)-quantile fixed designs;
- 80 repetitions per cell, iid Gaussian `sigma=0.1`;
- response seed `20261811`;
- total `alpha=0.05`, scale budget `eta=0.01`, contrast budget `0.04`;
- the frozen E33 separated family `{1,2,4}` and analytic Bonferroni bands;
- fixed equal-index nuisance blocks with target block lengths
  `{2,4,8,16}`; actual block count is `ceil(n/length)`;
- known-`sigma` E33 is recorded only as an efficiency reference.

Development may choose exactly one block length for a new-seed confirmation.
It may not remove a signal/design, alter the error split after seeing coverage,
or reinterpret a conservative interval as a failure.  Primary selection is
the smallest mean/median width subject to development coverage at least 0.925
in every cell and zero unexplained empty sets.  Ties favor the shorter block
length because it limits mean contamination at jumps.

## Metrics and risks

- target-set coverage and Wilson interval;
- median/mean interval width and nontrivial probability;
- upper-scale/true-scale ratio;
- empirical scale-bound coverage;
- unexplained empty-set count;
- width inflation relative to known-scale E33.

The chi-square result is uniform over arbitrary deterministic means, but a
poor nuisance approximation can inflate the upper scale and make the final set
uninformative.  The method still assumes independent homoskedastic Gaussian
errors.  A fixed block space is intentionally used: response-selected knots or
blocks would require a separate selection correction.

## Development result and confirmation freeze

All four candidate block lengths passed the development rule.  Their mean
width inflations relative to known-scale E33 were `1.220`, `1.305`, `1.413`
and `1.426` for target lengths 2, 4, 8 and 16 respectively; each had minimum
cell coverage `0.9625`.  The prospectively defined selection rule therefore
chooses target block length 2.  At `n=1000`, its mean scale ratio was about
`1.08`, and its median widths remained informative for cusp, onset and jump.

The separate confirmation is frozen before seed `20261911` is used:

- `n in {500,1000}`, both original designs, all seven original signals;
- 200 fresh responses per cell, iid Gaussian `sigma=0.1`;
- response seed `20261911`;
- target nuisance block length 2, implemented as `ceil(n/2)` consecutive
  equal-index blocks;
- scale budget `0.01`, contrast budget `0.04`;
- known-scale E33 at level `0.05` remains a paired efficiency reference;
- the E33 family and all other settings remain fixed.

Confirmation passes if unknown-scale coverage is at least `0.93` and scale
upper-bound coverage at least `0.95` in every cell, there are no unexplained
empty sets, all weak-logistic cells remain reported, and at `n=1000` the
unknown-scale median widths are below `0.20` for `f1`, `0.90` for `f2`, and
`0.025` for `f3` under both designs.  On the six informative large-sample
cells, mean width inflation over known-scale E33 must be below `1.5`.

The confirmation executable must write and validate a code-and-protocol
SHA-256 freeze before generating responses.  No post-confirmation change can
be presented as part of E34 confirmation.
