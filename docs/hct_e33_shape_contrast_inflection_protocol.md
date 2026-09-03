# E33 — shape-contrast inversion for honest S-shaped inflection sets

## Status and driving question

**Status:** development protocol written before generating E33 responses.

**Driving question:** can simultaneous multiscale convexity contrasts be
inverted into an honest, useful confidence set for the inflection point of an
S-shaped regression function over the full shape class, including the
nonsmooth signals that invalidate the derivative-band HCT theorem?

The target is the explicit uncertainty-quantification direction in Feng et
al., *Nonparametric, Tuning-Free Estimation of S-Shaped Functions*, JRSSB 84
(2022), DOI `10.1111/rssb.12481`.  The proposed object is a complement to the
official `Sshaped` least-squares point estimator, not another curve fit.

## Minimal construction

For ordered design points and every predeclared support window, split `3q`
successive observations into left, middle and right blocks of size `q`.  Pair
the `r`th observations of the three blocks and form the chord residual

$$
T_{a,q,r}
=
\frac{x_R-x_M}{x_R-x_L}Y_L
+
\frac{x_M-x_L}{x_R-x_L}Y_R
-Y_M.
$$

Average these `q` residuals.  Its expectation is nonnegative for every convex
function and nonpositive for every concave function; no derivative or
continuity at the unknown inflection is used.

Simultaneous two-sided Gaussian intervals are formed for a fixed multiscale
family.  A certified positive contrast on support `[a,b]` implies that an
S-shaped inflection cannot lie at or to the left of `a`, because the entire
support would then be concave.  A certified negative contrast implies that it
cannot lie at or to the right of `b`.  Intersecting all such restrictions gives

$$
C(Y)=\left[
\max_{T>0\ \mathrm{certified}} a_T,
\min_{T<0\ \mathrm{certified}} b_T
\right],
$$

with domain endpoints used when the relevant collection is empty.

## Predicted theorem signature

**Objects:** fixed ordered design, Gaussian regression vector, finite
predeclared contrast family, S-shaped mean function, its possibly non-singleton
inflection set `I_f`.

**Assumptions:** independent homoskedastic Gaussian noise with known scale for
the primary theorem; each contrast is a positive average of valid chord
residuals; all contrast errors are covered simultaneously at level
`1-alpha`.

**Conclusion:** with probability at least `1-alpha`, every admissible
inflection lies in `C(Y)`, hence `I_f subset C(Y)`.  The set may be empty only
off the simultaneous event.  Selection over locations and scales is already
paid for by the common calibration.

**Power target:** under the paper's local order-`gamma` separation around a
unique `m0`, supports of physical width `h` have standardized signal of order
`sqrt(n h) h^gamma`.  Thus the anticipated diameter is
`O_P((log n/n)^(1/(2 gamma+1)))` for the initial global finite family, matching
the paper's LSE localization upper rate.  Proving constants and irregular
design conditions is `OPEN` until the deterministic contrast lemma and
probabilistic scan bound are complete.

## Proof pattern and DAG

Pattern: simultaneous inequalities followed by deterministic inversion.

```text
convex chord inequality ----+
                            +--> sign-valid contrast expectation
concave reverse inequality -+

Gaussian family event --> all contrast means covered

sign-valid expectation + covered observed sign
    --> exclusion of one side of candidate m
    --> intersection retains every m in I_f
    --> P(I_f subset C(Y)) >= 1-alpha
```

The proof must not use the `Sshaped` point estimator, so the latter may be
computed from the same response without a post-selection leak.

## Development design

- all four Equation (15) paper signals `f1`--`f4`, including the cusp,
  one-sided onset and jump;
- increasing affine, convex quadratic and concave quadratic controls;
- `n in {100,200,500,1000}`;
- uniform and Beta(4,8)-quantile designs;
- iid Gaussian noise `sigma=0.1`;
- dyadic block sizes `q=1,2,4,...` with `3q <= n`;
- support starts spaced by `q`, so the family has linear rather than quadratic
  cardinality per scale;
- familywise level `alpha=0.05` using an analytic Gaussian union bound in the
  first implementation;
- 80 development repetitions per cell, seed `20261611`.

The official `Sshaped` point estimator is retained for point-error comparison.
E32 derivative HCT is reported only on its smooth applicability subset and is
not allowed to define or tune E33.

Development may replace the analytic union bound by a jointly calibrated
Gaussian maximum, refine the fixed scale grid, add a valid one-sided upper
noise-scale pivot, and select confirmation sample sizes.  It may not remove
any paper signal or design after outcomes are seen.

## Metrics

- simultaneous confidence-set coverage for the unique paper inflection;
- interval width, median width and nontriviality relative to `[0,1]`;
- empty-set probability;
- endpoint localization error around `m0`;
- official `Sshaped` point MAE and whether projection into `C(Y)` increases
  error on covered cases;
- for affine/convex/concave controls, containment of the correct full or
  boundary inflection set;
- computation time and family cardinality.

## Hypothesis ablation and risks

- Removing simultaneous calibration invalidates adaptive location/scale
  selection.
- Arbitrary linear three-block contrasts are not sign-valid; the chord weights
  and nonnegative averaging are essential.
- Without local separation, honest intervals must remain wide; affine signals
  are a required counterexample.
- A plug-in noise estimate is not valid by default.  Any unknown-scale version
  must use a separately proved one-sided pivot plus an explicit error-budget
  union.
- The initial rate argument may lose logarithmic or design-density factors.
- Multiscale shape inference and SiZer have direct priority.  Novelty requires
  explicit inflection-set inversion, finite-sample coverage over the full
  S-shaped class, a rate theorem, strong comparator experiments and a careful
  priority audit; the contrast family alone is not novel.

## Confirmation gate

No confirmation is allowed until the implementation tests the deterministic
shape inequalities and development identifies one fixed family.  Before
confirmation append untouched seeds, repetitions, executable hashes and
numeric gates.  Minimum qualitative gates are coverage at least 0.93 in every
paper cell, zero unexplained empty sets, explicit retention of wide weak-signal
  regimes, and no covered-case projection error increase.

## Development amendment A — separated blocks (before new responses)

The first analytic-Bonferroni development run retained coverage in every paper
cell (minimum point estimate `0.9375`) and exposed a clear power split.  At
`n=1000`, median widths were `0.0882/0.1588` for cusp `f1` under Beta/uniform
and `0.00170/0.004995` for jump `f3`, but remained at least `0.870` for onset
`f2` and essentially `1` for logistic `f4`.

The limitation comes from coupling the number `q` of averaged residuals to the
distance `q` between their left/middle/right observations.  The sign lemma
does not require adjacent blocks.  Before generating a second development
response, extend the fixed family to separation multipliers `s/q in {1,2,4}`:
each contrast averages triples `(a+r, a+s+r, a+2s+r)`, `r=0,...,q-1`, whenever
the support fits.  All coefficients remain nonnegative chord-interpolation
weights, so convex/concave sign validity is unchanged.  Starts remain spaced
by `q`; the family remains finite and fully Bonferroni-calibrated.

Use fresh development seed `20261621`, retain all signals, designs, sample
sizes, noise and metrics, and label the run `development_separated`.  This is
a declared family refinement permitted by the original protocol, not a
confirmation.  No further response-dependent family change may be folded into
the eventual confirmation without another written amendment and fresh seed.

## Development amendment B — joint Gaussian maximum (before new responses)

Separated blocks reduced median widths for `f1` and `f2` in the main
large-sample cells (for example, `f2`, uniform, `n=1000`: `0.8701` to
`0.6384`) while retaining paper-cell coverage at or above `0.95` except one
`0.9625` cell.  Their larger family raises the analytic Bonferroni critical
value to `4.4518` at `n=1000` and can sacrifice power on the already easy jump
signal.

Before a third development response, replace only this calibration step by an
upper order statistic of the jointly simulated standardized Gaussian contrast
maximum.  Allocate total error `0.05` as data-maximum failure `0.045` plus
calibration failure `0.005`; use 4,000 calibration simulations with seed
`20261631`.  Keep the separated family `{1,2,4}`, all designs, signals,
sample sizes and 80 response repetitions, with fresh response seed `20261632`.
Label the run `development_joint`.

The calibration must be independent of the responses and computed separately
for each fixed `(design,n)` operator.  Its finite-simulation order-statistic
rank and family fingerprint must be stored.  This amendment does not authorize
signal-specific critical values or removal of an inconvenient scale.

## Development conclusion and amendment C — frozen comparator confirmation

The jointly simulated maximum did not improve the family.  At `n=1000` its
upper critical values (`4.4877` uniform and `4.5138` Beta) exceeded the
analytic Bonferroni value `4.4518`, because the family contains many nearly
independent fine-scale contrasts.  The final E33 method is therefore the
simpler separated-block family with multipliers `{1,2,4}` and the exact
analytic Gaussian Bonferroni calibration.  This choice was made from the three
declared development runs before the confirmation responses below existed.

The priority audit also found that `ShapeChange` 1.5 exposes a nominal 95%
bootstrap interval for a spline-based inflection estimate.  E33 therefore may
not claim that inflection intervals did not previously exist.  The frozen
comparison has two distinct external roles:

- `Sshaped` 1.2 is the official tuning-free point estimator for the full
  S-shaped class of Feng et al.;
- `ShapeChange` 1.5 with `changept(y ~ ip(x, sh=1), fir=TRUE, ci=TRUE,
  nloop=1000)` is the interval comparator.

The package source shows that the iid-Gaussian interval resamples residuals
around the fitted smooth cubic spline and takes bootstrap 2.5%/97.5%
quantiles.  Its documentation assumes a smooth convex--concave curve.  The
comparison on nonsmooth Feng signals is consequently a robustness comparison,
not an assertion that the spline assumptions hold.  The smooth logistic signal
is retained as an in-class check, every fitting error counts as noncoverage,
and neither external result changes the E33 set.

### Frozen confirmation configuration

Frozen on 2026-09-02 before seed `20261711` generated a confirmation response:

- sample sizes `{500,1000}`;
- uniform and Beta(4,8)-quantile fixed designs;
- all four Equation (15) Feng signals and all three affine/boundary controls;
- 200 independent responses per cell, Gaussian `sigma=0.1`;
- response seed `20261711`;
- separately generated per-trial R bootstrap seeds from seed `20261712`;
- nominal `alpha=0.05`;
- dyadic block sizes, start stride one block, separation multipliers `{1,2,4}`;
- analytic two-sided Gaussian Bonferroni calibration with known `sigma`;
- 1,000 `ShapeChange` residual-bootstrap refits per paper-signal response;
- four computational shards.  Sharding changes neither data nor bootstrap
  seeds and exists only to reduce wall-clock time.

The executable writes `frozen_config.json`, including SHA-256 hashes of the
protocol, Python driver, R bridge, E33 module and installed external package
descriptions, before `--stage prepare` will generate any confirmation response.
Later stages refuse to run if those hashes or prepared-response hashes change.

### Frozen numeric gates

Core validity/usefulness passes only if:

1. E33 empirical target-set coverage is at least `0.93` in every one of the 28
   paper and control cells;
2. there is no empty E33 set on a trial where every true contrast mean lies in
   its simultaneous band;
3. both external methods succeed on at least `0.98` of trials in every paper
   cell, with failures retained as interval noncoverage;
4. at `n=1000`, E33 median width is below `0.15` for `f1`, below `0.85` for
   `f2`, and below `0.02` for `f3`, under both designs;
5. projection of the `Sshaped` point into the E33 set never increases absolute
   error on an E33-covered trial and does not increase the aggregate mean
   error over paper cells;
6. all four weak-logistic cells remain in the report, with no width/power gate.

A separate publication-separation gate asks for an E33 minus `ShapeChange`
coverage difference of at least `0.10` in at least two of the 12 nonsmooth
`f1`--`f3` cells.  This is not needed for validity of E33; it tests whether the
experiment demonstrates a practically material robustness gap rather than
only a theorem with conservative intervals.

No threshold is conditional on the sign of a development comparison, and no
signal, design or failed comparator trial may be dropped after confirmation.
