# Publication blueprint: honest uncertainty for S-shaped transitions

**Decision date:** 3 September 2026  
**Status:** theorem and frozen-confirmation package exists; manuscript not yet
submission-ready.

## 1. The publication problem

The broadly important data-science problem is not recovery of an unfamiliar
tree representation.  It is this:

> A flexible S-shaped regression method can estimate where a process changes
> from accelerating to diminishing returns, but a point estimate alone does
> not say which transition locations remain statistically compatible with the
> data.  Smooth spline bootstrap intervals can be extremely narrow and badly
> undercover when the true S-shape has a cusp, a one-sided onset or a jump —
> precisely the cases allowed by the leading nonparametric S-shaped model.

This is relevant to dose response, disease progression, growth and production
curves, and physical transition localization.  In the LIDAR example the
inflection estimates the centre of an atmospheric mercury plume.

## 2. Method to put in the paper

Use the public name **shape-contrast inversion (SCI)**.  The method:

1. fixes a multiscale collection of three-block chord contrasts;
2. constructs a simultaneous Gaussian band for all contrast expectations;
3. certifies intervals that must lie on the convex or concave side;
4. inverts those certificates to return every still-admissible inflection
   location;
5. reports the official tuning-free S-shaped LSE as the point estimate and SCI
   as its uncertainty layer.

The central output is `(point estimate, honest set)`, not a modified hard
point estimate.  Projecting the point estimate into the set is a valid
geometric lemma on the coverage event, but the frozen aggregate MAE gate
failed by `0.000402`; projection therefore must not be marketed as a generic
accuracy booster.

This is a principled departure from HCRD.  Chord geometry is inherited from
that research line, but the hierarchy and exact reconstruction are irrelevant
to the new scientific claim.  The paper should mention HCRD only as software
provenance, if at all.

## 3. The theorem package now available

### T1 — finite-sample honest inflection set

For a fixed ordered design and independent Gaussian errors with known noise
scale, SCI contains the **entire admissible inflection set** with probability
at least `1-alpha`.  It requires neither derivatives, continuity at the
inflection nor uniqueness of the inflection.  Thus cusp, onset, affine and
jump cases belong to the theorem, not merely to a robustness appendix.

### T2 — localization rate

For the exact cubic local model, the set diameter is bounded with explicit
constants at scale

\[
  \{\sigma^2\log n/(B^2 n)\}^{1/7},
\]

the same logarithmic localization order as the S-shaped LSE upper rate for a
cubic inflection.  The method selects among dyadic scales without knowing the
correct local resolution.

### T3 — unknown homoskedastic noise

A fixed block-projection residual norm gives an exact one-sided Gaussian scale
bound uniformly over every mean vector.  Spending `eta` on that event and
`alpha-eta` on SCI preserves finite-sample coverage without data splitting or
replicates.  Under bounded variation, the scale inflation vanishes and the
cubic rate is unchanged.

### T4 — unknown bounded heteroskedastic noise

If

\[
 \max_i\sigma_i^2\leq\kappa n^{-1}\sum_i\sigma_i^2,
\]

a new block-residual concentration argument gives a simultaneous upper
envelope for all pointwise standard deviations, uniformly over the unrestricted
mean.  Combining it with SCI again gives finite-sample `1-alpha` coverage.
The parameter `kappa` is reported as a sensitivity assumption; it is not
estimated from one unreplicated response per design point.

### T5 — independent replicate curves

When several independent runs are observed on the same design, SCI can work
with the contrast values across runs. A Student interval is exact for each
contrast under multivariate Gaussian replicate curves. Bonferroni correction
then gives a finite-sample simultaneous band. This extension allows arbitrary
dependence and unequal variance between design points inside a run. It does
not require estimation of the full covariance matrix.

## 4. Frozen evidence

### E33: frontier-aligned known-scale comparison

Configuration: the four published Feng et al. signals, uniform and
`Beta(4,8)` designs, `n in {500,1000}`, `200` repetitions per cell, official
`Sshaped` point fits and `ShapeChange` intervals with `1000` residual-bootstrap
refits per response.

- SCI coverage over all 28 paper/control cells: minimum `0.935`.
- SCI coverage over the 16 paper cells: `0.965` to `0.995`.
- `ShapeChange` coverage was exactly zero in 9 of 16 paper cells.
- In 11 nonsmooth cells SCI exceeded `ShapeChange` coverage by at least `0.10`;
  the predeclared publication-separation gate passed.
- At `n=1000`, median SCI widths were `0.071/0.111` for the cusp,
  `0.786/0.670` for the onset, and `0.0017/0.0060` for the jump under
  beta/uniform designs.
- In the weak smooth logistic cells SCI coverage remained `0.975`, but its
  median width was essentially the entire domain.  `ShapeChange` covered
  `0.960`--`0.990` with widths around `0.44`--`0.55`.
- The frozen projection-MAE gate failed slightly: aggregate S-shaped LSE MAE
  `0.062357`, projected MAE `0.062759`.  No increase occurred on any trial
  where SCI covered the truth, exactly as the lemma predicts.

Interpretation: SCI solves a coverage problem in nonsmooth regimes; it does
not dominate a smooth bootstrap in interval width and it can honestly return
almost no localization information for weak curvature.

### E34: unknown homoskedastic scale

- all frozen gates passed;
- minimum SCI coverage across 28 cells: `0.990`;
- minimum scale-bound coverage: `0.975`;
- mean informative large-sample width inflation relative to known scale:
  `1.117`.

### E35: bounded heteroskedasticity

- all frozen gates passed across 48 cells;
- minimum SCI coverage: `0.995`;
- minimum variance-envelope coverage: `0.995`;
- the method remained valid on constant, linear and concentrated variance
  profiles, but became conservative as `kappa` increased.

These simulations verify the theorem mechanics and expose power boundaries;
they do not prove the theorems.

### E36: high-precision known-scale audit

E36 used 5,000 fresh responses in each of the 16 paper cells, or 80,000
responses in total. SCI coverage was `0.9770` to `0.9804`, and simultaneous
contrast coverage was `0.9524` to `0.9618`. Every coverage alarm passed.

The frozen `zero_empty_sets` gate failed. Empty sets occurred in `0.0002` to
`0.0228` of trials. This does not contradict the theorem, which allows an empty
set on its failure event with probability at most `alpha=0.05`. The failed gate
is retained in the manifest and explained in
`docs/sci_e36_posthoc_interpretation.md`.

### E38r1: matched honest baseline

The published `ShapeChange` bootstrap uses a smoother model than SCI. E38r1
therefore adds pointwise-band projection (PBP), our exact finite-sample
baseline for the same sampled convex-to-concave class. PBP projects a
simultaneous Gaussian confidence box for the whole mean vector through linear
shape-feasibility problems. It is related to Davies et al., but it is not
their official algorithm.

Both methods retained coverage in all 16 frozen cells. SCI reduced median
width by `57.9%` to `75.7%` for the cusp, `19.4%` to `32.8%` for the onset,
and `40.0%` to `50.0%` for the jump. Neither method improved on the other in
the weak logistic cells; both were essentially the full observed range. All
pre-specified E38r1 gates passed.

## 5. Real-data illustrations

### LIDAR

The current `Sshaped` package estimates the plume centre at `588 m`.  A smooth
`ShapeChange` fit estimates `606.0 m` with a nominal residual-bootstrap
interval `[592.6, 612.8] m`.  SCI gives:

| declared kappa | SCI interval (m) | width (m) |
|---:|---:|---:|
| 1.0 | [510, 664] | 154 |
| 1.5 | [486, 720] | 234 |
| 2.0 | [438, 720] | 282 |
| 3.0 | [390, 720] | 330 |
| 4.0 | [390, 720] | 330 |

This result is scientifically useful even though it is not visually
spectacular: the published data locate the plume centre only under fairly
strong variance assumptions.  Under substantial heteroskedasticity the honest
answer is that the data do not localize it.  The narrow spline interval is only
descriptive here because its iid residual bootstrap does not represent that
heteroskedasticity.

### Replicated DNase assay

E37 uses all 176 measurements from the public R `DNase` dataset. The two
technical readings are averaged within each run and concentration, leaving 11
independent run-level curves on eight concentrations. The exact replicate
Student band allows dependence and unequal variance across concentrations
inside a run.

The 95% SCI set is `[0.78125, 12.5]` in concentration units. A descriptive
four-parameter logistic fit places the transition at `4.14`. The SCI result is
one-sided in practice: it rules out a transition below `0.78125`, but it does
not find a reliable upper limit before the largest observed concentration.
This is a second real example with observable replication, not an assumed
variance-ratio sensitivity analysis.

## 6. Defensible contribution statement

Davies, Kovac and Meise (2009) already gave an honest non-asymptotic
confidence region for nonparametric regression and studied inflection
locations. Schmidt-Hieber, Munk and Dümbgen (2013) already turned multiscale
sign statements into confidence regions for roots of differential operators.
Both works also contain the regular cubic `(log(n)/n)^(1/7)` localization
order. SCI must therefore avoid a broad claim of being first.

A safe statement is:

> SCI gives a direct finite-sample outer confidence set for all
> convex-to-concave transition locations in fixed-design Gaussian regression.
> The transition may be nonunique, nonsmooth or discontinuous. The method also
> combines with one-sided scale bounds for unknown homoskedastic noise and with
> a sensitivity model for bounded heteroskedasticity. Independent replicate
> curves permit an exact Student version with arbitrary within-curve
> covariance.

Forbidden claims:

- “first confidence interval for an inflection point” — `ShapeChange` already
  supplies a residual-bootstrap interval;
- “first multiscale convexity test” — Dümbgen--Spokoiny is direct prior art;
- “better point estimator than Sshaped” — SCI is an inference layer;
- “distribution-free” — the current calibration is Gaussian;
- “valid for arbitrary heteroskedasticity” — validity requires a correct
  declared `kappa` or additional replicates/variance structure.
- “first finite-sample inflection confidence method” — Davies et al. are prior
  work;
- “new `(log(n)/n)^(1/7)` rate” — that order is already known.

## 7. Manuscript spine

1. Practical failure: point localization without trustworthy uncertainty.
2. S-shaped model and the possibly set-valued inflection estimand.
3. SCI algorithm and deterministic inversion lemma.
4. Finite-sample known-scale theorem and cubic localization rate.
5. Unknown homoskedastic and bounded-heteroskedastic extensions.
6. Frozen Feng-benchmark comparison with `Sshaped` and `ShapeChange`.
7. LIDAR sensitivity analysis and replicated DNase assay.
8. Limits: weak curvature, dependence, non-Gaussian noise, misspecified
   `kappa` and model checking.

The opening sentence should answer the SIMODS criticism directly: the paper
addresses reliable uncertainty quantification for a scientifically meaningful
structural transition, not a new representation in search of a task.

## 8. What is still needed for a strong submission

1. An external adversarial proof review of T1--T4. The internal audit has made
   the singular-covariance and zero-noise steps in T4 explicit, but an
   independent specialist is still needed.
2. A systematic database/citation-chain priority audit through 2026. Davies et
   al. and Schmidt-Hieber et al. are now included, but citation chaining is not
   complete.
3. A function-class lemma that verifies the new general contrast-margin
   theorem on irregular designs. The `1/(2 gamma+1)` rate is now proved under
   explicit margin, support and norm conditions.
4. A matching confidence-set length lower bound if aiming above a solid
   specialist statistics journal.
5. A full implementation of the multiscale Davies confidence region would
   strengthen the historical comparison, although the exact matched PBP
   baseline now covers the main fairness objection.
6. Standalone archive metadata and a tagged release for the new
   `shapecontrast` namespace.

## 9. Implementation update after the audit

The new `shapecontrast` namespace does not build a dense contrast matrix.
Uniform designs use prefix sums, while irregular designs evaluate weights in
bounded-memory chunks. Tests compare it with the frozen dense implementation
on both regular and irregular designs.

On the current machine, the full three-separation family at `n=1,000,000`
contained 5,999,750 contrasts. It used 53.4 MiB for the stored family, built in
0.035 seconds and evaluated one curve in 0.732 seconds. A dense matrix with the
same rows would require about 44,702 GiB. These times are hardware-specific;
the stored sizes are deterministic. Full results are in
`results/sci/matrix_free_scaling/`.

## 10. Go/no-go decision

**Go** for a new shape-constrained-inference paper.  The central known problem,
finite-sample solution, frontier comparator and honest negative regime are all
present.  **No-go** for presenting it as an HCRD superiority paper or as a
generic hybrid that boosts point accuracy.  The strongest current story is
“frontier point estimator plus a new honest uncertainty layer.”
