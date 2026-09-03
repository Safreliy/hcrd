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

## 5. Real LIDAR illustration

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

## 6. Defensible novelty statement

Subject to a final database and citation-chain audit:

> To our knowledge, SCI is the first derivative-free finite-sample confidence
> construction for the entire admissible inflection set in nonparametric
> S-shaped regression that remains valid for discontinuous and nonsmooth
> signals, and the first such construction extended to unknown Gaussian scale
> and bounded unknown heteroskedasticity without imposing a smooth mean model.

Forbidden claims:

- “first confidence interval for an inflection point” — `ShapeChange` already
  supplies a residual-bootstrap interval;
- “first multiscale convexity test” — Dümbgen--Spokoiny is direct prior art;
- “better point estimator than Sshaped” — SCI is an inference layer;
- “distribution-free” — the current calibration is Gaussian;
- “valid for arbitrary heteroskedasticity” — validity requires a correct
  declared `kappa` or additional replicates/variance structure.

## 7. Manuscript spine

1. Practical failure: point localization without trustworthy uncertainty.
2. S-shaped model and the possibly set-valued inflection estimand.
3. SCI algorithm and deterministic inversion lemma.
4. Finite-sample known-scale theorem and cubic localization rate.
5. Unknown homoskedastic and bounded-heteroskedastic extensions.
6. Frozen Feng-benchmark comparison with `Sshaped` and `ShapeChange`.
7. LIDAR sensitivity analysis.
8. Limits: weak curvature, dependence, non-Gaussian noise, misspecified
   `kappa` and model checking.

The opening sentence should answer the SIMODS criticism directly: the paper
addresses reliable uncertainty quantification for a scientifically meaningful
structural transition, not a new representation in search of a task.

## 8. What is still needed for a strong submission

1. An external adversarial proof review of T1--T4, especially the Anderson and
   Gaussian quadratic-form step in T4.
2. A systematic database/citation-chain priority audit through 2026.
3. A general local-order diameter theorem, or an explicit decision to market
   the proved cubic rate and treat other orders empirically.
4. A matching confidence-set length lower bound if aiming above a solid
   specialist statistics journal.
5. A second real dataset with raw replicates or a defensible variance-ratio
   bound; the LIDAR data alone cannot validate `kappa`.
6. A standalone `sci` API and short reproducible vignette, decoupled from the
   historical HCRD namespace.

## 9. Go/no-go decision

**Go** for a new shape-constrained-inference paper.  The central known problem,
finite-sample solution, frontier comparator and honest negative regime are all
present.  **No-go** for presenting it as an HCRD superiority paper or as a
generic hybrid that boosts point accuracy.  The strongest current story is
“frontier point estimator plus a new honest uncertainty layer.”

