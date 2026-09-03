# E33 priority audit: honest inference for an S-shaped inflection set

## Claim under audit

E33 is **not** novel because it uses chord inequalities or multiscale tests.
Those ingredients are classical.  The candidate contribution is narrower:

> invert a fixed family of simultaneous convexity/concavity contrasts into a
> finite-sample confidence set for every admissible inflection of a possibly
> nonsmooth S-shaped regression function, and use that set as an uncertainty
> layer around a modern point estimator.

The target class permits a cusp, a one-sided onset, affine regions and a jump
at the inflection.  The primary theorem assumes a fixed ordered design,
independent Gaussian errors and a known upper noise scale.  It does not assume
existence of derivatives, continuity at the inflection or uniqueness of the
inflection.

## Closest prior work checked

### Dümbgen and Spokoiny (2001)

*Multiscale Testing of Qualitative Hypotheses*, Annals of Statistics 29,
124--152, DOI `10.1214/aos/996986504`.

This is direct priority for multiscale testing of positivity, monotonicity and
concavity.  It explicitly produces simultaneous statements identifying
intervals on which a qualitative hypothesis is violated and observes that
multiple tests lead to nonparametric confidence sets.  Its continuous white
noise construction, optimized test signals and minimax-adaptive testing theory
are substantially more sophisticated than E33's Bonferroni calibration.

It does **not**, in the statements and examples checked, formulate the
convex-left/concave-right S-shaped model, invert local convexity and concavity
statements into a confidence set for its unknown inflection, cover the entire
possibly non-singleton admissible-inflection set, or evaluate such a set on the
four nonsmooth S-shaped signals used by Feng et al.  E33 must cite this paper as
the principal methodological ancestor and must not claim novelty for the
contrast scan itself.

Source: <https://doi.org/10.1214/aos/996986504>.

### Liao and Meyer (2017) and `ShapeChange`

*Change-point estimation using shape-restricted regression splines*, Journal
of Statistical Planning and Inference 188, 8--21, DOI
`10.1016/j.jspi.2017.03.007`.

This work estimates a mode or inflection by profiling constrained cubic
B-spline fits and proves convergence rates for the point estimator.  The
current CRAN package `ShapeChange` 1.5 additionally exposes a nominal 95%
bootstrap interval through `changept(..., ci=TRUE)`.  Inspection of the package
1.5 source shows that, for iid Gaussian errors, it resamples residuals around
the fitted spline, refits the change point and returns the 2.5% and 97.5%
empirical quantiles.  The documentation describes the inflection target as a
smooth convex--concave curve and uses cubic B-splines.

Consequently, **"the first confidence interval for an inflection point" is
false and forbidden**.  `ShapeChange` is the required uncertainty comparator.
The defensible distinction is that its documented bootstrap interval is tied
to a smooth spline model and no finite-sample, uniform coverage theorem over
the full nonsmooth S-shaped class was found.  E33 should compare coverage and
width honestly, not assume bootstrap failure.

Sources: <https://doi.org/10.1016/j.jspi.2017.03.007> and
<https://CRAN.R-project.org/package=ShapeChange>.

### Feng, Chen, Han, Carroll and Samworth (2022)

*Nonparametric, Tuning-Free Estimation of S-Shaped Functions*, JRSSB 84,
1324--1352, DOI `10.1111/rssb.12481`.

This is the estimand-defining frontier and the source of the official
`Sshaped` least-squares estimator.  It permits discontinuity at the inflection,
proves estimation and localization rates, supplies minimax lower bounds and
uses four signals containing a cusp, a one-sided onset, a jump and a smooth
logistic curve.  It explicitly identifies valid uncertainty quantification for
the inflection as a challenging future direction.

E33 therefore complements rather than replaces `Sshaped`: the official LSE is
the point estimate, while the E33 confidence set is computed from the same raw
response through an independent theorem.  Projection of the LSE point into an
E33 set cannot increase localization error whenever that set covers the true
inflection.

Source: <https://doi.org/10.1111/rssb.12481>.

### Cai, Low and Xia (2013)

*Adaptive confidence intervals for regression functions under shape
constraints*, Annals of Statistics 41, 722--750, arXiv `1305.5673`.

This work gives honest adaptive confidence intervals for the **value of a
monotone or convex regression function at a point**.  It is important direct
priority for honesty and adaptation under shape constraints, but its target is
not the location separating convex and concave pieces.

Source: <https://arxiv.org/abs/1305.5673>.

### Other nearby lines

- Kachouie and Schwartzman (2013) give a local-polynomial point estimator for
  a single smooth inflection; no full-class honest interval was found.
- SMUCE and related multiscale change-point methods invert multiscale tests to
  obtain confidence sets for step-function change points.  They are a close
  conceptual precedent for test inversion, but address a different structural
  model and estimand.
- SiZer and derivative-band methods localize significant changes of slope
  under smoothing assumptions.  E32 belongs to this smoother route and does
  not cover the full Feng class.

## Priority conclusion as of 2026-09-02

The literature audit supports a **candidate**, not yet categorical, novelty
claim:

> A derivative-free finite-sample confidence set for the entire admissible
> inflection set of a nonparametric S-shaped regression function, valid for
> discontinuous and nonsmooth signals, obtained by deterministic inversion of
> simultaneous multiscale shape contrasts and composable with any point
> estimator.

The claim must remain qualified as "to our knowledge" until a formal database
search and citation-chaining audit is completed.  Publication strength requires
all of the following:

1. the finite-sample coverage theorem and its proof;
2. a localization-rate result under an explicit local separation condition;
3. frozen comparison with the official `Sshaped` point estimator;
4. frozen interval comparison with `ShapeChange` bootstrap on both smooth and
   nonsmooth Feng signals;
5. a clear negative result on weak logistic curvature rather than selective
   reporting;
6. an unknown-noise extension or an explicit known-scale limitation.

The broad data-science problem is not "classify time series with HCT".  It is:
researchers can fit a flexible S-shaped response and report a change from
accelerating to diminishing returns, yet available point estimates and smooth
bootstrap intervals do not provide a finite-sample guarantee robust to the
nonsmooth regimes allowed by the leading S-shaped model.  E33 targets that
inferential gap.
