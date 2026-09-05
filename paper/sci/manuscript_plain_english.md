# Finite-Sample Confidence Sets for Nonsmooth Convex-to-Concave Transitions

Working manuscript spine. The final paper should keep this level of English:
short sentences, one idea per paragraph, and plain explanations before formal
statements.

## Abstract

Many scientific curves have a transition from increasing returns to
diminishing returns. A point estimate can locate this transition, but it does
not show which other locations are still compatible with the data. This is a
hard problem when the curve has a kink, a jump, or a flat section. Standard
smooth bootstrap intervals can then be misleading.

We introduce shape-contrast inversion (SCI). The method tests a fixed set of
local chord inequalities and inverts the reliable signs. It returns an outer
confidence set for every transition admitted by at least one
shape-constrained continuation of the sampled mean values. Under fixed-design
Gaussian regression, the coverage
guarantee is finite-sample. It does not require a derivative, a continuous
transition, or a unique transition. We also give versions for unknown constant
noise, bounded changes in noise variance, and independent replicate curves.

In 80,000 fresh simulations, known-scale SCI covered the full identified
transition set in 97.70% to 98.04% of responses across 16 settings formed from
four published S-shaped benchmark curves.
The method gave useful localization for strong nonsmooth transitions and wide
sets when the smooth signal was weak. Real-data studies on atmospheric LIDAR
and 11 replicated DNase assay runs show how the answer changes with the noise
information available. SCI is designed as an uncertainty layer for an
S-shaped point estimator, not as a replacement for one.

## 1. Introduction

Researchers often want to know where a response changes from acceleration to
saturation. Examples include dose-response curves, biological growth, disease
progression, and production curves. Modern S-shaped regression methods provide
a useful point estimate. The point alone, however, does not answer a basic
question: which transition locations can the data rule out?

This question is especially important when the curve is not smooth. A real
transition may be a kink, a jump, a one-sided onset, or a flat interval. In
these cases a method built around a smooth fitted derivative can report a
narrow interval even when its assumptions are wrong.

At a fixed design, the data identify the mean values at the observed points,
not the full curve between them. We therefore study all transition locations
admitted by at least one curve that passes through those mean values and is
convex before the transition and concave after it. This target also allows a
flat transition region. Reporting the whole set avoids a false claim of
precision.

This design-level set is easy to compute. We scan adjacent slopes from the
left to find convex prefixes and from the right to find concave suffixes. A
single linear inequality then gives the feasible part of each gap between
design points. The full calculation takes linear time in the sample size, and
it also proves that the identified target is always either empty or one closed
interval.

SCI starts from local chord contrasts. A positive mean contrast rules out a
wholly concave support. A negative mean contrast rules out a wholly convex
support. Neither sign establishes the shape over the whole interval.
We construct simultaneous bounds for all contrast means. We then use only the
signs that are supported by these bounds. The largest left support endpoint
among certified positive contrasts gives a lower limit for the transition.
The smallest right support endpoint among certified negative contrasts gives
an upper limit. A wide confidence set reflects the chosen contrasts and
calibration; it does not prove that every other valid method must be wide.

The main result is simple and finite-sample. If all contrast bounds cover their true
means, the SCI set contains every compatible transition. A simultaneous
Gaussian band makes this event occur with probability at least `1-alpha`.
The same inversion can use other valid bands. This modular view leads to
extensions for unknown noise and replicated experiments.

### Contributions

1. We define a design-level identified target: the closure of the set of all
   convex-to-concave transitions allowed by shape-constrained continuations of
   the sampled mean vector, rather than an assumed unique inflection point.
2. We give a direct finite-sample confidence set for this target in
   fixed-design Gaussian regression. The coverage theorem allows flat parts,
   kinks, and jumps.
3. We give finite-sample extensions for unknown constant noise, a stated bound
   on changing noise variance, and independent replicate curves with arbitrary
   within-curve covariance.
4. We show the known cubic localization order while keeping validity over the
   wider nonsmooth class. The width bound holds with probability `1-eta`
   independently of the fixed coverage budget; combining both events gives
   joint coverage and localization with probability `1-alpha-eta`.
5. We provide a matrix-free implementation. On the benchmark machine it
   evaluated about six million contrasts for one million observations in less
   than one second, while the equivalent dense matrix would require about
   44,702 GiB.
6. We report both successful and weak-information regimes. The method can
   return most of the design range when its chosen contrasts and calibration
   do not yield enough sign evidence for precise localization.

## 2. Relation to earlier work

SCI does not claim to be the first confidence method for an inflection point.
Davies, Kovac and Meise (2009) built a finite-sample confidence region for a
regression curve and studied inflection locations. Schmidt-Hieber, Munk and
Dümbgen (2013) used multiscale sign statements to locate roots of operators,
including second-derivative roots. Both lines already contain the regular
`(log(n)/n)^(1/7)` localization order.

The contribution here is narrower. SCI directly targets all locations that
are compatible with the sampled mean vector and one convex-to-concave
transition. Its basic coverage argument does not need the derivative to exist
and does not require the transition to be unique or continuous. The output is
also easy to compute without fitting a full confidence region for the curve.

The `Sshaped` least-squares estimator of Feng et al. is the main point
estimator in our experiments. SCI complements this estimator by adding
uncertainty. `ShapeChange` is the main interval comparator. Its residual
bootstrap is useful under a smooth spline model, but it does not provide the
same uniform finite-sample statement over the nonsmooth class studied here.

## 3. Main result in plain language

Suppose the true mean curve is convex before every valid transition and
concave after it. A reliable positive contrast cannot lie entirely to the
right of a valid transition. A reliable negative contrast cannot lie entirely
to its left. These two facts create lower and upper limits. Because the
contrast bounds hold together, the limits contain every valid transition at
the same time.

The word “outer” matters. SCI may include locations that are not true
transitions. Its guarantee is that it does not remove a true compatible
location more often than allowed by the confidence level. Strong data make the
set narrow. Weak data make it wide.

## 4. Evidence to report

The paper should separate four questions:

- coverage: does the set contain the true transition?
- information: how narrow is the set?
- robustness: what happens at kinks, jumps, flat parts, and changing variance?
- computation: can the method run without a dense contrast matrix?

The high-precision known-scale audit used 5,000 responses in each of 16 fixed
settings. A deterministic post-audit calculation checked coverage of the full
identified set without regenerating responses. Only two of 80,000 labels
changed, and coverage remained 97.70% to 98.04%. Median width among nonempty
outputs was small for the jump,
moderate for the cusp, and close to the full domain for the weak logistic
curve. A frozen zero-empty diagnostic failed because empty sets occurred in up
to 2.28% of trials. This is below the theorem's 5% allowance, but it remains a
reported failed diagnostic.

We also compare SCI with a pointwise-band baseline (PBP). PBP separately finds
a convex prefix and a concave suffix inside a simultaneous confidence box. The
two fitted pieces do not have to join at the same transition value, so this is
a conservative split relaxation, not an exact projection onto the SCI function
class. Its finite-sample guarantee covers the same identified target when PBP
is given the same declared domain; otherwise its default domain is only the
observed range. Across the 16 fixed cells, SCI reduced median width among nonempty
outputs by 19.4% to 75.7% for the cusp, onset, and jump signals. The same range
held when both methods were compared only on trials where both were nonempty.
Both methods returned essentially the
full observed range for the weak logistic signal. PBP is our simple baseline,
not the official method of Davies et al. or the strongest possible projection.

The DNase analysis uses 11 assay runs as independent replicates. Dependence
between concentrations inside a run is allowed. SCI gives the concentration
interval `[0.78125, 12.5]`, while a logistic point fit gives `4.14`. The wide
upper side is scientifically meaningful: these data support a lower limit,
but they do not locate the end of the transition precisely within the observed
range.

## 5. Limits

The exact guarantees are Gaussian. The heteroskedastic single-curve extension
needs a declared variance-ratio bound. The replicate extension needs
independent runs with a common mean and covariance. SCI is conservative when
many contrasts are weak. It is not a generic point-estimation improvement.
The current cubic rate is not a new minimax result. The general local-order
rate is conditional on an explicit contrast margin; a broad function-class
corollary and a matching lower bound remain open. A future comparison should
use one globally joined same-band projection rather than the current split
relaxation.

## References to include

- Davies, Kovac and Meise (2009), *Nonparametric Regression, Confidence
  Regions and Regularization*, Annals of Statistics.
- Schmidt-Hieber, Munk and Dümbgen (2013), *Multiscale Methods for Shape
  Constraints in Deconvolution*, Annals of Statistics.
- Feng, Chen, Han, Carroll and Samworth (2022), *Nonparametric, Tuning-Free
  Estimation of S-Shaped Functions*, JRSSB.
- Liao and Meyer (2017), *Change-Point Estimation Using Shape-Restricted
  Regression Splines*, Journal of Statistical Planning and Inference.
- Dümbgen (2003), *Optimal Confidence Bands for Shape-Restricted Curves*,
  Bernoulli.
- Cai, Low and Xia (2013), *Adaptive Confidence Intervals for Regression
  Functions Under Shape Constraints*, Annals of Statistics.
- Frick, Munk and Sieling (2014), *Multiscale Change Point Inference*, JRSSB.
- Chernozhukov, Hong and Tamer (2007), *Estimation and Confidence Regions for
  Parameter Sets in Econometric Models*, Econometrica.
