# Internal adversarial audit of Theorem E35.1

**Scope:** line-by-line review of the finite-sample variance-envelope proof.
This is an internal audit, not a substitute for an independent specialist.

## Review round 1

**First failure.** The proof invoked Anderson's inequality without stating
whether the result allows a singular Gaussian covariance.

**Why it matters.** The covariance `R Sigma R` is singular because `R` is a
residual projection. A theorem stated only for a positive density on the full
space would not apply directly.

**Precise question.** Does the shift inequality hold for every centred
jointly Gaussian vector, including a degenerate one, and for the closed
Euclidean ball used here?

**Repair.** The proof now states the required version explicitly. Standard
forms of Anderson's lemma cover centred jointly normal vectors without a
nonsingularity assumption. The Euclidean ball is convex and symmetric, so the
shift from `Z` to `R mu + Z` can only reduce its small-ball probability.

## Review round 2

**First failure after repair.** The phrase "Gaussian quadratic-form
inequality" did not identify the random variable or the exact trace terms.

**Why it matters.** A wrong factor of two would change the denominator of the
variance envelope and invalidate the coverage theorem.

**Precise question.** For `Z ~ N(0,A)`, is the claimed lower bound exactly

$$
\|Z\|^2\geq \operatorname{tr}(A)
-2\sqrt{\operatorname{tr}(A^2)t}
$$

with failure probability at most `exp(-t)`?

**Repair.** Diagonalizing `A` gives
`||Z||^2 = sum_j lambda_j G_j^2`. The Laurent--Massart weighted chi-square
lower-tail bound has exactly the displayed form. Taking `t=log(1/eta)` gives
the claimed event.

## Review round 3

**First failure after repair.** The proof divided through quantities derived
from `T=tr(A)` without treating `T=0`.

**Why it matters.** Degenerate zero-noise cases belong to the stated parameter
space.

**Precise question.** What happens when `T=0`?

**Repair.** Since
`T >= (1/2) sum_i sigma_i^2`, `T=0` implies that every `sigma_i=0`. The desired
upper-envelope event then holds automatically. The rest of the proof may
assume `T>0`.

## Remaining risk

No algebraic counterexample was found after these repairs. The next external
review should check the exact cited versions of Anderson's lemma and the
Laurent--Massart inequality, including measurability and equality cases. It
should also challenge the scientific interpretation of the declared `kappa`;
the proof does not estimate `kappa` from one response curve.
