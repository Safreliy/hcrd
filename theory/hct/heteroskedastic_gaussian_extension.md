# HCT under unknown bounded heteroskedasticity

## Problem

The known-scale HCT theorem remains valid when the independent Gaussian
errors have unequal *known* variances, after each contrast is standardized by
its exact standard deviation.  In applications the variances are usually
unknown.  With one response at each fixed design point, completely arbitrary
heteroskedasticity is not identifiable: a single coordinate may have an
arbitrarily large variance without leaving replicates from which to estimate
it.  We therefore make the weakest explicit global restriction used here,

\[
 \max_i \sigma_i^2 \leq \kappa\,\bar\sigma^2,
 \qquad
 \bar\sigma^2:=n^{-1}\sum_{i=1}^n\sigma_i^2,
\]

where the sensitivity parameter \(\kappa\geq1\) is specified before seeing the
confidence set.

## Theorem E35.1 (finite-sample variance envelope)

Let

\[
 Y_i=\mu_i+\epsilon_i,\qquad
 \epsilon_i\ \text{independent},\quad
 \epsilon_i\sim N(0,\sigma_i^2),
\]

where the mean vector \(\mu\in\mathbb R^n\) is arbitrary.  Fix independently
of the responses a partition of \([n]\) into consecutive blocks of size two
or three, and let \(R\) be the orthogonal residual projector after fitting a
separate constant to each block.  Put \(Q=\|RY\|_2^2\), choose
\(\eta\in(0,1)\), and define

\[
 d_{n,\kappa,\eta}
 :=1-2\sqrt{\frac{2\kappa\log(1/\eta)}{n}}.
\]

If \(d_{n,\kappa,\eta}>0\), then

\[
 \widehat\sigma_{\max}^2
 :=\frac{2\kappa Q}{n d_{n,\kappa,\eta}}
\]

satisfies

\[
 \inf_{\mu,\,\sigma:\,\max_i\sigma_i^2\leq\kappa\bar\sigma^2}
 \Pr_{\mu,\sigma}
 \left\{\widehat\sigma_{\max}\geq\max_i\sigma_i\right\}
 \geq 1-\eta.
\]

Thus lack of fit of the blockwise-constant nuisance model can widen the bound,
but cannot invalidate it.

### Proof

Write \(\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2)\),
\(Z=R\epsilon\), \(A=R\Sigma R\), and \(T=\operatorname{tr}(A)\).
Anderson's Gaussian inequality for the centrally symmetric Euclidean ball
gives, for every \(q\),

\[
 \Pr\{\|R\mu+Z\|_2^2\leq q\}
 \leq \Pr\{\|Z\|_2^2\leq q\}.
\]

The lower-tail Gaussian quadratic-form inequality gives, with probability at
least \(1-\eta\),

\[
 \|Z\|_2^2
 \geq T-2\sqrt{\operatorname{tr}(A^2)\log(1/\eta)}.
\]

Every diagonal element of a block residual projector is
\(1-|B|^{-1}\geq1/2\), so

\[
 T=\operatorname{tr}(R\Sigma)
 \geq \tfrac12\sum_i\sigma_i^2
 =\tfrac n2\bar\sigma^2.
\]

Moreover,

\[
 \operatorname{tr}(A^2)
 \leq\|A\|_{\mathrm{op}}T
 \leq(\max_i\sigma_i^2)T
 \leq\frac{2\kappa}{n}T^2.
\]

Combining the last three displays and the shift inequality yields
\(Q\geq T d_{n,\kappa,\eta}\) outside an event of probability at most
\(\eta\).  On that event,

\[
 \max_i\sigma_i^2
 \leq\kappa\bar\sigma^2
 \leq\frac{2\kappa T}{n}
 \leq\frac{2\kappa Q}{n d_{n,\kappa,\eta}},
\]

which proves the claim. \(\square\)

## Corollary E35.2 (honest heteroskedastic HCT set)

Let \(w_1,\ldots,w_M\) be the predeclared HCT chord contrasts and set
\(\beta=\alpha-\eta>0\).  Form simultaneous bands with radii

\[
 z_{1-\beta/(2M)}\,\widehat\sigma_{\max}\,\|w_j\|_2.
\]

Invert their certified signs exactly as in Theorem E33.1.  The resulting set
contains every admissible convex-to-concave change location with probability
at least \(1-\alpha\), uniformly over all S-shaped mean vectors and all
variance vectors satisfying the stated \(\kappa\)-bound.

### Proof

If the envelope event of Theorem E35.1 holds, every contrast radius is at
least its oracle homoskedastic upper bound because
\(\operatorname{Var}(w_j^T\epsilon)\leq
\sigma_{\max}^2\|w_j\|_2^2\).  A Gaussian Bonferroni event for the oracle
radii fails with probability at most \(\beta\).  The envelope event fails
with probability at most \(\eta\).  A union bound, requiring no independence
between the two events, gives failure probability at most
\(\eta+\beta=\alpha\); deterministic sign inversion then proves coverage.
\(\square\)

## What the assumption means

- \(\kappa=1\) is homoskedasticity.
- \(\kappa=2\) allows the largest pointwise variance to be twice the average.
- The theorem is a sensitivity guarantee: validity at a reported \(\kappa\)
  is conditional on that scientifically interpretable bound being credible.
- Without replicates or some restriction such as bounded \(\kappa\), no
  finite, informative, distribution-free upper bound on every pointwise
  variance is possible from one observation per location.

