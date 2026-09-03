# Proof DAG and hostile audit for E35

## Theorem signature

**Input.** One fixed-design Gaussian regression vector, a response-independent
partition into blocks of size two or three, failure allocation \(\eta\), and a
valid constant \(\kappa\) with
\(\sigma_{\max}^2\leq\kappa n^{-1}\sum_i\sigma_i^2\).

**Output.** A data-dependent scalar \(\widehat\sigma_{\max}\) that dominates
every pointwise noise standard deviation with probability at least
\(1-\eta\), uniformly over the unrestricted mean vector.  Composing it with
the E33 contrast event yields an inflection confidence set with coverage at
least \(1-\alpha\).

## Nodes

| ID | Type | Statement |
|---|---|---|
| A1 | assumption | errors are independent centred Gaussian with diagonal covariance \(\Sigma\) |
| A2 | assumption | residual partition is fixed independently of the response and every block has size 2 or 3 |
| A3 | assumption | \(\sigma_{\max}^2\leq\kappa\bar\sigma^2\) with declared finite \(\kappa\) |
| A4 | assumption | \(d_{n,\kappa,\eta}>0\) |
| D1 | definition | \(R\) is the block-constant residual projector, \(Q=\|RY\|^2\), \(A=R\Sigma R\), \(T=\operatorname{tr}A\) |
| L1 | external lemma | Anderson inequality: translating a centred Gaussian cannot increase probability of a centred Euclidean ball |
| L2 | external lemma | Gaussian quadratic lower tail: \(\|R\epsilon\|^2\geq T-2\sqrt{\operatorname{tr}(A^2)t}\) with probability \(1-e^{-t}\) |
| L3 | algebra | \(T\geq(n/2)\bar\sigma^2\) because every diagonal entry of \(R\) is at least \(1/2\) |
| L4 | algebra | \(\operatorname{tr}(A^2)\leq(2\kappa/n)T^2\) |
| C1 | claim | \(Q\geq T d_{n,\kappa,\eta}\) with probability at least \(1-\eta\), uniformly in the mean |
| T1 | theorem | \(\widehat\sigma_{\max}^2=2\kappa Q/(nd)\geq\sigma_{\max}^2\) with probability at least \(1-\eta\) |
| L5 | elementary lemma | Gaussian Bonferroni event covers all fixed chord-contrast means using oracle radius \(z\sigma_{\max}\|w_j\|\) |
| C2 | claim | on T1's event, every estimated-envelope radius dominates the oracle radius |
| T2 | corollary | E33 sign inversion covers the entire admissible inflection set with probability at least \(1-\alpha\) |

## Edges

1. `A1 + A2 + D1 -> L1, L2`.
2. `A2 + D1 -> L3`.
3. `A1 + A3 + L3 -> L4` through
   \(\|R\Sigma R\|_{op}\leq\sigma_{\max}^2\).
4. `L1 + L2 + L4 + A4 -> C1` with \(t=\log(1/\eta)\).
5. `A3 + L3 + C1 -> T1`.
6. `A1 + fixed contrasts -> L5`.
7. `T1 + L5 -> C2`, and a union bound spends \(\eta+(\alpha-\eta)\).
8. `C2 + E33 deterministic inversion -> T2`.

No edge assumes that the blockwise-constant nuisance fit is correct.  The
unrestricted mean enters only as the translation in L1.

## Hostile checks

1. **Does noncentral residual energy invalidate the lower-tail calibration?**
   No.  The rejection event is a centred ball in the residual subspace;
   Anderson's inequality makes its probability largest at zero projected
   mean.  Misspecification is conservative.
2. **Is \(\operatorname{tr}(A^2)\leq\|A\|_{op}\operatorname{tr}(A)\)
   valid?** Yes, because \(A\) is positive semidefinite.
3. **Can projection increase the covariance operator norm?** No:
   \(\|R\Sigma R\|_{op}\leq\|R\|_{op}^2\|\Sigma\|_{op}
   =\sigma_{\max}^2\).
4. **Where is block size used?** Only in L3.  A singleton has residual
   diagonal zero and would destroy the uniform trace-to-average relation.
5. **Is independence between scale estimation and HCT contrasts needed?** No.
   The proof intersects two marginal high-probability events and uses a union
   bound.
6. **Can \(\kappa\) be estimated from the same single-observation design
   without further structure?** Not uniformly.  A high-variance coordinate
   can be hidden at an unreplicated location.  Reported inference must remain
   conditional on the declared sensitivity bound.

## Failure boundaries

- If the supplied \(\kappa\) is smaller than the true variance ratio, T1 has
  no coverage guarantee.
- Arbitrary dependence breaks A1 and both the quadratic-form and contrast
  calibrations.
- Non-Gaussian errors require new small-ball and contrast-tail results.
- Selecting the residual partition after looking at responses requires an
  additional simultaneous-selection correction.
- When \(d_{n,\kappa,\eta}\leq0\), the stated finite-sample concentration
  inversion is deliberately undefined rather than silently returning a
  misleading finite number.

