# Iterative-filtering external comparison E3 (frozen before execution)

## Purpose and scope

E3 is a fresh-seed, task-specific comparison with the author-linked Python
implementation of Iterative Filtering (IF).  It tests the same affine
chord-lobe baseline-recovery class used by the exact-recovery theorem.  It is
not a claim that HCRD is a better IMF or time-frequency decomposition.

The three pre-specified paired hypotheses are:

1. at noise sigma 0, centred HCRD has lower baseline MSE than an oracle
   slow-tail IF baseline;
2. at sigma 0.03, MAD-thresholded HCRD has lower baseline MSE than that oracle;
3. at sigma 0.10, MAD-thresholded HCRD has lower baseline MSE than that oracle.

## Frozen data and methods

- Generator: `alternating_chord_lobes` with affine baseline, equal lobe
  amplitudes, and the package defaults for sample count and knots.
- Independent signal seeds: `20261202 + 10000 * noise_index + trial`.
- Conditions: sigma `0`, `0.03`, and `0.10`; 50 signals per condition.
- HCRD: centred minimum-curvature rule at sigma 0; plug-in MAD curvature
  threshold with `z_score=3.5` at the noisy conditions.
- IF: `iterativefiltering==1.0.4`, imported as `fifpy.IF`, with its unmodified
  defaults (`delta=0.001`, `ExtPoints=3`, `NIMFs=200`, `MaxInner=200`,
  `Xi=1.6`, `alpha='ave'`, `BCmode='clip'`).  No HCRD-derived parameter is
  passed to IF.
- IF baseline candidates: the slowest returned component alone and cumulative
  sums of up to the four slowest returned components.  Selection minimizes MSE
  against the known latent baseline separately for every signal.  This oracle
  deliberately favours IF and is not deployable.
- Exact reconstruction error of the returned IF components and the selected
  tail length are retained as diagnostics.
- Runtime: Python 3.12 because the published package excludes Python 3.13;
  independent signals use eight worker processes, with numerical-library
  thread counts fixed to one.

## Frozen inference

For each noise condition, form one paired baseline-MSE difference per latent
signal (HCRD minus oracle IF).  Report the mean difference, a seeded 20,000-draw
paired bootstrap 95% interval, HCRD win rate, and an exact two-sided sign test.
Apply Holm correction across the three pre-specified comparisons.  Superiority
is supported only if the interval lies strictly below zero and the adjusted
sign-test p-value is below 0.05.  Ties within `1e-12` are excluded from the sign
test.

All trial rows, aggregate summaries, comparisons, dependency versions,
elapsed time, and this protocol's SHA-256 are written to
`results/iterative_filtering_e3`.
