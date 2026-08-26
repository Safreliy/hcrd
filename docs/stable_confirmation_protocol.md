# S2 frozen confirmation: stable quadratic guide

This protocol was written before running S2.  The L1 and quadratic grids in
`results/proximal_pilot` and `results/quadratic_pilot*` are exploratory and are
not confirmatory evidence.

## Frozen design

- Generator: `alternating_chord_lobes` with its default eight intervals and
  variable amplitudes.
- Latent seeds: `20260950 + latent_index`, 30 independent latent signals.
- Observation repetitions: 10 per latent signal and noise condition.
- Noise standard deviations: 0.01, 0.03, 0.05, and 0.10.
- Quadratic guide: divided-curvature penalty with fixed
  `regularization = 3.0` in every condition.  It is not retuned by signal or
  noise level.
- Other methods: raw centred HCRD, plug-in MAD-thresholded HCRD (`z=3.5`),
  fixed Gaussian-guided HCRD (`sigma=2`), and the previously pilot-calibrated
  adaptive Gaussian guide.
- HCRD output: first centred level.

## Estimands and statistical unit

The independent statistical unit is the latent signal, not each repeated noisy
observation.  Observation-level metrics are averaged within latent signal
before paired inference.

Primary morphology endpoint:

- target-knot F1 with a one-sample matching tolerance.

Secondary endpoints:

- baseline MSE;
- median pairwise knot Jaccard within a latent signal;
- maximum observed L2 output/input perturbation ratio for the stable guide and
  for the subsequent hard HCRD baseline.

For quadratic-guide HCRD minus each comparator, report paired mean differences,
95% paired bootstrap intervals over the 30 latent signals, exact sign tests,
and Holm correction across the four comparators separately within each noise
condition and endpoint.  A positive F1 difference and a negative MSE
difference favour quadratic-guide HCRD.  No universal-superiority claim is
allowed; a morphology advantage and an MSE disadvantage may coexist.

## Interpretation guardrail

The theorem concerns the outer quadratic guide and residual maps.  The hard
HCRD knot hierarchy is not labelled globally Lipschitz even if all sampled
ratios happen to be below one.
