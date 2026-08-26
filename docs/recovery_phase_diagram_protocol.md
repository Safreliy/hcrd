# Finite-sample chord-lobe recovery phase diagram

## Question

Does the finite-sample HCRD recovery condition predict the transition from
failed to exact boundary recovery on its declared signal class?

## Signal class

Every signal has `K` equal-width, equal-amplitude parabolic lobes with
alternating sign around a random affine baseline. Adjacent lobes meet at a
single sampled zero-curvature point. Gaussian sample noise has standard
deviation one. The four fixed configurations are `(K,m) = (2,4), (4,8),
(8,16), (16,8)`, where `m` is the number of sample intervals per lobe.

The x-axis is the normalized active-curvature strength

`rho = gamma / tau`,

where `tau` is the simultaneous curvature threshold at `delta = 0.05`. The
theorem certifies exact first-level knots and the stated reconstruction bounds
for `rho > 2`.

## Frozen grid and endpoints

- Curvature ratios: `0.50, 0.75, 1.00, 1.15, 1.30, 1.45, 1.60, 1.75,
  1.90, 2.05, 2.25`.
- Replications: 1000 independent draws per cell.
- Primary endpoint: exact equality of the estimated and population knot sets.
- Secondary endpoints: zero-tolerance knot F1, symmetric Hausdorff boundary
  error divided by lobe width, baseline/detail MSE, retained-knot compression,
  and construction time.
- Certificate audit: exact knots together with the theorem's sup-norm baseline
  and detail bounds.

The experiment tests a sufficient bound on the declared class. It is not a
claim that all waveform populations have this transition or that the bound is
necessary.

## Reproduction

```bash
python experiments/run_recovery_phase_diagram.py
python experiments/generate_recovery_phase_figure.py
```
