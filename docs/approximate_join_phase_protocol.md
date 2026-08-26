# Approximate-join recovery phase experiment

Protocol fixed on 2026-08-26 after the approximate-join corollary was written
and before any result from this experiment was computed.

## Question

Does the sufficient boundary

`gamma / tau > eta / tau + 2`

predict exact first-level HCRD recovery when adjacent parabolic chord lobes
have unequal amplitudes and therefore nonzero sampled join curvature? How much
is lost if the algorithm retains the zero-join tolerance `tau` rather than the
join-aware tolerance `eta + tau`?

## Fixed design

- iid Gaussian sample noise with `sigma=1`, `delta=0.05`, and unit spacing;
- configurations `(lobes, samples_per_lobe)` equal to `(4,8)`, `(8,8)`, and
  `(8,16)`;
- join ratios `eta/tau` in `0, 0.25, 0.5, 1, 1.5, 2`;
- active-curvature ratios `gamma/tau` in
  `1.5, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5`;
- 1000 independent draws per configuration and ratio pair, for 162,000 draws;
- master seed `20260826`.

For each ratio pair, lobe amplitudes alternate between `A_min` and
`A_min + Delta A`. They are chosen analytically so that the minimum active
curvature is exactly `gamma` and every join magnitude is exactly `eta`.

Each noisy signal is decomposed twice:

1. join-aware HCRD with absolute tolerance `eta + tau`;
2. noise-only HCRD with absolute tolerance `tau`.

Both use zero relative tolerance and the centred minimum-curvature rule.

## Endpoints

For both variants record exact knot recovery, knot F1, symmetric Hausdorff
localisation error divided by lobe width, retained knots, and baseline/detail
errors. For the join-aware variant also record the simultaneous curvature and
sample-noise events and the theorem's joint reconstruction certificate.

Report cellwise Wilson 95% intervals. A code or theorem implementation failure
is declared if either tolerance inside the strict certified region has the
simultaneous curvature event but does not recover the exact knots. Results
outside the strict boundary are descriptive and do not falsify the sufficient
theorem.
