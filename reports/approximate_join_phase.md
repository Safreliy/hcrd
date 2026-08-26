# Approximate-join recovery phase experiment

The frozen experiment evaluated 162,000 independent Gaussian-noise draws over
three parabolic chord-lobe configurations, six normalized join magnitudes, and
nine normalized active-curvature magnitudes.

## Certified region

The strict corollary region is

`gamma / tau > eta / tau + 2`.

It contained 72,000 draws. Both the original noise-only tolerance `tau` and the
conservative join-inactivating tolerance `eta + tau` recovered every declared
knot in every draw. The smallest configuration-cell exact-recovery probability
was 1.000 and its smallest 95% Wilson lower bound was 0.9962. There were zero
violations of the deterministic implication on the simultaneous curvature
event.

## Departure behaviour

The sufficient boundary is conservative. The join-inactivating tolerance
produced a visible transition below the boundary: at `eta/tau=1`, its pooled
exact-recovery probability rose from 0.3357 at `gamma/tau=2.25` to 0.8913 at
2.50, 0.9940 at 2.75, and 1.000 at 3.00.

The original noise-only tolerance was at least as accurate in every evaluated
cell and was often substantially better below the certified boundary. On this
variable-amplitude parabolic subclass, a nonzero join is the smaller-magnitude
side of an adjacent sign transition, so the centred HCRD rule can select it
without first thresholding it inactive. This empirical pattern motivated the
stronger corollary covering both tolerances.

The result does not establish necessity of the sufficient boundary or extend
to arbitrary nonsampled transitions. It establishes the stated approximate
sampled-join guarantee and exposes the practical cost of an unnecessarily high
join-inactivating threshold.

## Reproduction

```bash
python experiments/run_approximate_join_phase.py
python experiments/generate_approximate_join_phase_figure.py
```

The runner writes raw trials locally. The public package includes the cell
aggregates, summary, and provenance hashes under
`results/approximate_join_phase_r1/`.
