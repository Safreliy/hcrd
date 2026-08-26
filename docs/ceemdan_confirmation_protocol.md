# CEEMDAN external comparison E2 (frozen before execution)

## Question and hypotheses

E2 tests whether the previously confirmed task-specific HCRD advantage survives
comparison with a noise-assisted modern EMD variant.  Three paired hypotheses
are fixed:

1. at Gaussian noise sigma 0, centred HCRD has lower latent-baseline MSE than
   an oracle slow-tail CEEMDAN baseline;
2. at sigma 0.03, MAD-thresholded HCRD has lower MSE than that oracle;
3. at sigma 0.10, MAD-thresholded HCRD has lower MSE than that oracle.

The theorem-matched equal-amplitude affine chord-lobe generator is reused with
50 fresh latent/noise seeds per condition beginning at 20261201.  No result from
these seeds has been inspected before this protocol.

## Frozen CEEMDAN configuration

- `EMD-signal` / `PyEMD.CEEMDAN`, package version recorded at runtime.
- 20 noise realizations, `epsilon=0.005`, `noise_scale=1.0`, and all other
  algorithmic/stopping parameters at package defaults.
- CEEMDAN internal parallelism disabled; each decomposition receives a fixed
  noise seed derived from the signal seed.
- Complete independent signals are evaluated by an eight-process outer pool.
- The numerical residue is retained.  Candidate baselines are the residue plus
  the cumulative one, two, three, or four slowest returned CEEMDAN components
  (or all available components when fewer exist).
- The candidate with lowest MSE to the known latent baseline is selected
  separately for each signal.  This oracle deliberately favours CEEMDAN and is
  labelled as such; the final-component-only result is also retained.

## Endpoint and decision rule

Baseline MSE is primary.  Differences below `1e-12` are ties.  Each of the
three paired mean differences receives a 20,000-resample bootstrap interval and
an exact sign test.  Holm correction is applied over all three hypotheses.
HCRD superiority is supported only when the upper confidence limit is below
zero and the adjusted sign-test value is below 0.05.

CEEMDAN exact reconstruction error, selected tail length, package versions,
seeds, trial rows, and elapsed time are retained under
`results/ceemdan_confirmation_e2`.  Process scheduling must not change seeded
outputs; trial rows are sorted before writing.
