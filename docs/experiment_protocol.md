# Synthetic and benchmark experiment protocol

Version: 0.5 — pilot amendments are recorded in `protocol_amendments.md`.

## Scope

The project does **not** test or claim universal superiority over signal
decomposition methods.  The primary task class is:

> recovery of an affine chord baseline and alternating locally
> convex/concave transient lobes whose endpoints lie on that baseline.

This class is directly matched to the geometry of HCRD and occurs as an
idealised model of impulses, half-waves, and morphology-preserving compression.

## Primary synthetic hypothesis H1

On the noiseless alternating-chord-lobe class, the first centred-HCRD baseline
has baseline NMSE below every tested generic smoother.

- Primary metric: baseline MSE.  Descriptive normalised values use centred
  observed-signal power, never latent-baseline variance, because an almost
  constant baseline makes the latter denominator arbitrarily small.
- Secondary metrics: detail NMSE and knot F1 with one-sample tolerance.
- Comparator hyperparameters are selected by an oracle using the known ground
  truth separately for every signal.  This deliberately favours comparators.
- Success criterion against a comparator: paired mean MSE difference
  `HCRD - comparator < 0`, its 95% paired-bootstrap confidence interval is
  entirely below zero, and the two-sided exact sign-test has `p < 0.05` after
  Holm correction.
- Differences below `1e-12` are numerical ties.  HCRD is not declared superior
  to a method that is exact up to this tolerance.

## Robustness hypothesis H2

For additive iid Gaussian noise, robust HCRD has a higher median pairwise knot
Jaccard similarity across repeated noise realisations than raw HCRD.

- Primary metric: first-level knot Jaccard similarity, tolerance one sample.
- Secondary metrics: variance of the first baseline and number of knots.
- Noise levels: sigma in `{0.01, 0.03, 0.05, 0.10}` relative to lobe amplitudes
  of approximately one.
- At least 30 independent latent signals and 10 noise realisations per signal.

## Out-of-class falsification suite H3

The following cases are reported even when HCRD loses:

- quadratic and cubic trends;
- two close sinusoids and crossing chirps;
- weak high-frequency oscillations under strong background curvature;
- endpoint outliers;
- irregular sampling;
- coloured and impulsive noise.

These tests delimit applicability.  They are not pooled into the primary claim.

## Baselines

The dependency-light first pass includes affine least squares, moving average,
Gaussian smoothing, Fourier low-pass, and equal-budget RDP approximation.  The
full comparison will add EMD/CEEMDAN, ITD, VMD, iterative filtering, L1 trend
filtering, and LULU/DPT where a reproducible implementation is available.

## Reproducibility

- Confirmatory version-0.3 seeds begin at `20260915`.
- Per-trial observations are retained, not only aggregates.
- Generated files include a UTC timestamp, git commit when available, Python
  and NumPy versions, and exact command-line arguments.
- Failed runs remain visible in logs and are not silently deleted.

## Exploratory real-data experiment R1

R1 uses the Case Western Reserve University 12 kHz drive-end bearing data at
four motor loads.  The classes are normal, 0.007-inch inner-race fault,
0.007-inch ball fault, and 0.007-inch outer-race fault at the six-o'clock
position.

- Primary descriptive metric: leave-one-load-out balanced accuracy.
- Secondary metrics: macro F1, per-class recall, fit/predict time, and feature
  extraction time.
- A fold trains on three complete load recordings per class and tests on the
  fourth.  Windows from the same recording can never cross the split.
- Each recording contributes 24 deterministic, evenly spaced, nonoverlapping
  windows of 2048 samples.  Every window is median-centred and RMS-scaled.
- The same multinomial logistic-regression pipeline and the same component
  feature map are used for all representations.
- Representations: undecomposed signal, four-level Gaussian Laplacian pyramid,
  four-level Daubechies-4 wavelet decomposition, EMD, raw HCRD, and
  pilot-calibrated guided HCRD.  Each is represented by five components; absent
  components are zero padded.
- Only four independent load conditions are available. Window-level confidence
  intervals are therefore not interpreted as if the windows were independent
  experiments, and R1 is treated as descriptive.

## External synthetic comparison E1

E1 repeats the exact and variable chord-lobe suites with 50 fresh trials per
noise condition and adds EMD, VMD, and second-order L1 trend filtering.  Its
configuration was frozen after runtime-only smoke tests of one trial.

- Primary condition: noiseless exact chord-lobe class.
- Reference: centred HCRD first baseline.
- External comparators: final EMD residue, lowest-centre-frequency VMD mode
  with `K=5, alpha=2000`, and L1 trend filtering oracle-selected from
  `lambda in {1,3,10,30,100,300}` independently on every signal.
- The Gaussian oracle remains as a common calibration comparator.
- Success against one comparator requires a negative paired mean MSE
  difference, a paired-bootstrap 95% interval below zero, and a Holm-adjusted
  exact sign-test `p < 0.05`; differences below `1e-12` are ties.
- All noisy and variable-class results are descriptive boundaries, not pooled
  into the primary superiority statement.

## Fresh-seed noisy confirmation C2

E1 descriptively suggested that MAD-thresholded HCRD can recover the affine
baseline of the exact chord-lobe class under Gaussian noise better than the
external methods.  C2 was frozen after E1 and before inspecting any C2 seed.

- Two separate confirmatory experiments use `sigma=0.03` and `sigma=0.10`.
- Each experiment uses 100 fresh trials beginning at seed `20261101`; seed
  families are separated by the noise offset already encoded by the runner.
- Reference: HCRD with plug-in MAD curvature threshold and fixed `z=3.5`.
- Comparators: EMD residue, VMD low mode, per-signal oracle L1 trend filter,
  and per-signal oracle Gaussian smoother.  Centred and adaptive HCRD are
  retained as internal diagnostics.
- The same MSE, numerical-tie, paired-bootstrap, exact sign-test, and Holm
  criteria as H1/E1 apply separately at each noise level.
- This confirms a distribution-and-task-specific empirical statement.  The
  family-wise theorem in the paper assumes externally known sigma and does not
  by itself certify the plug-in MAD estimator.
