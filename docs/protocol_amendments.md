# Protocol amendments and pilot decisions

## Recovery and E2 robustness sensitivities

Three secondary protocols were fixed to test approximate sampled joins,
conditional source-model inference, and qscore implementation choices before
their outcomes were computed.  The approximate-join phase
experiment varies structural join curvature and active curvature on fresh
seeds; its sufficient boundary is inherited from the new corollary.  The E2
sensitivity refits source learners inside paired retention-time-block
resamples and separately recomputes representations after acquisition-file
delete groups.  The qscore sensitivity changes its minimum-point and
across-file aggregation rules while pairing every HCRD comparison with the
identical qscore variant.  These analyses test robustness of the established
mechanism and E2 contrast; they do not retroactively redefine the frozen
primary endpoints.

The complete specifications are
`docs/approximate_join_phase_protocol.md`,
`docs/ms_metrics_e2_refit_sensitivity_protocol.md`, and
`docs/qscore_implementation_sensitivity_protocol.md`.

## E2/E3 — modern decomposition confirmations

The CEEMDAN E2 protocol and the Iterative Filtering E3 protocol were written
before their fresh seeds were generated.  Each used 50 independent signals at
sigma 0, 0.03, and 0.10 and allowed the external method a per-signal oracle
choice among up to four cumulative slow-tail candidates.  E2 used 20 seeded
CEEMDAN noise realizations; E3 used the author-linked
`iterativefiltering==1.0.4` package with defaults.  HCRD met both frozen
superiority criteria in every condition.  These are baseline-recovery claims
for the theorem-matched chord-lobe generator, not IMF-quality claims.

E3 ran under Python 3.12 because the released package excludes Python 3.13;
`pyfftw==0.15.1` was installed explicitly because version 1.0.4 imports it but
does not declare it as a dependency.  Eight outer worker processes handled
independent signals.

## R3 — post-lock modern QTDB audit

After R2 outcomes were known, `docs/qtdb_modern_baseline_protocol.md` froze an
untuned NeuroKit2 DWT pipeline on the same 80 records.  NeuroKit cleaned each
channel, detected its own R peaks, and delineated QRS boundaries.  Its
penalized joint errors were 77.02 and 99.61 ms on channels 0 and 1, versus
22.74 ms for quadratic HCRD.  The comparison is explicitly exploratory and
asymmetric: HCRD is conditioned on the supplied expert R fiducial.  Therefore
R3 cannot be relabelled as confirmatory end-to-end superiority.

## P2/P2R/P3 — sparse representation and process throughput

P2 was frozen before the first dense/sparse comparison and all configurations
reproduced the same knot digest.  However, the once-per-run CPU gate did not
prevent external load from returning between trials; pre-trial load reached
45.8%.  P2 is retained as an exactness diagnostic but no P2 timing is used.

Before replacement measurements, P2R froze a stricter gate of five consecutive
one-second samples, each at most 20%, before **every** trial.  Sparse serial was
11.89x faster than dense materialization on 384 windows; four processes gave a
further 1.25x.  Before inspecting large-batch outcomes, P3 froze a tenfold
3840-window workload under the same gate.  Eight processes then gave 1.96x
end-to-end speedup, including pool startup, serialization, and result objects.
The complete audit and hashes are in `docs/reproducibility_audit.md`.

## P1 — controlled recalculation and batch parallelism

Following concern about background CPU contention, S2 and locked QTDB R2 were
rerun into separate directories.  Every scientific output was byte-identical;
the audit and hashes are recorded in `docs/reproducibility_audit.md`.  Hence CPU
contention affected only wall-clock measurements.

Before the replacement performance run, `docs/parallel_runtime_protocol.md`
froze the workload, load gate, backends, worker counts, repetitions, order
randomization, and exact-output guardrail.  On the tested 16-core/22-thread
machine, eight worker processes gave the best median batch latency (4.163 s
versus 10.279 s serial, 2.47x).  Python threads were slower because the current
knot walk is GIL-bound.  This is a machine-specific scaling characterization,
not a universal runtime claim.

## R2 — QTDB record-held-out QRS delineation

Before waveform inspection, the 105 official QTDB record names were sorted by
a salted SHA-256 digest into 25 pilot and 80 locked records.  The pilot selected
quadratic guidance (`lambda=10`, channel 0, structure ratio 0.20) and fixed all
comparators.  `docs/qtdb_confirmation_protocol.md` was written before locked
signals were downloaded or read.

R2 retained all 2785 complete locked expert QRS triplets and averaged beat
losses within record before inference.  Quadratic HCRD confirmed superiority
over raw HCRD and the frozen derivative detector, did not confirm superiority
over Gaussian-guided HCRD, and was worse than both distributed `ecgpuwave`
single-lead annotations.  The latter result is an explicit negative external
benchmark, not grounds for post-hoc retuning.

## S2 — proximal-guide pilot and frozen morphology confirmation

The initial L1-proximal grid (`results/proximal_pilot`) was exploratory.  It
verified nonexpansiveness of the outer guide numerically but produced poor knot
F1 and large perturbation amplification after hard HCRD.  This negative result
motivated the quadratic-curvature member of the same convex proximal family.

Two exploratory quadratic grids (`results/quadratic_pilot*`) selected
`regularization = 3.0` for a morphology/stability tradeoff.  Before new seeds
were generated, the design, independent unit, endpoints, comparisons, and
interpretation guardrail were frozen in `docs/stable_confirmation_protocol.md`.
The S2 result is stored under `results/stable_confirmation_s2`; repetitions are
averaged within each of 30 latent signals before inference.  S2 supports a
task-specific knot-morphology claim, not global stability of the hard knot map
or universal baseline superiority.

## Version 0.4 — real-data protocol declared before execution

The CWRU leave-one-load-out bearing classification experiment R1 was specified
before downloading or inspecting the selected signals and before computing any
classification result.  Its purpose is an exploratory portability check, not a
confirmatory superiority claim.  The split is by complete recording/load to
avoid the common leakage error of randomly splitting highly overlapping
windows from the same bearing record.

The R1 result showed a ceiling effect: raw features already reached mean
balanced accuracy 0.982, Gaussian, wavelet, EMD, and raw HCRD each reached
1.000, and guided HCRD reached 0.945.  Consequently R1 does not discriminate
among the four tied decompositions and is retained as a negative/neutral
result.

## External comparison E1 — runtime smoke tests

One-trial runs under `results/external_smoke*` were used only to replace a slow
restarted ADMM oracle with an equivalent OSQP solution of the box-constrained
L1-trend dual.  The candidate lambdas, signals, metrics, seeds, and scientific
hypotheses were not selected from those smoke results.  The full E1 protocol
above was frozen before the 50-trial run.

E1 then produced 50/50 centred-HCRD wins against EMD, VMD, L1 trend filtering,
and Gaussian smoothing on the noiseless exact class.  Its noisy exact-class
rows suggested a distinct thresholded-HCRD advantage.  Protocol C2 was added
after observing that descriptive E1 pattern but before generating the fresh C2
seeds; C2 is therefore a confirmatory follow-up rather than a relabelling of E1.

## Amendment 0.2 — after the first 20-trial pilot

The pilot run is retained under `results/synthetic/`.  It revealed two facts:

1. The original generator used random amplitudes and a randomly kinked
   piecewise-affine baseline.  Those kinks can change the curvature transition
   locations, so the generated signal did not satisfy the sufficient conditions
   of the exact chord-lobe recovery theorem.  Mean noiseless HCRD baseline NMSE
   was about 0.08 rather than numerical zero.
2. Raw second-difference signs were destroyed by even moderate white noise.
   Pointwise MAD thresholding reduced the knot count but did not recover the
   baseline reliably.

No confirmatory claim is made from that pilot.  Protocol version 0.2 separates:

- `exact`: an affine baseline with equal-amplitude alternating lobes.  It is the
  confirmatory theorem-matched class and uses a fresh seed family beginning at
  `20260901`;
- `variable`: random-amplitude lobes and a kinked baseline.  It is an
  exploratory near-class test and is never pooled into the exact-class claim.

The noisy comparison adds Gaussian-guided HCRD.  The guide residual is retained
as an explicit component, so reconstruction remains exact.  A fixed guide scale
is evaluated separately from an oracle-tuned guide scale.  This amendment was
written before running the fresh `20260901` confirmatory trials.

## Amendment 0.3 — metric denominator audit

The version 0.2 pilot showed that NMSE normalised by latent-baseline variance is
ill-conditioned when a randomly sampled affine slope is close to zero.  This
produced enormous ratios despite small absolute errors.  Version 0.3 therefore
uses baseline MSE as the primary endpoint and normalises descriptive NMSE by the
centred power of the observed signal.  Numerical differences below `1e-12` are
ties; this prevents round-off from turning exact RDP or thresholded-HCRD recovery
into a false win.  Fresh confirmatory seeds begin at `20260915`.
