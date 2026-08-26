# C2: real WSD sparse-local-transient confirmation

## Purpose

Test the prospectively stated positioning that the unchanged `hcrd_L8_max`
area-density detector is especially useful for short, low-occupancy local
events, while remaining training-free and directly traceable to signed chord
structures.

This study uses all 86 WSD series in TSB-AD-U that occur in neither the
official 48-series tuning list nor the official 350-series evaluation list.
Every WSD identifier is unique in this population.  No C2 method score is
permitted before `frozen_population.json` is written.

## Label-only primary class

For length (n), anomalous-sample count (M), and longest contiguous anomaly
run (R), the primary stratum is

\[
R/n\le0.005,\qquad M/n\le0.01.
\]

Exactly 71 of 86 unused real WSD series satisfy it.  The thresholds were chosen
after inspecting only label durations and occupancies, but before executing or
inspecting any C2 baseline or HCRD score.  Consequently the score comparison
is prospective, while the exact class threshold should be externally
replicated rather than described as historically preregistered.  All 86 series
form a fixed secondary analysis, and the 15 outside-class series are retained.

## Fixed methods and metrics

The baseline family uses the pinned official TSB-AD wrappers and official
univariate optimal parameters:

- MMPAD, KShapeAD, SAND;
- Sub-PCA and Matrix Profile;
- spectral residual (SR), POLY, and subsequence isolation forest.

`n_job=1` for MMPAD and `n_jobs=1` where accepted change resource scheduling,
not the mathematical score.  `np.Inf=np.inf` restores an alias removed by
NumPy 2 and is needed by the pinned KShapeAD/SAND implementation.  NORMA and
Series2Graph cannot be executed from the official repository because their
provided source archives are password-protected and marked patent-restricted;
they are disclosed as unavailable and are not silently approximated.

The primary endpoint is mean per-series exact VUS-PR on the 71 primary series.
AUC-PR, VUS-ROC, AUC-ROC, the full 86-series population, and the 15-series
outside-class subset are secondary.  Wall-clock time is diagnostic only and
must not support any claim because the host CPU was concurrently loaded.

## Leakage-prevention sequence

1. Freeze file identities/hashes, label-only strata, methods, parameters,
   metrics, inference, runner, and protocol.
2. Execute and cache every baseline on all 86 series.  Failures are saved and
   never converted to missing-row deletion.
3. Among methods successful on all 71 primary series, select the one with the
   highest primary mean VUS-PR.  Require at least four eligible baselines and
   write `comparator_frozen.json`.
4. The runner refuses HCRD execution before that file exists and refuses
   comparator selection if any HCRD metric already exists.
5. Execute the unchanged eight-level maximum-fusion HCRD detector.
6. Analyse once all fixed methods are complete on all 86 series.

## Inference

The single frozen primary comparison reports the paired mean VUS-PR difference,
a 50,000-draw paired percentile-bootstrap 95% interval, paired t-test,
Wilcoxon signed-rank test, exact sign test, wins/ties/losses, and paired
standardized effect.  The t-test targets the mean under its usual sampling
conditions; Wilcoxon and the sign test answer different distributional
questions and are reported as robustness checks.

Comparisons against the complete fixed baseline family are secondary and use
Holm adjustment separately within each stratum.  Spearman associations between
the HCRD-minus-comparator difference and anomaly duration/occupancy are
exploratory effect-modifier analyses.

## Interpretation boundary

Success establishes an independently identified real-data niche among the
tested runnable classical methods.  It does not prove superiority to every
detector, does not make the label-only class equivalent to the theoretical
HCRD-visible chord-lobe class, and does not close the minimax detection problem.
Those bridges remain explicit theory obligations.
