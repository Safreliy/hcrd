# QTDB modern-baseline extension R3 (frozen before execution)

## Status and question

R3 is a post-lock exploratory extension: the 80 R2 records and HCRD outcomes
were already known before this baseline was added.  It cannot create a new
confirmatory superiority claim.  Its purpose is to test whether the R2 method
is competitive with a current, reproducible wavelet delineation pipeline, not
only the historical distributed `ecgpuwave` annotations.

## Frozen pipeline

- Library: NeuroKit2 0.2.13, recorded at runtime.
- Each complete QTDB waveform is cleaned with
  `neurokit2.ecg_clean(..., method="neurokit")` at the record's native sampling
  rate.
- R peaks are detected from the cleaned signal with
  `neurokit2.ecg_peaks(..., method="neurokit", correct_artifacts=False)`.
- QRS onsets and offsets are delineated with
  `neurokit2.ecg_delineate(..., method="dwt", check=False)` using those detected
  peaks.  All remaining arguments keep library defaults.
- Channels 0 and 1 are evaluated separately and both are reported.  Neither
  channel is selected from its result.
- For each expert QRS complex, the nearest detected R peak must be within 45 ms
  of the expert fiducial.  Otherwise the event is a failure.  Missing,
  nonfinite, reversed, or unmatched DWT boundaries are also failures.
- No QTDB-specific thresholds, filters, resampling choices, or per-record
  repairs may be introduced after execution.

Unlike the frozen HCRD boundary rule, which was conditional on the supplied
expert QRS fiducial, this baseline detects R peaks itself.  It therefore solves
a strictly broader task; this disadvantage to the comparator must be retained
in interpretation.

## Endpoint and inference

The R2 endpoint is reused exactly: mean absolute onset/offset error per beat,
with failed events assigned 160 ms, then averaged within record.  The unit of
inference remains the record.  Quadratic HCRD minus each NeuroKit channel is
summarized by a 20,000-resample paired record bootstrap interval and an exact
sign test with Holm correction over the two channel comparisons.  These values
are descriptive/post-lock regardless of interval or p-value.

## Parallel execution and reproducibility

Complete records are independent.  They are processed by an order-preserving
eight-process pool with compiled-library thread counts fixed to one.  Parallel
execution may change runtime, not outputs.  The locked download manifest is
verified before waveform processing; library versions, worker count, failures,
trial rows, record summaries, and hashes are stored under
`results/qtdb_modern_r3`.
