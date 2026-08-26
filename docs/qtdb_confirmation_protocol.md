# Frozen QTDB confirmation protocol R2

This protocol was written after the 25-record exploratory pilot and before any
signal or annotation from the 80 locked records was downloaded or read.  The
split is stored in `data/qtdb/record_split.json` and was created before reading
any waveform values.

## Task and data

- Dataset: PhysioNet QT Database v1.0.0, 105 two-channel 15-minute Holter
  excerpts, 250 Hz in the pilot records.
- Confirmation partition: the 80 records under `confirmation_locked` in the
  SHA-256 split.
- Reference: expert annotator-1 second-pass waveform boundaries (`q1c`).
- Task: QRS onset and offset delineation conditional on the supplied expert QRS
  fiducial.  This is not QRS detection and not a clinical validation.
- All complete ordered QRS onset–fiducial–offset triplets are included.  A
  method failure or unordered output is retained and penalized, not dropped.
- Independent inference unit: record.  Beat errors are averaged within record.
  Subject identifiers across the contributing source databases are not
  reliably available; the design is record-held-out, not patient-held-out.

## Frozen methods

All learned choices below came only from the 25 pilot records.

1. `hcrd_quadratic`: channel 0, quadratic-curvature guide lambda 10 on the unit
   sample grid, HCRD amplitude ratio 0.20.
2. `hcrd_gaussian`: channel 0, Gaussian guide sigma 2 samples, HCRD amplitude
   ratio 0.30 (best Gaussian HCRD pilot candidate).
3. `hcrd_raw`: channel 1, amplitude ratio 0.10 (best raw HCRD pilot candidate).
4. `derivative_threshold`: channel 0, Gaussian sigma 2, derivative ratio 0.05.
5. `official_pu0` and `official_pu1`: QTDB-distributed `ecgpuwave` boundaries,
   matched to the supplied fiducial within 100 ms on their respective channels.

HCRD/derivative search windows are 140 ms before and 180 ms after the supplied
fiducial; the HCRD anchor radius is 45 ms.  No parameter may be changed after a
locked error is read.

## Endpoints and inference

Primary loss per beat is joint boundary absolute error,

`(|predicted onset - reference onset| + |predicted offset - reference offset|)/2`,

in milliseconds.  A failure is assigned 160 ms, half the full 320 ms search
window.  Secondary endpoints are onset absolute error, offset absolute error,
QRS-duration absolute error, and failure rate.

For each method, average beat losses within record.  For quadratic HCRD minus
each of the five comparators, report the paired mean record difference, 95%
paired bootstrap interval (20,000 resamples), exact two-sided sign test, and a
Holm correction across the five primary comparisons.

A task-specific superiority claim is allowed only for a comparator when the
primary-loss interval lies strictly below zero and the Holm-adjusted sign-test
value is below 0.05.  No noninferiority margin is introduced post hoc.  The
`ecgpuwave` comparisons remain valid negative evidence if they favour the
official algorithm.

## Reproducibility

Store per-beat rows, per-record aggregates, comparison statistics, failure
counts, software versions, record list, and SHA-256 download manifest.  The
locked runner must verify that it reads exactly the 80 predeclared records.
