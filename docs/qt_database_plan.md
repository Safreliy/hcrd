# QT Database delineation protocol (R2)

The QT Database contains 105 fifteen-minute, two-channel Holter excerpts and
expert waveform boundaries for at least 30 representative beats per record.
It also distributes `ecgpuwave` automatic boundaries (`pu0`, `pu1`), which
provide a native established comparator. Source: PhysioNet QT Database v1.0.0.

## Record partition

The official `RECORDS` list is partitioned deterministically before reading
signal or annotation values:

- compute SHA-256 of `"hcrd-r2-20260824:" + record_name`;
- sort by the resulting hexadecimal digest;
- use the first 25 records for model selection;
- use the remaining 80 records for evaluation.

The unit of inference is the record after averaging beat-level errors. QTDB
does not expose reliable subject identifiers across all contributing source
databases, so the split is record-held-out rather than patient-held-out.

## Model-selection objective

Expert second-pass annotator-1 files (`q1c`) provide QRS onset, fiducial, and
offset triplets. Candidate HCRD representations include raw, fixed Gaussian,
and fixed quadratic-curvature guides. The official single-lead `pu0`/`pu1`
boundaries and a derivative-threshold method are comparators.

The grid contains channels 0 and 1 separately, HCRD amplitude ratios
`{0.10, 0.20, 0.30}`, Gaussian guide sigma `{1, 2, 4}` samples, and quadratic
guide lambda `{1, 10, 100, 1000}`. The search window is 140 ms before and
180 ms after the supplied expert QRS fiducial, with a 45 ms anchor radius. The
derivative comparator uses the same window, Gaussian sigma `{1, 2, 4}`, and
threshold ratios `{0.05, 0.10, 0.20, 0.30}`.

Within each method family, the selected configuration minimizes the mean over
model-selection records of joint boundary absolute error
`(|onset error| + |offset error|)/2`. Failed or unordered predictions receive
160 ms for each missing boundary. Ties within 0.25 ms favour the smaller guide
parameter and then the larger amplitude ratio.

## Endpoints

- absolute QRS-onset error in milliseconds;
- absolute QRS-offset error in milliseconds;
- QRS-duration absolute error in milliseconds;
- fraction of beats without ordered boundaries;
- per-record paired differences and bootstrap intervals.

Clinical interpretation is outside scope; this is an algorithmic waveform-
delineation benchmark.
