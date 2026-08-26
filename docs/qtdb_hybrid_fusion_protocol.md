# QTDB ecgpuwave + multilevel HCRD hybrid protocol M2

Status: fixed after M1 and before running M2.

Date fixed: 2026-08-24.

M1 showed that four HCRD levels improve the same learner over level 1, but the
mean error remained 0.84 ms above `pu0` with an inconclusive confidence
interval. M2 asks a different, explicitly post-outcome question: does HCRD add
complementary boundary information to the two distributed `ecgpuwave` channel
outputs? This is a development experiment on previously used evaluation
records, not independent confirmation.

## Fixed comparison

Using the unchanged 25-pilot/80-evaluation record split and supplied expert QRS
fiducial:

1. `learned_ecgpuwave`: the fixed M1 LightGBM L1 learner receives onset,
   offset, width, and success from `pu0` and `pu1`;
2. `learned_ecgpuwave_hcrd`: the identical learner receives those eight inputs
   plus all 144 fixed M1 multilevel HCRD coordinates.

No target-derived feature, evaluation-record fitting, hyperparameter search,
or post-run choice of HCRD levels is allowed. Failed `pu` matches are encoded
as missing coordinates plus a zero success flag. Other learner settings,
clipping, sampling-grid rounding, loss, record-level aggregation, bootstrap,
and failure penalty are identical to M1.

The primary contrast is hybrid minus learned `ecgpuwave` fusion. A complementary
HCRD contribution requires a 95% paired record-bootstrap interval below zero.
All comparisons with raw `pu0`/`pu1` are secondary. Even a positive result
requires a new patient/device-held-out dataset because M2 was conceived after
viewing M1 on these 80 records.
