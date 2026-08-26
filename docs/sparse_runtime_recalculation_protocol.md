# Load-gated sparse runtime recalculation P2R (frozen before execution)

## Motivation and status

P2 began behind one load gate, but unrelated CPU work appeared during later
trials (recorded pre-trial load reached 45.8%).  P2 is therefore retained as a
diagnostic exactness run and is not used for timing claims.  P2R is a fresh
replacement protocol frozen before looking at its outcomes.

## Workload and controls

- The same 384 deterministic, normalized 2048-sample CWRU windows as P1/P2.
- Full centred HCRD hierarchy, default tolerances and no level cap.
- Five repetitions per configuration, randomized within repetition with seed
  `20260826`.
- Numerical-library thread counts fixed to one before NumPy import.
- **Before every timed trial**, obtain five consecutive one-second whole-system
  CPU measurements, each no greater than 20%.  Abort if this condition is not
  reached within 300 seconds.
- Timed calls include pool construction, serialization, result collection,
  hierarchy allocation, and (for dense mode) dense materialization.

## Frozen configurations

1. backwards-compatible dense serial decomposition;
2. sparse serial decomposition;
3. sparse process decomposition with 2, 4, and 8 workers.

The dense/sparse ratio measures different output representations: dense mode
materializes every length-`n` baseline and detail, while sparse mode stores
only ordered knot indices.  Process scaling is evaluated against sparse
serial, and all conclusions remain specific to this machine and batch size.

## Exactness guardrails and reporting

- Dense and sparse reference knot sets must agree exactly at every level.
- Every timed output must reproduce one canonical SHA-256 of all knot sets.
- Every sparse hierarchy must satisfy
  `sum_j |K_j| <= 2(n-1) + depth`.
- Report all trials, the five pre-trial load samples, medians, interquartile
  ranges, signals/s, representation speedup, and process efficiency.

Outputs are written to `results/sparse_runtime_p2r`.
