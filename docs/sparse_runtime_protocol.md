# Sparse hierarchy runtime protocol P2 (frozen before execution)

## Question

Does the knot-only implementation realize the linear-work consequence of the
centred-rule halving recurrence in practice, while preserving the exact knot
hierarchy?  Once dense materialization is removed, does outer process
parallelism still improve the fixed CWRU batch?

## Frozen workload and environment

- The same 384 deterministic, normalized 2048-sample CWRU windows as P1.
- Full centred HCRD hierarchy, default tolerances and no level cap.
- Five repetitions per configuration, randomized within repetition with seed
  `20260825`.
- Numerical-library thread counts fixed to one before NumPy import.
- Start only after five consecutive one-second whole-system CPU samples average
  no more than 20%; record CPU load before each trial.
- Timed calls include pool construction, serialization, result collection, and
  object allocation.

## Frozen configurations

1. backwards-compatible dense serial decomposition;
2. sparse serial decomposition;
3. sparse threads with 4 and 8 workers;
4. sparse processes with 2, 4, 8, and 16 workers.

The dense/sparse timing ratio is an output-representation comparison: dense
mode additionally materializes every length-`n` baseline, detail, input copy,
and structure table.  It is not attributed solely to the knot walk.

## Exactness guardrails and reporting

- Before timing, every sparse knot set must equal its dense counterpart for
  every window and level.
- A canonical SHA-256 over all ordered knot sets must be identical for every
  timed configuration.
- The stored sparse knot count must satisfy
  `sum_j |K_j| <= 2(n-1) + depth` for every centred hierarchy.
- Report median, interquartile range, signals/s, sparse speedup versus dense
  serial, and parallel speedup/efficiency versus sparse serial.
- Process/thread conclusions are restricted to this hardware and batch size.

Outputs are written to `results/sparse_runtime_p2`.
