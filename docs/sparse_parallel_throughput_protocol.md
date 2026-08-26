# Sparse process-throughput protocol P3 (frozen before execution)

## Question

Does exact independent-signal process parallelism provide useful scaling once
the sparse knot-only workload is large enough that Windows pool startup and
serialization no longer dominate the computation?

## Frozen workload and controls

- Repeat the fixed list of 384 normalized 2048-sample CWRU windows ten times,
  preserving order, for a 3840-signal throughput batch.
- Full centred sparse HCRD hierarchy, default tolerances and no level cap.
- Three repetitions per configuration, randomized within repetition with seed
  `20260827`.
- Numerical-library thread counts fixed to one before NumPy import.
- Before every trial, require five consecutive one-second whole-system CPU
  samples, each no greater than 20%; timeout after 300 seconds.
- Pool construction, input serialization, output collection, and hierarchy
  allocation are all included.

## Frozen configurations and guardrails

- sparse serial;
- sparse processes with 2, 4, and 8 workers.

Every timed configuration must reproduce the same canonical SHA-256 over all
ordered knot sets as sparse serial.  Every hierarchy must obey
`sum_j |K_j| <= 2(n-1) + depth`.  Report all trials, median and interquartile
latency, throughput, speedup, and parallel efficiency.  Conclusions are
specific to this hardware, Python implementation, and batch size.

Outputs are written to `results/sparse_parallel_p3`.
