# Parallel runtime protocol P1 (frozen before execution)

## Question

Does process-level parallelism over independent signal windows reduce the wall
time of the current HCRD feature implementation, and do all execution backends
produce exactly the same features?

## Fixed workload

- The 384 nonoverlapping CWRU windows already used in study R1 (16 official
  records, 24 deterministic windows per record, 2048 samples per window).
- The frozen five-component `hcrd` representation and its 50-dimensional
  feature vector.  File loading, window selection, and model fitting are outside
  the timed region.
- Execution modes: serial; four and eight Python threads; and two, four, eight,
  and sixteen worker processes.
- Five measured repetitions per mode.  Mode order is independently permuted in
  every repetition with seed 20260824.  A short untimed warm-up precedes them.
- Process-pool construction, input serialization, output serialization, and
  shutdown are inside the timed region; this is end-to-end batch latency rather
  than steady-state executor throughput.
- BLAS/OpenMP library thread counts are fixed to one before NumPy is imported,
  preventing nested parallelism from being attributed to HCRD.

## Load control and reporting

Execution begins only after five consecutive one-second whole-system CPU-load
samples average at most 20%.  If this condition is not met within 180 seconds,
the benchmark aborts instead of silently accepting a contaminated run.  The
CPU model, physical/logical core counts, affinity, Python/NumPy versions,
per-trial pre-run CPU load, median, interquartile range, throughput, and speedup
relative to the serial median are recorded.

The primary runtime statistic is median wall time over five repetitions.  No
formal stochastic significance claim is attached to one machine.  The study is
a reproducible scaling characterization, not a hardware-independent complexity
theorem.

## Correctness guardrail

Every timed feature matrix must be bitwise identical to the serial reference.
The reference is also compared with the previously saved R1 HCRD feature
matrix.  A mismatch aborts the benchmark.  Thus background CPU activity may
affect latency but cannot change the quality endpoints or numerical outputs.

## Interpretation boundary

The levels of one HCRD hierarchy are data-dependent and sequential.  The
parallel claim is restricted to independent signals/windows (and, in a larger
system, independent channels or records).  Since the current knot walk is
pure Python, process workers are expected to scale better than threads under
CPython's global interpreter lock; this expectation is frozen before seeing
the result.
