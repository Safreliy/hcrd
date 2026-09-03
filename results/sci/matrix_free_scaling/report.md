# Matrix-free SCI scaling

The benchmark uses the full `(1, 2, 4)` separation family. Times are
hardware-specific. Stored array sizes and dense-matrix counterfactuals
follow directly from the constructed family.

| n | contrasts | build (s) | evaluate (s) | stored (MiB) | dense matrix (GiB) |
|---:|---:|---:|---:|---:|---:|
| 1000 | 5872 | 0.0006 | 0.0015 | 0.05 | 0.04 |
| 10000 | 59831 | 0.0009 | 0.0039 | 0.53 | 4.46 |
| 100000 | 599790 | 0.0042 | 0.0287 | 5.34 | 446.88 |
| 1000000 | 5999750 | 0.0346 | 0.7318 | 53.40 | 44701.62 |

The compact family stores scale metadata and start indices. On a
uniform design, contrast evaluation uses prefix sums. No
contrast-by-observation matrix is created.
