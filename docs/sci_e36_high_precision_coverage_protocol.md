# E36 protocol: high-precision SCI coverage audit

**Frozen target:** known-scale SCI only. The expensive E33 comparison with
`ShapeChange` remains unchanged.

## Purpose

E33 used 200 repetitions per cell. That was enough to detect large failures,
but it was not enough to estimate coverage near 0.95 precisely. E36 repeats
the SCI part with 5,000 fresh responses per cell.

## Fixed design

- signals: the four published Feng et al. examples (`f1` cusp, `f2` onset,
  `f3` jump, and `f4` logistic);
- observation designs: uniform and `Beta(4,8)` quantile grids;
- sample sizes: 500 and 1,000;
- noise: independent Gaussian with known standard deviation 0.1;
- confidence level: 95%;
- repetitions: 5,000 in each of 16 cells, 80,000 responses in total;
- random seed: 20262111;
- contrast separations: 1, 2 and 4 times the block size;
- implementation: the matrix-free `shapecontrast` namespace.

The Monte Carlo standard error at coverage 0.95 is about 0.0031, compared with
about 0.015 in E33. Every reported coverage value is accompanied by a 95%
Wilson interval.

## Frozen checks

1. Every cell is retained, including the weak logistic cells.
2. The empirical SCI coverage is at least 0.94 in every cell.
3. The lower end of the Wilson interval is reported rather than hidden.
4. Simultaneous contrast coverage is at least 0.94 in every cell.
5. No confidence set is empty while still excluding no true transition.

The 0.94 thresholds are implementation alarms, not replacements for the
coverage theorem. Simulation cannot prove finite-sample validity.

## Outputs

- `frozen_config.json`: configuration and pre-run code hashes;
- `trial_scores.csv`: one row per simulated response;
- `summary.csv`: cellwise coverage, Wilson intervals, width and diagnostics;
- `manifest.json`: output hashes, environment and frozen decisions;
- `report.md`: short plain-language summary.
