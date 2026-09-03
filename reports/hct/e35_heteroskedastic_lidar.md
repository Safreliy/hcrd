# E35: unknown heteroskedastic HCT confirmation

## Frozen simulation result

Minimum HCT coverage across 48 cells: `0.995`.
Minimum variance-envelope coverage: `0.995`.

### Frozen gates

- PASS — `all_48_unknown_cells_retained`
- PASS — `coverage_at_least_0_93_every_cell`
- PASS — `scale_coverage_at_least_0_93_every_cell`
- PASS — `zero_unexplained_empty_sets`
- PASS — `weak_signal_cells_retained`

## LIDAR sensitivity illustration

Current `Sshaped` point estimate: `588.0 m`.

| kappa | HCT interval (m) | width (m) | upper noise scale |
|---:|---:|---:|---:|
| 1.0 | [510.0, 664.0] | 154.0 | 0.1063 |
| 1.5 | [486.0, 720.0] | 234.0 | 0.1416 |
| 2.0 | [438.0, 720.0] | 282.0 | 0.1778 |
| 3.0 | [390.0, 720.0] | 330.0 | 0.2617 |
| 4.0 | [390.0, 720.0] | 330.0 | 0.3818 |

Nominal ShapeChange residual-bootstrap estimate: `606.0 m`, interval `[592.6, 612.8] m`.

The HCT rows are finite-sample confidence statements only when errors are independent Gaussian and the displayed kappa is a valid upper bound. The ShapeChange interval is descriptive here because its iid residual bootstrap does not model the visible heteroskedasticity.
