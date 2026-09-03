# E34: honest unknown-scale confirmation

Fresh-seed confirmation using a finite-sample upper Gaussian noise-scale bound.

| signal | design | n | coverage | median width | scale coverage | mean scale ratio |
|---|---:|---:|---:|---:|---:|---:|
| affine | beta_4_8 | 1000 | 1.000 | 1.0000 | 0.990 | 1.081 |
| affine | beta_4_8 | 500 | 1.000 | 1.0000 | 0.995 | 1.116 |
| affine | uniform | 1000 | 1.000 | 1.0000 | 0.995 | 1.078 |
| affine | uniform | 500 | 0.990 | 1.0000 | 0.995 | 1.111 |
| concave | beta_4_8 | 1000 | 1.000 | 1.0000 | 0.985 | 1.075 |
| concave | beta_4_8 | 500 | 0.990 | 1.0000 | 0.995 | 1.117 |
| concave | uniform | 1000 | 0.995 | 0.5754 | 0.985 | 1.078 |
| concave | uniform | 500 | 1.000 | 0.5749 | 0.985 | 1.110 |
| convex | beta_4_8 | 1000 | 0.995 | 1.0000 | 0.980 | 1.075 |
| convex | beta_4_8 | 500 | 0.990 | 1.0000 | 0.995 | 1.115 |
| convex | uniform | 1000 | 0.995 | 0.5754 | 0.995 | 1.076 |
| convex | uniform | 500 | 0.990 | 0.5749 | 0.985 | 1.121 |
| f1_cusp | beta_4_8 | 1000 | 1.000 | 0.0768 | 0.990 | 1.082 |
| f1_cusp | beta_4_8 | 500 | 0.995 | 0.1115 | 1.000 | 1.116 |
| f1_cusp | uniform | 1000 | 1.000 | 0.1508 | 1.000 | 1.078 |
| f1_cusp | uniform | 500 | 0.995 | 0.2056 | 0.985 | 1.116 |
| f2_onset | beta_4_8 | 1000 | 0.995 | 0.7863 | 0.990 | 1.076 |
| f2_onset | beta_4_8 | 500 | 1.000 | 0.8149 | 0.995 | 1.114 |
| f2_onset | uniform | 1000 | 0.995 | 0.7023 | 0.995 | 1.079 |
| f2_onset | uniform | 500 | 1.000 | 0.7645 | 0.990 | 1.118 |
| f3_jump | beta_4_8 | 1000 | 1.000 | 0.0024 | 0.995 | 1.079 |
| f3_jump | beta_4_8 | 500 | 1.000 | 0.0088 | 1.000 | 1.221 |
| f3_jump | uniform | 1000 | 1.000 | 0.0070 | 0.995 | 1.082 |
| f3_jump | uniform | 500 | 1.000 | 0.0140 | 0.995 | 1.116 |
| f4_logistic | beta_4_8 | 1000 | 0.995 | 1.0000 | 0.975 | 1.079 |
| f4_logistic | beta_4_8 | 500 | 1.000 | 1.0000 | 0.990 | 1.112 |
| f4_logistic | uniform | 1000 | 0.995 | 1.0000 | 0.990 | 1.077 |
| f4_logistic | uniform | 500 | 1.000 | 1.0000 | 0.995 | 1.118 |

## Frozen gates

- PASS — `coverage_at_least_0_93_every_cell`
- PASS — `scale_coverage_at_least_0_95_every_cell`
- PASS — `zero_unexplained_empty_sets`
- PASS — `large_n_width_limits`
- PASS — `mean_informative_width_inflation_below_1_5`
- PASS — `weak_logistic_cells_retained`

Mean informative width inflation versus known sigma: `1.117`.

The guarantee remains specific to independent homoskedastic Gaussian errors. Projection lack of fit can widen intervals but cannot invalidate coverage.
