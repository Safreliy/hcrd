# Post-audit uncertainty summary for E38r1

The 200 frozen responses per cell were not rerun. Coverage intervals are
95% Wilson intervals. Width-reduction intervals are paired percentile
bootstrap intervals with 20,000 resamples and seed
20260904. Empty sets count as noncoverage and are excluded from
the method-specific median-width calculation, matching the frozen summary.
These intervals are descriptive post-audit analyses, not pre-specified gates.

| Signal | Design | n | SCI coverage (95% CI) | PBP coverage (95% CI) | Median-width reduction (95% CI) |
|---|---|---:|---:|---:|---:|
| Cusp | uniform | 500 | 0.985 [0.957, 0.995] | 1.000 [0.981, 1.000] | 57.9% [56.7%, 59.5%] |
| Cusp | uniform | 1000 | 0.965 [0.930, 0.983] | 1.000 [0.981, 1.000] | 71.3% [69.0%, 72.0%] |
| Cusp | beta_4_8 | 500 | 0.950 [0.910, 0.973] | 1.000 [0.981, 1.000] | 72.2% [67.4%, 72.9%] |
| Cusp | beta_4_8 | 1000 | 0.995 [0.972, 0.999] | 1.000 [0.981, 1.000] | 75.7% [75.1%, 76.7%] |
| Onset | uniform | 500 | 0.975 [0.943, 0.989] | 0.995 [0.972, 0.999] | 26.5% [26.5%, 29.7%] |
| Onset | uniform | 1000 | 0.980 [0.950, 0.992] | 1.000 [0.981, 1.000] | 32.8% [29.6%, 32.8%] |
| Onset | beta_4_8 | 500 | 0.980 [0.950, 0.992] | 1.000 [0.981, 1.000] | 19.4% [19.4%, 23.7%] |
| Onset | beta_4_8 | 1000 | 0.985 [0.957, 0.995] | 1.000 [0.981, 1.000] | 27.3% [26.7%, 27.3%] |
| Jump | uniform | 500 | 0.965 [0.930, 0.983] | 1.000 [0.981, 1.000] | 44.4% [33.3%, 47.4%] |
| Jump | uniform | 1000 | 0.980 [0.950, 0.992] | 0.995 [0.972, 0.999] | 44.4% [33.3%, 50.0%] |
| Jump | beta_4_8 | 500 | 0.975 [0.943, 0.989] | 0.995 [0.972, 0.999] | 40.0% [33.3%, 45.5%] |
| Jump | beta_4_8 | 1000 | 0.985 [0.957, 0.995] | 1.000 [0.981, 1.000] | 50.0% [40.0%, 50.0%] |
| Logistic | uniform | 500 | 0.970 [0.936, 0.986] | 1.000 [0.981, 1.000] | 0.0% [0.0%, 0.0%] |
| Logistic | uniform | 1000 | 0.965 [0.930, 0.983] | 1.000 [0.981, 1.000] | 0.0% [0.0%, 0.0%] |
| Logistic | beta_4_8 | 500 | 0.975 [0.943, 0.989] | 1.000 [0.981, 1.000] | 0.0% [0.0%, 0.0%] |
| Logistic | beta_4_8 | 1000 | 0.975 [0.943, 0.989] | 1.000 [0.981, 1.000] | 0.0% [0.0%, 0.0%] |
