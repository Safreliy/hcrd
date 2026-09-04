# E38 post-audit target coverage and common-support width sensitivity

No responses were regenerated. Coverage refers to the full design-identified
transition set. Method-specific medians exclude that method's empty outputs;
the final column restricts both methods to the same nonempty trials.

| signal | design | n | SCI target coverage | PBP target coverage | SCI/PBP empty rates | own-nonempty reduction | both-nonempty reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| cusp | uniform | 500 | 0.985 [0.957, 0.995] | 1.000 [0.981, 1.000] | 0.010/0.000 | 57.9% | 58.1% |
| cusp | uniform | 1000 | 0.965 [0.930, 0.983] | 1.000 [0.981, 1.000] | 0.030/0.000 | 71.3% | 71.5% |
| cusp | beta_4_8 | 500 | 0.950 [0.910, 0.973] | 1.000 [0.981, 1.000] | 0.040/0.000 | 72.2% | 72.1% |
| cusp | beta_4_8 | 1000 | 0.995 [0.972, 0.999] | 1.000 [0.981, 1.000] | 0.005/0.000 | 75.7% | 75.7% |
| onset | uniform | 500 | 0.975 [0.943, 0.989] | 0.995 [0.972, 0.999] | 0.005/0.000 | 26.5% | 26.5% |
| onset | uniform | 1000 | 0.980 [0.950, 0.992] | 1.000 [0.981, 1.000] | 0.000/0.000 | 32.8% | 32.8% |
| onset | beta_4_8 | 500 | 0.980 [0.950, 0.992] | 1.000 [0.981, 1.000] | 0.005/0.000 | 19.4% | 19.4% |
| onset | beta_4_8 | 1000 | 0.985 [0.957, 0.995] | 1.000 [0.981, 1.000] | 0.000/0.000 | 27.3% | 27.3% |
| jump | uniform | 500 | 0.965 [0.930, 0.983] | 1.000 [0.981, 1.000] | 0.035/0.000 | 44.4% | 44.4% |
| jump | uniform | 1000 | 0.980 [0.950, 0.992] | 0.995 [0.972, 0.999] | 0.020/0.005 | 44.4% | 44.4% |
| jump | beta_4_8 | 500 | 0.975 [0.943, 0.989] | 0.995 [0.972, 0.999] | 0.025/0.005 | 40.0% | 33.3% |
| jump | beta_4_8 | 1000 | 0.985 [0.957, 0.995] | 1.000 [0.981, 1.000] | 0.015/0.000 | 50.0% | 50.0% |
| logistic | uniform | 500 | 0.970 [0.936, 0.986] | 1.000 [0.981, 1.000] | 0.000/0.000 | 0.0% | 0.0% |
| logistic | uniform | 1000 | 0.965 [0.930, 0.983] | 1.000 [0.981, 1.000] | 0.000/0.000 | 0.0% | 0.0% |
| logistic | beta_4_8 | 500 | 0.975 [0.943, 0.989] | 1.000 [0.981, 1.000] | 0.000/0.000 | 0.0% | 0.0% |
| logistic | beta_4_8 | 1000 | 0.975 [0.943, 0.989] | 1.000 [0.981, 1.000] | 0.000/0.000 | 0.0% | 0.0% |

Point and full-target coverage agree in all 6,400 method rows. On the
common nonempty subset, the reduction range over the 12 informative cells is
19.4%--75.7%.
