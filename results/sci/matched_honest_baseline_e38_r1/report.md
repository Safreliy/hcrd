# E38r1 SCI versus a conservative pointwise-band baseline

All frozen checks passed: **True**.

PBP is our conservative discrete split relaxation, not an exact projection onto
the SCI function class and not an official implementation of Davies et al. This
wording was corrected after external audit; the frozen trial data are unchanged.

Coverage intervals and paired bootstrap intervals for width reduction are in
`uncertainty_report.md`.

| signal | design | n | SCI cov. | PBP cov. | SCI width | PBP width | reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| cusp | uniform | 500 | 0.985 | 1.000 | 0.1737 | 0.4122 | 57.9% |
| cusp | uniform | 1000 | 0.965 | 1.000 | 0.1109 | 0.3861 | 71.3% |
| cusp | beta_4_8 | 500 | 0.950 | 1.000 | 0.0877 | 0.3151 | 72.2% |
| cusp | beta_4_8 | 1000 | 0.995 | 1.000 | 0.0712 | 0.2934 | 75.7% |
| onset | uniform | 500 | 0.975 | 0.995 | 0.7325 | 0.9960 | 26.5% |
| onset | uniform | 1000 | 0.980 | 1.000 | 0.6703 | 0.9980 | 32.8% |
| onset | beta_4_8 | 500 | 0.980 | 1.000 | 0.5460 | 0.6776 | 19.4% |
| onset | beta_4_8 | 1000 | 0.985 | 1.000 | 0.5170 | 0.7116 | 27.3% |
| jump | uniform | 500 | 0.965 | 1.000 | 0.0100 | 0.0180 | 44.4% |
| jump | uniform | 1000 | 0.980 | 0.995 | 0.0050 | 0.0090 | 44.4% |
| jump | beta_4_8 | 500 | 0.975 | 0.995 | 0.0041 | 0.0068 | 40.0% |
| jump | beta_4_8 | 1000 | 0.985 | 1.000 | 0.0017 | 0.0034 | 50.0% |
| logistic | uniform | 500 | 0.970 | 1.000 | 0.9960 | 0.9960 | 0.0% |
| logistic | uniform | 1000 | 0.965 | 1.000 | 0.9980 | 0.9980 | 0.0% |
| logistic | beta_4_8 | 500 | 0.975 | 1.000 | 0.6776 | 0.6776 | 0.0% |
| logistic | beta_4_8 | 1000 | 0.975 | 1.000 | 0.7116 | 0.7116 | 0.0% |
