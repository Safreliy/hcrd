# E36 high-precision SCI coverage audit

All frozen checks passed: **False**.

Post-audit correction: a single trial with equal lower and upper exclusion
boundaries is now classified as empty. Coverage and every displayed value are
unchanged; the affected cell's empty rate changes from 0.0058 to 0.0060.

| signal | design | n | coverage | 95% MC interval | median width |
|---|---|---:|---:|---:|---:|
| cusp | uniform | 500 | 0.9784 | [0.9740, 0.9821] | 0.1737 |
| cusp | uniform | 1000 | 0.9804 | [0.9762, 0.9839] | 0.1109 |
| cusp | beta_4_8 | 500 | 0.9798 | [0.9755, 0.9833] | 0.0877 |
| cusp | beta_4_8 | 1000 | 0.9772 | [0.9727, 0.9810] | 0.0712 |
| onset | uniform | 500 | 0.9772 | [0.9727, 0.9810] | 0.7325 |
| onset | uniform | 1000 | 0.9788 | [0.9744, 0.9824] | 0.6703 |
| onset | beta_4_8 | 500 | 0.9778 | [0.9733, 0.9815] | 0.8149 |
| onset | beta_4_8 | 1000 | 0.9774 | [0.9729, 0.9812] | 0.7611 |
| jump | uniform | 500 | 0.9798 | [0.9755, 0.9833] | 0.0100 |
| jump | uniform | 1000 | 0.9786 | [0.9742, 0.9823] | 0.0060 |
| jump | beta_4_8 | 500 | 0.9784 | [0.9740, 0.9821] | 0.0034 |
| jump | beta_4_8 | 1000 | 0.9770 | [0.9725, 0.9808] | 0.0020 |
| logistic | uniform | 500 | 0.9790 | [0.9746, 0.9826] | 1.0000 |
| logistic | uniform | 1000 | 0.9776 | [0.9731, 0.9813] | 0.9990 |
| logistic | beta_4_8 | 500 | 0.9798 | [0.9755, 0.9833] | 1.0000 |
| logistic | beta_4_8 | 1000 | 0.9784 | [0.9740, 0.9821] | 1.0000 |

These simulations measure Monte Carlo behaviour of the code. The
finite-sample guarantee comes from the theorem, not from this table.
