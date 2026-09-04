# E36 post-audit coverage of the full identified target

No responses were regenerated. The design-identified transition set was
computed once per cell and compared with the 80,000 saved SCI intervals.
Empty outputs count as noncoverage.

| signal | design | n | identified target | point coverage | target coverage (95% Wilson CI) | changed trials |
|---|---|---:|---:|---:|---:|---:|
| cusp | uniform | 500 | [0.299103, 0.301696] | 0.9784 | 0.9784 [0.9740, 0.9821] | 0 |
| cusp | uniform | 1000 | [0.299551, 0.300849] | 0.9804 | 0.9804 [0.9762, 0.9839] | 0 |
| cusp | beta_4_8 | 500 | [0.299491, 0.300419] | 0.9798 | 0.9798 [0.9755, 0.9833] | 0 |
| cusp | beta_4_8 | 1000 | [0.299673, 0.300225] | 0.9772 | 0.9772 [0.9727, 0.9810] | 0 |
| onset | uniform | 500 | [0.300000, 0.304618] | 0.9772 | 0.9770 [0.9725, 0.9808] | 1 |
| onset | uniform | 1000 | [0.300000, 0.302311] | 0.9788 | 0.9786 [0.9742, 0.9823] | 1 |
| onset | beta_4_8 | 500 | [0.300000, 0.301350] | 0.9778 | 0.9778 [0.9733, 0.9815] | 0 |
| onset | beta_4_8 | 1000 | [0.300000, 0.300608] | 0.9774 | 0.9774 [0.9729, 0.9812] | 0 |
| jump | uniform | 500 | [0.299401, 0.301397] | 0.9798 | 0.9798 [0.9755, 0.9833] | 0 |
| jump | uniform | 1000 | [0.299700, 0.300699] | 0.9786 | 0.9786 [0.9742, 0.9823] | 0 |
| jump | beta_4_8 | 500 | [0.299558, 0.300239] | 0.9784 | 0.9784 [0.9740, 0.9821] | 0 |
| jump | beta_4_8 | 1000 | [0.299705, 0.300045] | 0.9770 | 0.9770 [0.9725, 0.9808] | 0 |
| logistic | uniform | 500 | [0.297779, 0.302811] | 0.9790 | 0.9790 [0.9746, 0.9826] | 0 |
| logistic | uniform | 1000 | [0.298889, 0.301407] | 0.9776 | 0.9776 [0.9731, 0.9813] | 0 |
| logistic | beta_4_8 | 500 | [0.299071, 0.300778] | 0.9798 | 0.9798 [0.9755, 0.9833] | 0 |
| logistic | beta_4_8 | 1000 | [0.299472, 0.300350] | 0.9784 | 0.9784 [0.9740, 0.9821] | 0 |

Only 2 of 80,000 classifications change. Both are
onset/uniform trials. The overall target-coverage range remains
0.9770--0.9804.
