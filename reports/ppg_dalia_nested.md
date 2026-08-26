# PPG-DaLiA nested subject-wise HCRD result

Author: Saveliy Baturin, Independent Researcher

All values are cross-fitted outer-test results from the frozen five-fold subject protocol.

| Method | Median exact F1 | Micro-F1 | Motion macro F1 |
|---|---:|---:|---:|
| P0 find_peaks | 0.8013 | 0.8078 | 0.7275 |
| HCRD mass-only | 0.7659 | 0.7613 | 0.7137 |
| HCRD geometry | 0.7945 | 0.7904 | 0.7413 |
| HCRD hybrid / primary | 0.7918 | 0.7923 | 0.7382 |

| Activity | Primary | Geometry | Mass-only | P0 |
|---|---:|---:|---:|---:|
| car_driving | 0.8313 | 0.8250 | 0.7983 | 0.8412 |
| cycling | 0.8760 | 0.8913 | 0.8731 | 0.8802 |
| lunch_break | 0.8089 | 0.8010 | 0.7721 | 0.8236 |
| sitting | 0.7628 | 0.7741 | 0.7277 | 0.8545 |
| stair_climbing | 0.7743 | 0.7818 | 0.7688 | 0.7539 |
| table_soccer | 0.6548 | 0.6548 | 0.6332 | 0.6460 |
| walking | 0.7856 | 0.7873 | 0.7392 | 0.7827 |
| working | 0.8204 | 0.8119 | 0.7702 | 0.8451 |

## Paired subject bootstrap

- primary_vs_p0: overall median difference -0.0094 (95% CI -0.0476 to 0.0125); motion difference 0.0107 (95% CI -0.0069 to 0.0270).
- geometry_vs_p0: overall median difference -0.0068 (95% CI -0.0469 to 0.0138); motion difference 0.0138 (95% CI -0.0135 to 0.0300).
- primary_vs_mass: overall median difference 0.0259 (95% CI 0.0152 to 0.0563); motion difference 0.0245 (95% CI 0.0075 to 0.0411).
- geometry_vs_mass: overall median difference 0.0286 (95% CI 0.0146 to 0.0544); motion difference 0.0275 (95% CI 0.0002 to 0.0415).
- exploratory motion_gate_vs_p0: overall median difference 0.0027 (95% CI -0.0058 to 0.0214); motion difference 0.0057 (95% CI -0.0023 to 0.0201). This analysis is post-outer-test.

## Interpretation

The predeclared broad success rule failed: HCRD does not beat P0 over all activities. The full hierarchy materially beats the mass-only representation. The primary HCRD model beats P0 on all three predeclared motion-intensive activities, but the subject bootstrap determines whether that narrower effect is sufficiently stable for a claim.
