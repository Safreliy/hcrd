# WSD C2: independent real sparse-transient confirmation

## Outcome

The broad class hypothesis did not confirm.  On the 71 pre-specified
primary WSD series, HCRD L8-max had mean VUS-PR 0.375596 and the comparator
POLY, frozen before HCRD execution, had 0.411531.  The paired mean difference
was -0.035935 with a 50,000-draw file-bootstrap 95% interval
[-0.130974, 0.061572].  Paired t p=0.4720, Wilcoxon p=0.3051, exact sign
p=0.00855, with 24/0/47 wins/ties/losses.  Thus mean inferiority is uncertain,
but losses are systematically more frequent.

The primary population contained all real WSD web-service KPI series unused by
the official TSB-AD tuning/evaluation lists satisfying maximum anomaly-run
fraction <=0.005 and occupancy <=0.01.  The thresholds used labels but were
fixed before any C2 method score.

## Comparator audit

Four official wrappers completed all 86 series and were eligible:

| Method | Primary mean VUS-PR |
|---|---:|
| POLY | 0.411531 |
| SR | 0.175079 |
| Sub-PCA | 0.113596 |
| Sub-IForest | 0.033178 |

MMPAD, KShapeAD, SAND, and Matrix Profile were incomplete.  SAND also raised
an official-wrapper sample-size error on short training regions.  Partial
outputs and tracebacks are retained and are not counted as HCRD wins.  The
post-execution exporter amendment changed no endpoint, stratum, comparator,
seed, test, or multiplicity rule.

## Frozen structural audit

The secondary outcome was per-series HCRD-minus-POLY VUS-PR.  Four directional
hypotheses did not survive Holm correction:

| Descriptor | Direction | Spearman rho | Permutation p | Holm p |
|---|---|---:|---:|---:|
| Sign coherence | positive | -0.0984 | 0.7944 | 1.0000 |
| Peak/background MAD | positive | -0.1615 | 0.9085 | 1.0000 |
| Curvature contrast | positive | 0.0550 | 0.3256 | 0.9769 |
| Duration fraction | negative | -0.1972 | 0.0487 | 0.1950 |

The prespecified exploratory shape-concentration descriptor had rho=0.3157
and two-sided 100,000-permutation p=0.00791.  It is mechanism-generating, not a
post-hoc class definition.  The next valid test must freeze a compact-impulse
task on an untouched source.

No runtime claim is made because the host CPU was concurrently loaded.
