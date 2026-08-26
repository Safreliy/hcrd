# TSB-AD: temporal HCRD area spectrum for anomaly detection

## Outcome

This is the first broad real-data result that identifies a defensible practical
class for HCRD. The frozen, training-free detector is **not** competitive as a
universal time-series anomaly detector. It is, however, particularly strong on
isolated point anomalies: on the 49 predeclared point-anomaly evaluation series
it has the highest mean VUS-PR among the published nonpretrained baselines in
the official score tables.

That sentence is intentionally narrower than a state-of-the-art claim. The
paired bootstrap interval versus the strongest nonpretrained point estimate
includes zero, and pretrained models remain better. The result locates a useful
task class. A later pre-specified extension on 134 additional Yahoo
point-anomaly series confirms the unchanged detector against the strongest
published non-neural comparator, while still leaving cross-source replication
open.

## Why this benchmark

[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) was introduced in the NeurIPS
2024 Datasets and Benchmarks track. It aggregates heterogeneous univariate and
multivariate anomaly-detection data and uses range-aware VUS metrics. We used
the official univariate split exactly as published:

- 48 tuning series;
- 350 sealed evaluation series;
- mean per-series VUS-PR as the selection and evaluation metric;
- the benchmark's rank-1 autocorrelation window and 250 score thresholds.

The official repository was fixed at commit
`e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48`. The data archive SHA-256 is
`0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961`.

## Detector

For HCRD detail level `l`, define the temporal polygon-mass density

`a_l(t) = |d_l(t)|`.

Its integral is exactly the sum of the polygon areas of all level-`l`
structures. This is an L1 geometric identity, not a Parseval-energy statement.
It retains where the mass occurs, whereas one scalar area per record discards
the temporal organization.

Each level is normalized without labels:

`z_l(t) = max((a_l(t) - median(a_l)) / scale_l, 0)`,

where `scale_l` is the 90th percentile minus the median, with a maximum-minus-
median fallback. The tuning split selected eight levels and

`score(t) = max_l z_l(t)`.

The `max` rule is meaningful: it lets a transient appear at whichever
hierarchy scale matches its width, while summing all levels can dilute that
localized response with ordinary activity at unrelated scales. The score is
invariant to adding an affine signal and to nonzero vertical rescaling.

## Frozen selection

All 15 combinations of depths 4, 8, and complete with five fixed aggregation
rules were screened on the official tuning set. Five passed to exact VUS
evaluation. The selected configuration was written before evaluation was read.

| Tuning candidate | Mean VUS-PR |
|---|---:|
| HCRD, 8 levels, max | **0.3538** |
| HCRD, complete, max | 0.3537 |
| HCRD, 8 levels, L2 | 0.3515 |
| raw absolute median deviation | 0.2679 |

The near tie between eight and all levels supports the fixed-depth practical
implementation: more hierarchy is not automatically more information for a
point score.

## Sealed evaluation

### Overall

| Method | Training regime | Mean VUS-PR |
|---|---|---:|
| HCRD L8-max | none | 0.3469 |
| Matrix Profile | none | 0.3496 |
| Series2Graph | none | 0.3881 |
| KShapeAD | none | 0.4008 |
| Sub-PCA | none | 0.4234 |
| MMPAD | none | 0.4399 |
| StreamVAE | none | 0.4508 |
| TSPulse ZS | pretrained | 0.4801 |
| Time-RCD+MAFT FT | pretrained | 0.5856 |

HCRD is essentially tied with Matrix Profile overall: paired difference
`-0.0027`, bootstrap 95% interval `[-0.0450, 0.0405]`, with 178/0/172
wins/ties/losses. It is significantly below several general detectors. This
rejects universal TSAD superiority.

### Point-anomaly stratum

| Method | Training regime | Mean VUS-PR |
|---|---|---:|
| Time-RCD+MAFT FT | pretrained | 0.8418 |
| TSPulse ZS | pretrained | 0.7807 |
| **HCRD L8-max** | **none** | **0.6755** |
| KShapeAD | none | 0.6003 |
| MMPAD | none | 0.5948 |
| Matrix Profile | none | 0.5402 |
| Series2Graph | none | 0.4136 |
| Sub-PCA | none | 0.3566 |

HCRD has the highest mean among the published nonpretrained baselines. Versus
KShapeAD the paired mean difference is `+0.0752`, with 28/6/15
wins/ties/losses, but its file-bootstrap 95% interval
`[-0.0632, 0.2130]` crosses zero. Hence this is a strong specialization signal,
not yet a statistically confirmed dominance claim.

The same pattern appears by source and domain. HCRD is strong on TODS
(`0.7796`), YAHOO (`0.6761`), synthetic (`0.5965`), and web-service (`0.4230`)
series. It is poor on long sequence or repetitive morphology anomalies,
especially human activity (`0.1446`) and medical series (`0.1049`).

## Frozen C1 new-series confirmation

Before computing any extension score, C1 excluded all TSB-AD tuning and
evaluation files, content-matched 220 remaining Yahoo series to the official
TSB-UAD archive, froze the unchanged `hcrd_L8_max` implementation, and declared
the 134 official point-anomaly files primary. NORMA was the strongest published
non-neural method on that fixed population before HCRD execution.

| Method | Mean per-series AUC-PR |
|---|---:|
| CNN | 0.9052 |
| **HCRD L8-max** | **0.5285** |
| LSTM | 0.4640 |
| NORMA | 0.3383 |
| IForest1 | 0.2960 |
| Matrix Profile | 0.0758 |

HCRD-minus-NORMA was `+0.1903`, paired bootstrap interval
`[+0.1294, +0.2510]`, with 91/6/37 wins/ties/losses. This passed the
predeclared criterion. HCRD's mean was above LSTM, but that paired interval
crossed zero; CNN was decisively higher. C1 therefore confirms a useful
training-free point-transient niche within new Yahoo files. It is not an
independent-source confirmation. Full details and immutable hashes are in
`reports/tsb_uad_yahoo_c1.md` and `results/tsb_uad_yahoo_c1/`.

## Did a second analysis of the area series help?

Not yet. Protocol A2 froze and tested temporal transformations of the full
area spectrum on the tuning split. The direct score remained best:

| A2 candidate | Mean VUS-PR |
|---|---:|
| direct HCRD L8-max | **0.3538** |
| direct + spectral-residual fusion | 0.3453 |
| direct + total-area spectral residual | 0.3245 |
| total-area spectral residual | 0.3029 |

So the user's idea is mathematically valid and has produced a reusable
multivariate time series, but a generic spectral-residual postprocessor adds no
value here. That negative result does not rule out sequence models, change-point
methods, or scale-transport analysis on tasks with regime changes.

## Did the signed decomposition and a learned model help?

Protocol A3 made the signed eight-level detail decomposition primary and used
area only as a supplement. Isolation Forest candidates included signed details,
absolute details, raw signal, first differences, total mass, scale barycentre,
and scale entropy. The best fusion reached `0.3471`; direct HCRD remained at
`0.3538`. Signed details alone reached `0.2393`.

This rules out one simple hybrid, not learning in general. A pointwise
Isolation Forest ignores ordering and the tree relations among nested HCRD
structures. A promising learned continuation should consume the hierarchy as a
sequence or graph and should be developed only on the tuning split or on a new
dataset.

## Runtime and parallelism

The clean recalculation processed all 350 series (18,160,361 samples) on the
local machine:

| Mode | Detector/wall time | Wall speedup | Median detector ms / 10k |
|---|---:|---:|---:|
| one process | 49.94 / 53.09 s | 1.00x | 12.26 |
| four processes | 97.03 / 28.79 s | 1.84x | 21.49 |

Independent files parallelize exactly, but four workers do not give fourfold
speedup. The dense eight-level area matrix is memory-bandwidth intensive and
process scheduling adds overhead. At fixed eight-level depth the detector is
linear in series length; the current dense materialization uses `O(8n)` memory.

## What has been established

- **Exact:** every area-density row integrates to the corresponding HCRD
  polygon mass; affine additions vanish; normalized scores are amplitude-scale
  invariant.
- **Confirmed on the sealed benchmark:** the frozen detector is specialized
  for point anomalies and is not a universal TSAD method.
- **Observed, not proved:** the maximum across scales is a useful inductive bias
  for isolated transients.
- **Falsified here:** generic spectral-residual processing and a pointwise
  Isolation Forest improve the frozen direct score.

## Interpretation and scope

The C1 result supports an interpretable, label-free geometric detector for
point-transient anomalies. Its evidence is limited to the evaluated Yahoo
families: the independent-source KDD21/UCR screen contained only five eligible
point-anomaly files, below the prespecified minimum of ten. Independent-source
validation and a pulse-plus-background detection analysis would test whether
the observed specialization transfers beyond these generators.
