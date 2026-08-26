# PPG-DaLiA D3 local-motion development result

Author: Saveliy Baturin, Independent Researcher

Status: post-outer-test exploratory development. The D3 specification was
hashed before this run, but the underlying PPG-DaLiA outer tests had already
been inspected. Independent-cohort confirmation is mandatory.

The full eight-level, 140-coordinate HCRD geometry remained the main
representation. Eleven local wrist-acceleration coordinates were appended as
context; the 39-coordinate mass-only representation remained an ablation.

| Method | Median exact F1 | Micro-F1 | Motion macro F1 |
|---|---:|---:|---:|
| Tuned find_peaks P0 | 0.8013 | 0.8078 | 0.7275 |
| Frozen HCRD geometry | 0.7945 | 0.7904 | 0.7413 |
| Frozen HCRD geometry+morphology | 0.7918 | 0.7923 | 0.7382 |
| D3 geometry+motion | 0.7923 | 0.7881 | 0.7376 |
| D3 geometry+morphology+motion | 0.7918 | 0.7968 | 0.7419 |
| D3 cross-fitted primary | 0.7918 | 0.7941 | 0.7394 |

Validation selected hybrid_motion in folds 0, 1, 3, and 4 and
geometry_motion in fold 2. The selected thresholds were 0.35, 0.35, 0.20,
0.40, and 0.30.

## Paired subject bootstrap

- D3 primary minus P0: median F1 -0.0095 (95% CI -0.0486 to 0.0234);
  micro-F1 -0.0137 (-0.0267 to -0.0023); motion macro +0.0119
  (-0.0129 to 0.0321).
- D3 primary minus frozen hybrid HCRD: median F1 -0.0001
  (-0.0160 to 0.0210); micro-F1 +0.0018 (-0.0027 to 0.0059); motion macro
  +0.0012 (-0.0125 to 0.0098).
- D3 primary minus mass-only: median F1 +0.0259
  (0.0185 to 0.0665); micro-F1 +0.0328 (0.0279 to 0.0376); motion macro
  +0.0257 (0.0084 to 0.0424).

## Interpretation

Local acceleration does not repair the overall gap to the simple peak
detector and does not materially improve the frozen HCRD models. It does not
change the representation conclusion: the complete hierarchy remains
substantially more informative than mass/energy alone. The main practical
bottleneck is transfer of event calibration across unseen subjects, especially
quiet versus motion-corrupted regimes.
