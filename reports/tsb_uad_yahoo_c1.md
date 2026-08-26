# C1: held-out Yahoo confirmation of the HCRD point-transient detector

## Result

The unchanged, training-free `hcrd_L8_max` detector passed its pre-specified
confirmation criterion on 134 additional Yahoo point-anomaly series:

| Method | Training | Mean per-series AUC-PR |
|---|---|---:|
| CNN | learned | 0.9052 |
| **HCRD L8-max** | **none** | **0.5285** |
| LSTM | learned | 0.4640 |
| NORMA | none | 0.3383 |
| IForest1 | none | 0.2960 |
| POLY | none | 0.1273 |
| Matrix Profile | none | 0.0758 |

NORMA was declared as the primary comparator before HCRD was executed because
it had the highest fixed-population mean among the published non-neural
methods. The paired HCRD-minus-NORMA difference was `+0.1903`, with a 95%
file-bootstrap interval `[+0.1294, +0.2510]` and 91/6/37
wins/ties/losses. The lower interval endpoint is positive, so the predeclared
success condition was met.

HCRD's point estimate also exceeded LSTM by `0.0646`, but the paired interval
`[-0.0129, +0.1421]` crosses zero. CNN remained substantially better by
`0.3766`, with interval `[-0.4458, -0.3020]` for HCRD minus CNN. The supported
claim is therefore strongest-training-free-classical performance on this fixed
point-transient population, not universal or learned-model state of the art.

As a secondary result on all 220 unique matched series, HCRD reached `0.5859`
versus LSTM `0.5086`, NORMA `0.3021`, and CNN `0.7961`. HCRD-minus-LSTM was
`+0.0773`, interval `[+0.0196, +0.1360]`. This is descriptive because the
point-anomaly stratum, not the full population, was the primary endpoint.

## Why this is a confirmation rather than another search

The official TSB-AD tuning and evaluation lists had already been consumed.
Before computing a C1 HCRD score, the experiment:

1. excluded every TSB-AD-U tuning and evaluation filename;
2. content-matched the remaining Yahoo series to the official TSB-UAD Public
   archive using length, first anomaly, exact labels, and signals within
   `1e-10` (maximum observed difference `5.7e-14`);
3. excluded four ambiguous duplicate-content matches;
4. fixed the resulting 220-series manifest and its 134-series official
   `point_anom == 1` primary stratum;
5. recorded hashes of the detector, protocol, manifest, official score table,
   file lists, and public archive;
6. fixed NORMA, AUC-PR, the paired bootstrap, and a positive lower confidence
   bound as the success rule.

Only then was evaluation mode allowed to call HCRD. The detector is exactly the
eight-level maximum area-surprise rule selected in A1; no label, threshold,
source-specific parameter, or new fit was introduced.

## Independence boundary

These are previously unevaluated series, but they come from the same Yahoo
dataset family as part of TSB-AD. C1 is strong **new-series replication within
one source family**, not independent-source replication. Published comparator
values are taken from the official TSB-UAD per-series AUC-PR table rather than
rerun. The comparison is paired and exact for the fixed content-matched files,
but it inherits any limitations of those archived implementations and scores.

## Practical interpretation

HCRD has now shown the same specialization twice:

- on the sealed TSB-AD evaluation, its point-anomaly VUS-PR was 0.6755 versus
  0.6003 for KShapeAD, although the 49-file interval was inconclusive;
- on 134 additional frozen Yahoo point-anomaly files, it beat the strongest
  published non-neural TSB-UAD comparator by a large, interval-separated
  margin.

The mechanism fits isolated transients: a short deviation creates localized
polygon mass at one or more hierarchy levels, and maximum aggregation prevents
unrelated scales from diluting it. Long contextual or collective anomalies do
not necessarily create a locally exceptional convexity structure, explaining
why the same detector is not universally strong on TSB-AD.

This supports a practically useful niche: a fast, fitting-free, interpretable
first-stage detector for isolated spikes, dips, and pulse-like faults. CNN shows
that substantially higher accuracy is possible when labelled/family-specific
learning is acceptable. HCRD instead offers no training, affine invariance,
linear fixed-depth work, a scale-localized explanation, and deterministic
parallel processing across independent series.

## Reproducibility artifacts

- `docs/tsb_uad_yahoo_c1_protocol.md`: frozen protocol and independence claim;
- `results/tsb_uad_yahoo_c1/frozen_configuration.json`: pre-execution hashes;
- `results/tsb_uad_yahoo_c1/confirmation_manifest.csv`: fixed matched files;
- `results/tsb_uad_yahoo_c1/confirmation_metrics.csv.gz`: per-file outcomes;
- `results/tsb_uad_yahoo_c1/point_comparisons.csv`: paired primary comparisons;
- `experiments/run_tsb_uad_yahoo_confirmation.py`: freeze/evaluate runner.

The external archives and benchmark repositories are not redistributed.
