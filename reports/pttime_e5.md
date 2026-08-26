# Pttime E5 conditional external verification

## Decision

The E5 success rule, frozen before outcome inspection, **passed**. A model trained only on
the pooled, fully labelled Falkor and MESOSCOPE studies transferred without
refitting to the independently acquired Pttime selected subset. The primary
operational task was to rank the 17 residual `Bad` features among 348 `Good`
features that the source two-variable model had already selected.

| Representation | AP-bad | ROC AUC-bad | Top 1% (4) | Top 5% (19) | Top 10% (37) |
|---|---:|---:|---:|---:|---:|
| qscore | 0.0391 | 0.3403 | 0 | 0 | 1 |
| DOMAIN+Q | 0.3873 | 0.8115 | 4 | 6 | 6 |
| HCRD-1+Q | 0.2845 | 0.7792 | 2 | 5 | 7 |
| **HCRD-8+Q** | **0.5085** | **0.8646** | **4** | **7** | **8** |
| HCRD geometry+Q | 0.3540 | 0.8081 | 3 | 6 | 7 |
| Area/energy+Q | 0.1367 | 0.7194 | 1 | 3 | 5 |

At the 5% budget, HCRD-8+Q has 36.8% queue precision and retrieves 41.2% of
the residual bad cases. To retrieve at least 9 of 17 cases, HCRD-8+Q requires
44 reviews versus 284 for qscore, an 84.5% workload reduction at the same
integer recall target.

The 10,000-replicate paired class-stratified bootstrap gave:

- HCRD-8+Q minus qscore: +0.4694, 95% CI [0.2489, 0.6762], p=0.00020;
- HCRD-8+Q minus HCRD-1+Q: +0.2240, [0.0707, 0.3544], p=0.00060;
- HCRD-8+Q minus DOMAIN+Q: +0.1212, [-0.0100, 0.2794], p=0.0818.

Thus the result confirms that the multilevel decomposition adds useful
conditional residual-error information beyond both the source score and one
HCRD level. HCRD-8 had the best point estimate, but E5 does not establish its
superiority to the larger conventional domain bank.

## Frozen population and correction

The source repository labels only 400 of 7,781 Pttime mass features: 348 Good,
17 Bad, and 35 Ambiguous cases selected at source-model probability above 0.9.
Ambiguous cases were excluded. After archive download, but before inspecting
any target waveform or computing any E5 score, archive inspection revealed 52
POS and 52 NEG mzML files. The published source pipeline and feature boxes use
the POS schema, so the extraction was corrected to the 52 matching POS files.
This pre-extraction correction is recorded in the frozen protocol.

## Interpretation

E5 supports conditional triage of residual false positives after a
high-specificity HILIC feature-quality filter. It cannot estimate performance,
recall, calibration, or FDR over all 7,781 Pttime features and is not a full
peak-picking comparison.

## Reproduction record

- protocol: `docs/pttime_e5_protocol.md`;
- runner: `experiments/run_pttime_e5.py`;
- compact result: `results/pttime_e5/evaluation/results.json`;
- target predictions: `results/pttime_e5/evaluation/predictions.csv`;
- review-utility output: `results/pttime_e5/evaluation/review_utility.json`;
- utility runner: `experiments/analyze_pttime_review_utility.py`;
- source repository commit: `491deaf1d5f27f9d276e58acb4c1dfca2a2e21b9`;
- ST002077 archive SHA-256:
  `6e06eed561c82a5d4667b88089a37ea620b2c7cedc7a35fa46f9bd5d727d5040`;
- label-file SHA-256:
  `0a9e03ed7afdc8dc9181137f0f8ebd748467afa72c94443d88ff5b93c8ce778d`.

Raw third-party mzML files and per-file NumPy caches are intentionally excluded
from the public release. The compact fitted models, predictions, metadata, and
statistics are included.
