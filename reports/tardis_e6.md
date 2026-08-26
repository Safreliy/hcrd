# TARDIS E6: external no-refit FAME transfer

## Outcome

E6 did not pass its two co-primary superiority rules.  On 192 unambiguous
positive-polarity FAME targets, the pooled Falkor+MESOSCOPE HCRD-8+Q model
improved residual-Bad average precision from 0.4353 for qscore to 0.5078, a
paired difference of 0.0725 (20,000-replicate percentile 95% interval
[-0.0155, 0.1345], two-sided bootstrap p=0.1115).  HCRD-8+Q was essentially
tied with HCRD-1+Q: difference -0.0016 [-0.0500, 0.0453].  DOMAIN+Q had the
largest AP point estimate, 0.5193.

This is useful no-refit external-shift evidence, but it is not evidence that
eight HCRD levels are uniformly preferable.  No window, representation,
learner, target subset, endpoint or bootstrap rule was changed after the
result.

## Frozen design and availability amendment

The protocol was locked in `docs/tardis_e6_protocol.md` before any released
rating row was inspected or chromatogram reconstructed.  The Zenodo archive
was independently assembled from byte ranges and verified at 5,669,337,245
bytes with MD5 `38ddb2822551b1d281b57d610eb56986`.

Pre-score archive inspection established that the released raw scale-analysis
files are positive-polarity only.  Negative-polarity diagnostic plots and
summary tables are available, but numerical negative EICs are not.  The full
FAME panel therefore fails the frozen waveform gate; the positive stratum
passes it.  This availability amendment was recorded before computing any HCRD
features or model scores.

- 119 released positive QC mzXML files;
- 212 targets with complete Component/m/z/RT keys;
- 108 Good, 84 Bad, 20 Ambiguous;
- 192 Good/Bad targets in the binary endpoint;
- all 212 targets had at least one valid feature bank;
- median valid-QC fraction across targets: 0.9412.

Three workbook rows containing only a Component number and no rating, m/z, RT,
ID or name were treated as unrated placeholders.  The runner rejects, rather
than drops, any rated row missing a waveform key.

## Frozen representation and learner

Every EIC was reconstructed from MS1 scans as the summed intensity within 10
ppm of released target m/z and a fixed 60-second box around released expected
RT.  Missing ions within a scan contribute zero intensity.  The same E2 feature
bank was aggregated across all available QC files using the predeclared median,
0.9 quantile, maximum, availability fraction and median qscore.

For each representation, one `StandardScaler` plus balanced L2 logistic model
was fit on the pooled 4,014 unambiguous Falkor and MESOSCOPE targets (491 Good,
3,523 Bad), then applied to FAME without target fitting or recalibration.

## Results

| representation | AP-Bad | ROC AUC-Bad | top-decile Bad enrichment |
|---|---:|---:|---:|
| qscore | 0.4353 | 0.5230 | 0.686 |
| DOMAIN+Q | **0.5193** | 0.6030 | 1.257 |
| HCRD-1+Q | 0.5094 | 0.5902 | **1.486** |
| HCRD-8+Q | 0.5078 | 0.5998 | 1.371 |
| geometry+Q | 0.5044 | **0.6049** | **1.486** |
| area-only+Q | 0.3847 | 0.4377 | 0.571 |

| paired AP contrast | difference | percentile 95% CI | two-sided p |
|---|---:|---:|---:|
| HCRD-8+Q - qscore | +0.0725 | [-0.0155, 0.1345] | 0.1115 |
| HCRD-8+Q - HCRD-1+Q | -0.0016 | [-0.0500, 0.0453] | 0.9483 |
| HCRD-8+Q - DOMAIN+Q | -0.0115 | [-0.0679, 0.0438] | 0.6803 |

The ordered-label Spearman correlations (Good < Ambiguous < Bad) were 0.0387
for qscore, 0.1876 for DOMAIN+Q, 0.1614 for HCRD-1+Q, 0.1799 for HCRD-8+Q,
0.1849 for geometry+Q and -0.0984 for area-only+Q.

## Interpretation

The cross-laboratory shift supports a narrower conclusion than E2 and E5:
decomposition/domain shape information transfers better in point estimate than
the two-variable qscore, but the sample does not establish the gain and does
not favour eight levels over one.  Area/energy alone reverses direction, so it
should remain an auxiliary descriptor rather than the principal representation.
The result is consistent with a task-dependent bias--variance trade-off: richer
multilevel coordinates help when fine lobe defects align across studies, while
compact conventional or one-level shape summaries can be more robust under a
large acquisition/processing shift.

## Reproduction

```powershell
python experiments/run_tardis_e6.py extract `
  --mzxml-dir data/external/tardis_extracted/data_zenodo/scale_analysis/files700 `
  --labels-workbook data/external/tardis_extracted/data_zenodo/quality_analysis/fame_feat_table_pos.xlsx `
  --archive data/external/data_zenodo.rar `
  --output-dir results/tardis_e6/fame_positive --workers 4

python experiments/run_tardis_e6.py evaluate `
  --falkor-dir results/ms_metrics_e2/falkor `
  --mesoscope-dir results/ms_metrics_e2/mesoscope `
  --target-dir results/tardis_e6/fame_positive `
  --output-dir results/tardis_e6/evaluation --bootstrap 20000
```

Compact machine-readable results are in
`results/tardis_e6/evaluation/e6_results.json`.  Third-party raw data and large
derived tensors are intentionally not redistributed in the public release.
