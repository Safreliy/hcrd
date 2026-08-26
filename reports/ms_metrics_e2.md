# E2 cross-dataset LC--MS mass-feature quality transfer

Protocol: `docs/ms_metrics_e2_protocol.md`, frozen before raw-waveform
inspection. Source: Kumler, Hazelton and Ingalls (2023), DOI
`10.1186/s12859-023-05533-4`; official repository commit
`491deaf1d5f27f9d276e58acb4c1dfca2a2e21b9`.

## Data and fidelity

- Falkor: 41 mzML files; 242 Good, 1579 Bad, 219 excluded labels.
- MESOSCOPE: 168 mzML files; 249 Good, 1944 Bad, 341 excluded labels.
- Train on one complete dataset and apply to the other without refitting,
  recalibration, threshold selection, or random within-dataset splitting.
- The independently re-extracted global-window qscore correlates strongly with
  the authors' published Falkor metrics: Pearson/Spearman 0.906/0.871 for SNR
  and 0.789/0.805 for bell correlation (`n=1495`).

## Primary result

| Transfer | qscore AP | HCRD-1+Q AP | HCRD-8+Q AP | HCRD-8 − qscore | paired 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Falkor → MESOSCOPE | 0.777441 | 0.851557 | **0.895494** | +0.118053 | [0.061314, 0.178074] | 0.000400 |
| MESOSCOPE → Falkor | 0.798416 | 0.844576 | **0.898962** | +0.100546 | [0.055896, 0.146970] | 0.000400 |

Every pre-specified E2 success criterion was met in both directions.

## Strong comparators and ablations

| Transfer | DOMAIN+Q | HCRD geometry+Q | Area/energy spectrum+Q | best point AP |
|---|---:|---:|---:|---:|
| Falkor → MESOSCOPE | 0.891895 | **0.915000** | 0.909361 | HCRD geometry+Q |
| MESOSCOPE → Falkor | 0.916050 | 0.933915 | **0.941210** | area/energy spectrum+Q |

Full HCRD-8 did not establish superiority to DOMAIN+Q: AP differences were
`+0.003599`, CI `[-0.046856, 0.056105]`, and `-0.017088`, CI
`[-0.047939, 0.015395]`. The best point representation belonged to the HCRD
family in each direction, but family-wise superiority over every tested
non-neural representation was not a predeclared or statistically established
claim.

HCRD-8 exceeded HCRD-1 by 0.043936 in Falkor→MESOSCOPE (CI
`[-0.016725, 0.105995]`) and 0.054385 in MESOSCOPE→Falkor (CI
`[0.007583, 0.095480]`, Holm `p=0.037196`). Thus the multilevel direction is
consistent in both transfers and statistically resolved in one.

## Interpretation

This is the first independent real-data task-class confirmation for HCRD to
meet its pre-specified success criteria. It supports multilevel convex-lobe geometry as an
incremental representation for curation of pre-bounded HILIC LC--MS mass
features. It does not show that the dense full hierarchy is always the optimal
summary: E1 favored raw plus full HCRD-8, whereas E2 favored compact multilevel
geometry or area/energy spectra. That heterogeneity argues for a small declared
HCRD representation family and a new external selection/confirmation design,
not for silently selecting the winning E2 ablation.

The benchmark does not test upstream feature discovery, compound
identification, other chromatography modes, or neural image models.
