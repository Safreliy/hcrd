# Reproducing the TSB-AD area-spectrum study

Raw TSB-AD data and third-party benchmark code are intentionally excluded from
this repository. Their original licenses and source-dataset terms continue to
apply. The downloader places them in the sibling `third_party/` directory used
by the frozen scripts and verifies both the data hash and benchmark commit.

```powershell
python -m pip install -e ".[dev,tsad]"
python experiments/download_tsb_ad.py
python -m pytest -q tests/test_energy.py tests/test_anomaly.py `
  tests/test_temporal_anomaly.py tests/test_component_anomaly.py `
  tests/test_tsad_metrics.py
```

## Fast audit

The published scores can be analysed without rerunning the detector:

```powershell
python experiments/analyze_tsb_ad_area_results.py
python experiments/generate_tsb_ad_figures.py
```

`analyze_tsb_ad_area_results.py` reads the saved HCRD scores and the official
per-file comparator tables. It does not tune or rerun HCRD.

## Full protocol order

Use a fresh clone if you want to reproduce every generated artifact, because
the following commands intentionally rebuild the frozen result directories.
The order matters: A1 must freeze before A2, A2 before A3, and all three before
the evaluation script will run.

```powershell
python experiments/run_tsb_ad_area_tuning.py --jobs 4
python experiments/run_tsb_ad_temporal_tuning.py --jobs 4
python experiments/run_tsb_ad_component_tuning.py --jobs 4
python experiments/run_tsb_ad_area_evaluation.py --jobs 4
python experiments/analyze_tsb_ad_area_results.py
python experiments/run_tsb_ad_area_runtime.py --jobs 1
python experiments/run_tsb_ad_area_runtime.py --jobs 4
python experiments/generate_tsb_ad_figures.py
```

Expected frozen A1 candidate: `hcrd_L8_max`. Expected evaluation mean VUS-PR:
`0.34687822104563465` over 350 series. Runtime is hardware-specific; per-file
scores, exact VUS values, file-list hashes, and frozen configuration hashes are
the cross-machine targets.

## C1 new-series Yahoo confirmation

The C1 extension additionally uses the official TSB-UAD Public archive and
published per-series AUC-PR table. It downloads only the 367 Yahoo files from
the verified archive. On a fresh clone, freeze must precede evaluation:

```powershell
python experiments/download_tsb_uad_yahoo.py
python experiments/run_tsb_uad_yahoo_confirmation.py --freeze-only
python experiments/run_tsb_uad_yahoo_confirmation.py --evaluate
python experiments/generate_tsb_uad_yahoo_figure.py
```

Expected primary result on 134 point-anomaly series: HCRD mean AUC-PR
`0.5285392608108096`, NORMA `0.33828595726119404`, paired difference
`0.1902533035496156`, bootstrap interval
`[0.12940898122750497, 0.25095177463815976]`. The freeze mode verifies the
external archive, benchmark commit, official score table, TSB-AD exclusion
lists, matched manifest, and implementation hashes before evaluation.

## Evidence map

- `docs/tsb_ad_a1_protocol.md`: direct area-spectrum protocol;
- `docs/tsb_ad_a2_protocol.md`: temporal/spectral transformations;
- `docs/tsb_ad_a3_protocol.md`: signed decomposition and learned ablation;
- `results/tsb_ad_a1/`: frozen configuration, tuning, evaluation, runtime, and
  post-evaluation subgroup analysis;
- `results/tsb_ad_a2/` and `results/tsb_ad_a3/`: retained negative development;
- `reports/tsb_ad_area.md`: interpretation and publication boundary;
- `reports/area_series_model.md`: exact identities and unproved temporal models.
- `docs/tsb_uad_yahoo_c1_protocol.md`: fixed C1 population and success rule;
- `results/tsb_uad_yahoo_c1/`: frozen manifest, per-file scores, and paired
  comparisons;
- `reports/tsb_uad_yahoo_c1.md`: interpretation and independence boundary.
