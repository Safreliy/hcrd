# Reproducibility guide

## Tier 1: implementation audit

```bash
python -m pip install -e ".[dev,comparisons,lcms]"
python -m pytest -q
python examples/quickstart.py
python scripts/verify_release.py
```

Expected unit-test result for this snapshot: `180 passed`.

## Tier 2: synthetic and theorem-linked studies

These runs need no external signal dataset:

```bash
python experiments/run_recovery_phase_diagram.py
python experiments/generate_recovery_phase_figure.py
python experiments/run_approximate_join_phase.py
python experiments/generate_approximate_join_phase_figure.py
python experiments/run_synthetic_benchmark.py
python experiments/run_external_comparison.py --trials 50 --output results/external_comparison_v02
python experiments/run_external_comparison.py --trials 50 --noise 0 --suite exact --output results/external_comparison_itd_c2
python experiments/run_falsification_suite.py
python experiments/run_stability_study.py
python experiments/run_stable_confirmation.py
python experiments/run_persistence_stability.py
python experiments/run_ceemdan_confirmation.py
python experiments/run_lobe_scan_monte_carlo.py
python experiments/run_continuous_lobe_scan_audit.py
```

Expected finite-sample recovery result: 44,000 trials; exact recovery 1.000 in
all eight cells with `gamma/tau > 2`; smallest cellwise 95% Wilson lower bound
0.9962. The joint knot-and-reconstruction certificate has minimum probability
0.9980 above the boundary. Compact outputs are in
`results/recovery_phase_r1/`.

Expected approximate-join result: 162,000 unequal-amplitude signals; exact
recovery in all 72,000 draws satisfying `gamma/tau > eta/tau + 2` under both
the join-inactivating and noise-only tolerances; smallest cellwise 95% Wilson
lower bound 0.9962 and zero theorem-implication violations. Compact outputs are
in `results/approximate_join_phase_r1/`.

Expected rank-aware continuum audit: affine-residual rank 127, exact
projected-norm quantile 12.421816, direct projected-norm exceedance 0.05074 in
50,000 draws (95% CI [0.04883, 0.05270]), and the corresponding conservative
80%-power sufficient norm 13.263437. This calibrates a dominating norm and
therefore gives scan level at most alpha; it is not generally the scan's exact
null quantile. Compact outputs are in
`results/continuous_lobe_scan_audit_v2/`.

E3 uses Python 3.12 and the separate `iterative-filtering` dependency group.

## Tier 3: external-data studies

### LC--MS E1: double group holdout

E1 uses the 255,000 expert-labelled extracted-ion chromatograms of Mueller et
al. The protocol independently hashes sample and candidate-ion groups, so the
confirmation block shares neither with model development.

```bash
python -m pip install -e ".[lcms]"
python experiments/download_lcms_eic_subset.py
python experiments/run_lcms_eic_e1.py --help
```

Expected primary result: HCRD-8 has the best confirmation point AP (0.4092),
but its two-way cluster-bootstrap advantage over the frozen conventional bank
has interval `[-0.0566, 0.1010]`. The pre-specified success criterion is
therefore not met. Large flattened feature caches are deliberately excluded.

### LC--MS E2: cross-study HILIC transfer

E2 is the main independent real-data confirmation. It uses all unambiguous
expert labels from the Falkor and MESOSCOPE studies and evaluates both transfer
directions without target refitting, calibration, or threshold selection.

```bash
python -m pip install -e ".[lcms]"
python experiments/download_ms_metrics_e2.py
python experiments/run_ms_metrics_e2.py extract-dataset --help
python experiments/run_ms_metrics_e2.py fit-evaluate --help
python experiments/run_ms_metrics_e2_matched_capacity.py --bootstrap 10000
python experiments/run_ms_metrics_e2_refit_sensitivity.py --help
python experiments/run_ms_metrics_e2_fixed_source_block.py --help
python experiments/run_ms_metrics_e2_file_group_sensitivity.py --help
python experiments/run_qscore_implementation_sensitivity.py extract-min5 --help
python experiments/run_qscore_implementation_sensitivity.py evaluate --bootstrap 10000
python experiments/generate_revision_sensitivity_artifacts.py
```

Expected primary AP results:

- Falkor to MESOSCOPE: qscore 0.777441, HCRD-8+Q 0.895494, difference
  0.118053, paired 95% CI `[0.061314, 0.178074]`;
- MESOSCOPE to Falkor: qscore 0.798416, HCRD-8+Q 0.898962, difference
  0.100546, paired 95% CI `[0.055896, 0.146970]`.

At the pre-specified 60-second block width, conditional target-side
retention-time resampling with the saved source model gave intervals
`[0.013014, 0.219030]` and `[0.033783, 0.196454]` in the two transfer
directions. Five of six fixed-model intervals were positive across the 30-,
60-, and 120-second sensitivity grid; the 120-second Falkor-to-MESOSCOPE
interval was `[-0.022713, 0.241809]`. Source-refit block bootstraps retain wider
intervals crossing zero, so this experiment addresses target dependence but
does not absorb source-training uncertainty.

The Holm-adjusted primary value is 0.000400 in both directions. Exact source
ZIP hashes, repository commit, labels, runner hash, compact models, and
statistics are included. Raw mzML and multi-gigabyte per-file arrays are not.
See `docs/ms_metrics_e2_protocol.md` and `reports/ms_metrics_e2.md`.

The qscore implementation sensitivity evaluates the current two medians, the
authors' five-point minimum, an author-like five-summary version, and a
seven-variable robust summary. Across both transfer directions, HCRD-8 adds
0.0765 to 0.1517 AP over the identical qscore variant, and all eight paired
95% intervals remain positive (maximum Holm-adjusted bootstrap value 0.0020). See
`docs/qscore_implementation_sensitivity_protocol.md` and
`reports/qscore_implementation_sensitivity.md`.

The source-refit sensitivity uses 1,000 paired 60-second RT-block bootstrap
replicates and 300 replicates at 30 and 120 seconds. The acquisition-file
sensitivity removes ten deterministic file groups, recomputes qscore and the
full HCRD-8 aggregation, and refits both directions. See
`docs/ms_metrics_e2_refit_sensitivity_protocol.md` and
`reports/ms_metrics_e2_refit_sensitivity.md` for the reproduced intervals and
delete-group ranges. Source-refit mean differences and percentile 95% intervals
for Falkor to MESOSCOPE / MESOSCOPE to Falkor are:

- 30 s: `0.108739 [-0.010481, 0.207425]` / `0.055614 [-0.067086, 0.160175]`;
- 60 s: `0.100014 [-0.044448, 0.217858]` / `0.051523 [-0.093545, 0.174151]`;
- 120 s: `0.088602 [-0.151378, 0.250768]` / `0.057409 [-0.080313, 0.176147]`.

The mean is positive in all six designs and 80.7--96.3% of paired replicates
are positive, but all percentile intervals cross zero. All ten file-deletion
folds retain a positive HCRD gain:
`[0.102994, 0.139720]` AP for Falkor to MESOSCOPE and
`[0.051747, 0.105754]` in the reverse direction.

The supplementary equal-dimensional sensitivity analysis compares HCRD-8+Q
with a Gaussian smoothing/derivative/curvature bank of the same 2,847-variable
width. Expected HCRD-minus-control AP differences are -0.010832 (95% CI
`[-0.059388, 0.040455]`) for Falkor to MESOSCOPE and +0.117495
(`[0.062431, 0.165414]`) in the reverse direction. Holm-adjusted values are
0.765323 and 0.000400. These target-feature intervals condition on the fitted
source models; see `docs/ms_metrics_e2_matched_capacity.md`.

### LC--MS E5: conditional Pttime residual-error triage

E5 pools Falkor and MESOSCOPE for training and applies each fixed E2
representation without refitting to the 365 unambiguous Pttime features that
the source model had already selected. The official archive contains both ion
modes; the published Pttime feature-box schema uses the 52 POS files, as
recorded before target feature extraction.

```bash
python -m pip install -e ".[lcms]"
python experiments/run_pttime_e5.py extract-target --help
python experiments/run_pttime_e5.py fit-evaluate --help
python experiments/analyze_pttime_review_utility.py
```

Expected primary results are HCRD-8+Q AP-bad 0.508535 versus qscore 0.039092
(difference 0.469442, paired class-stratified 95% CI
`[0.248934, 0.676224]`) and HCRD-1+Q 0.284522 (difference 0.224013,
`[0.070727, 0.354447]`). At 1%, 5%, and 10% review budgets, HCRD-8 retrieves
4, 7, and 8 of 17 bad features, versus 0, 0, and 1 for qscore. The claim is
conditional on the source-model-selected population; it does not estimate
full-population recall, calibration, or FDR. See `docs/pttime_e5_protocol.md` and
`reports/pttime_e5.md`.

### LC--MS E6: external TARDIS/FAME stress test

E6 freezes a pooled Falkor+MESOSCOPE learner and applies it without target
fitting to the fully rated positive-polarity FAME stratum. Downloading requires
about 5.67 GB; 7-Zip is required for selective extraction.

```bash
python -m pip install -e ".[lcms]"
python experiments/download_tardis_e6.py --extract
python experiments/run_tardis_e6.py extract \
  --mzxml-dir data/external/tardis_extracted/data_zenodo/scale_analysis/files700 \
  --labels-workbook data/external/tardis_extracted/data_zenodo/quality_analysis/fame_feat_table_pos.xlsx \
  --archive data/external/data_zenodo.rar \
  --output-dir results/tardis_e6/fame_positive --workers 4
python experiments/run_tardis_e6.py evaluate \
  --falkor-dir results/ms_metrics_e2/falkor \
  --mesoscope-dir results/ms_metrics_e2/mesoscope \
  --target-dir results/tardis_e6/fame_positive \
  --output-dir results/tardis_e6/evaluation --bootstrap 20000
```

Expected result on 192 unambiguous targets: qscore AP-bad 0.435323,
HCRD-8+Q 0.507824 (difference +0.072500, 95% CI
`[-0.015515, 0.134451]`), HCRD-1+Q 0.509405, and DOMAIN+Q 0.519280. Both
co-primary superiority rules fail. The archive size/MD5, availability amendment,
compact evaluation JSON, and extraction metadata are included; raw mzXML,
third-party workbooks, per-QC tensors and fitted models are not redistributed.

### CWRU

`experiments/download_cwru.py` downloads the 16 frozen drive-end records and
checks every label, URL, byte count, and hash against
`data/manifests/cwru_manifest.json`.  The public
runner expects the downloaded files under `data/raw/cwru/`.

```bash
python experiments/download_cwru.py
python experiments/run_cwru_study.py
python experiments/run_parallel_runtime.py
python experiments/run_sparse_runtime_recalculation.py
python experiments/run_sparse_parallel_throughput.py
```

Runtime values are hardware-specific.  Exact output digests, not elapsed time,
are the cross-machine reproducibility target.

### QTDB

The salted record split and download hashes are in `data/manifests/`.  The
frozen R2 protocol is `docs/qtdb_confirmation_protocol.md`.

```bash
python experiments/prepare_qtdb.py --partition all
python experiments/run_qtdb_confirmation.py
python experiments/run_qtdb_modern_baseline.py
python experiments/run_qtdb_multilevel_fusion.py
python experiments/run_qtdb_hybrid_fusion.py
```

R3, M1, and M2 are post-lock development analyses and must not be relabelled
as confirmatory.

### XJTU-SY

Download the six official archive parts listed in
`data/manifests/xjtu_sy_manifest.json`, verify their SHA-256 values, and extract
the dataset under `data/raw/xjtu_sy/XJTU-SY_Bearing_Datasets`. Install the RUL
dependency group, then reproduce the feature matrices and retained sequence of
negative experiments:

```bash
python -m pip install -e ".[rul]"
python experiments/run_xjtu_standard_features.py
python experiments/run_xjtu_energy_features.py
python experiments/run_xjtu_rul.py --help
python experiments/run_xjtu_tree_rul.py --help
python experiments/analyze_xjtu_multiseed.py --help
python experiments/analyze_xjtu_energy_indicators.py --help
```

The repository publishes compact fold, comparison, and indicator artifacts,
not the raw waveforms or large derived feature/prediction matrices.

### PRONOSTIA / NASA FEMTO

H1 is an independently frozen transfer test, not a tuning dataset. Download
the official NASA FEMTO archive and verify it against
`data/manifests/pronostia_manifest.json`. Extract `Training_set.zip` and
`Validation_Set.zip` into the paths documented by the runner, then execute:

```bash
python experiments/run_pronostia_health_indicator.py --workers 8
```

Expected outcome: the frozen HCRD indicator fails the primary criterion
(`success: false`); this negative result must reproduce, not be optimized away.

### PPGopt

The public scripts download the seven-subject wearable-PPG dataset and verify
its archive against `data/manifests/ppgopt_manifest.json`. Install the PPG and
comparison dependency groups, then preserve the locked phase order:

```bash
python -m pip install -e ".[ppg,comparisons]"
python experiments/download_ppgopt.py
python experiments/prepare_ppgopt_features.py --phase development --workers 8
python experiments/run_ppgopt_benchmark.py --phase development
python experiments/prepare_ppgopt_features.py --phase validation --workers 8
python experiments/run_ppgopt_benchmark.py --phase validation
python experiments/prepare_ppgopt_features.py --phase confirmation --workers 8
python experiments/run_ppgopt_benchmark.py --phase confirmation
python experiments/analyze_ppgopt_confirmation.py
```

The feature matrix uses the complete eight-level candidate trajectory. The
confirmation set contains only subjects 6 and 7, so its positive result is a
pilot rather than a population estimate.

### PPG-DaLiA

`data/manifests/ppg_dalia_manifest.json` records all official activity-wise
archives. `ppg_dalia_records.json` fixes the 117 available records, their ECG
annotation counts, and the five outer subject folds.

```bash
python -m pip install -e ".[ppg,comparisons]"
python experiments/download_ppg_dalia.py
python experiments/prepare_ppg_dalia_records.py
python experiments/prepare_ppg_dalia_features.py --workers 8
python experiments/run_ppg_dalia_benchmark.py
python experiments/analyze_ppg_dalia_results.py
```

The public aggregate contains every outer-test record and paired
subject-bootstrap intervals. `run_ppg_dalia_motion_gate.py` is intentionally
separate: the gate was conceived after inspecting the outer tests and is not a
confirmatory result.

The separately fixed D3 local-motion augmentation is also post-test
development:

```bash
python experiments/run_ppg_dalia_local_motion.py
```

Its expected outcome is negative: simple local-acceleration concatenation does
not beat P0 and does not materially improve the frozen HCRD models.

### MIMIC PERform diagnostic

The download, record-preparation, feature, and benchmark scripts are retained,
but the current local result is not promoted. It used a different ECG
consensus detector from the official MATLAB benchmark and preceded a correction
to the benchmark remainder-block alignment. Recalculate it before making any
comparison with published PPG-beat results.

### TSB-AD temporal polygon-mass spectrum

Install the TSAD dependency group, download the official archive, and pin the
benchmark code. The downloader verifies the archive SHA-256, all 870 CSV files,
and the exact TSB-AD commit; raw data and third-party code are not redistributed.

```bash
python -m pip install -e ".[tsad]"
python experiments/download_tsb_ad.py
python experiments/analyze_tsb_ad_area_results.py
python experiments/generate_tsb_ad_figures.py
```

The commands above provide a fast audit of saved scores. For a full rebuild,
including A1 direct tuning, A2 temporal processing, A3 signed-component
learning, sealed evaluation, and runtime recalculation, follow
`docs/tsb_ad_reproducibility.md` in a fresh clone. Expected evaluation mean
VUS-PR is `0.34687822104563465` over 350 series. A2 and A3 are retained
negative development stages and must precede evaluation in the protocol order.

For the pre-specified C1 extension, download the verified TSB-UAD Public
archive (only its Yahoo and KDD21 subsets are extracted), freeze the matched
manifest, and only then evaluate:

```bash
python experiments/download_tsb_uad_yahoo.py
python experiments/run_tsb_uad_yahoo_confirmation.py --freeze-only
python experiments/run_tsb_uad_yahoo_confirmation.py --evaluate
python experiments/generate_tsb_uad_yahoo_figure.py
```

Expected primary result on 134 point-anomaly series: HCRD mean AUC-PR 0.528539,
NORMA 0.338286, paired difference 0.190253 with bootstrap interval
[0.129409, 0.250952]. The implementation, official archive/table, exclusion
lists, content-matched manifest, and protocol are hashed before HCRD execution.
See `docs/tsb_uad_yahoo_c1_protocol.md` for the within-Yahoo independence
boundary.

The same download includes the KDD21 subset needed for the cross-source D1
population screen:

```bash
python experiments/run_tsb_uad_kdd21_confirmation.py --freeze-only
```

Expected outcome is an intentional pre-score abort: 145 unique matches but only
5 official point-anomaly files, below the protocol minimum of 10. No
`frozen_configuration.json` or HCRD score should be produced. The population
manifest and integrity hashes are retained under `results/tsb_uad_kdd21_d1/`.

### UCR U1 morphology classification

Install the time-series classification dependencies. The discovery stage uses
the complete HCRD component/structure collection and strong raw/wavelet
MiniRocket controls:

```bash
python -m pip install -e ".[tsc]"
python experiments/prepare_ucr_u1_manifest.py
python experiments/run_ucr_u1_benchmark.py --stage discovery --workers 8
python experiments/run_ucr_u1_structure_trees.py --stage discovery --workers 8
python experiments/compute_ucr_u1_training_gate.py --stage discovery --workers 8
python experiments/analyze_ucr_u1_discovery.py
```

Expected result: no positive discovery subgroup satisfying the minimum
ten-dataset rule. Do not run the confirmation stage unless a future,
pre-specified representation produces a positive rule; its 48 outcomes remain
uninspected in this release.

## Artifact provenance

- `docs/experiment_protocol.md` specifies synthetic and benchmark designs.
- `docs/protocol_amendments.md` records protocol versions.
- `docs/reproducibility_audit.md` records byte-identical reruns and hashes.

Experiment scripts write results; published CSV/JSON files should not be
hand-edited. Rebuild `release_manifest.json` after changing packaged files:

```bash
python scripts/build_release_manifest.py
python scripts/verify_release.py
```
