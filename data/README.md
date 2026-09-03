# Data policy

No large third-party raw waveform is committed to this repository. One small
public table, the R `DNase` example used for replicate-curve SCI, is included
under `external/dnase/`. Its official documentation, mirror URL, and SHA-256
checksum are recorded in `source_manifest.json`. The table remains under its
source terms and is not relicensed by this repository.

- CWRU Bearing Data Center records are retrieved by
  `experiments/download_cwru.py`; the frozen URLs, labels, byte counts, and
  SHA-256 values are in `manifests/cwru_manifest.json`.
- QT Database records are retrieved by `experiments/prepare_qtdb.py`; the
  immutable salted pilot/locked split and download manifests are in this
  directory.
- XJTU-SY run-to-failure bearing archives are obtained from the official
  dataset page. The six archive hashes, extracted file counts, and expected
  directory layout are in `manifests/xjtu_sy_manifest.json`; raw archives and
  extracted CSV files are not redistributed.
- PRONOSTIA is obtained as the NASA FEMTO Bearing archive. The official URL,
  outer/nested archive hashes, 17 complete-trajectory file counts, and H1 data
  boundary are in `manifests/pronostia_manifest.json`.
- PPGopt is downloaded by `experiments/download_ppgopt.py`; its archive hash
  and the development/validation/confirmation subject split are in
  `manifests/ppgopt_manifest.json`.
- PPG-DaLiA activity-wise archives are downloaded by
  `experiments/download_ppg_dalia.py`. Source hashes are in
  `manifests/ppg_dalia_manifest.json`, while the 117-record inventory and
  nested subject folds are in `manifests/ppg_dalia_records.json`.
- MIMIC PERform source and exported-record hashes are in
  `manifests/mimic_perform_manifest.json` and
  `manifests/mimic_perform_records.json`.
- The UCR U1 population, resource exclusions, and SHA-256 discovery/confirmation
  assignment are recorded in `manifests/ucr_u1_manifest.json`.
- TSB-AD-U is downloaded by `experiments/download_tsb_ad.py`, which verifies
  archive SHA-256 `0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961`,
  870 extracted CSV files, and benchmark-code commit
  `e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48`.
- The Mueller et al. EIC collection is downloaded by
  `experiments/download_lcms_eic_subset.py`; the local RData/flattened feature
  cache is intentionally excluded.
- The Kumler et al. Falkor and MESOSCOPE HILIC archives are downloaded by
  `experiments/download_ms_metrics_e2.py`. Official archive hashes, study IDs,
  file counts, and the pinned source-repository commit are recorded in
  `manifests/ms_metrics_e2_manifest.json`; raw mzML and derived per-file arrays
  are not redistributed.

Users are responsible for checking the current terms and citations of each
source dataset.  Generated experiment outputs under `results/` do not contain
the full raw waveforms.
