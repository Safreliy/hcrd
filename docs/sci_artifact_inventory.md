# Shape-contrast inversion artifact inventory

This repository snapshot contains the auditable artifact package for
shape-contrast inversion (SCI), the current publication track developed from
the HCRD/HCT research programme.

## Included artifacts

- `src/shapecontrast/`: standalone matrix-free SCI, known- and unknown-scale
  bands, bounded-heteroskedastic sensitivity, and exact replicate-curve
  Student bands.
- `src/hcrd/shape_inflection_confidence.py`: multiscale chord contrasts,
  simultaneous Gaussian bands, and confidence-set inversion used by the
  frozen E33--E35 experiments.
- `src/hcrd/noise_scale_confidence.py`: one-sided scale bounds for unknown iid
  Gaussian noise and an unrestricted mean vector.
- `src/hcrd/heteroskedastic_scale_confidence.py`: a Gaussian scale envelope
  under a declared max-to-mean variance ratio.
- `tests/test_*scale_confidence.py` and
  `tests/test_shape_inflection_confidence.py`: implementation and seeded
  coverage checks.
- `theory/hct/`: theorem files frozen with the earlier experiments.
- `theory/sci/`: revised statements, proof audits, the general contrast-margin
  theorem, and the replicate-curve extension.
- `docs/hct_e3*.md`: frozen protocols and the frontier/priority audit.
- `experiments/hct/`: Python and R runners for E33--E35 and the publication
  figure generator.
- `results/hct/`: frozen configurations, trial-level scores, summaries,
  manifests, and compact real-data outputs.
- `reports/hct/`: result interpretation, passed and failed gates, and claim
  boundaries.
- `paper/sci/figures/`: publication-labelled PDF and PNG figures generated
  only from the included compact results.
- `experiments/sci/` and `results/sci/`: the million-point scaling audit, the
  80,000-response E36 coverage audit, the replicated DNase E37 analysis, and
  the matched honest E38r1 comparison.
- `data/external/dnase/`: the small public DNase CSV, its source URL, and its
  SHA-256 checksum.

The experiment manifests record configuration, code, result, and environment
hashes. `release_manifest.json` independently hashes every distributed SCI
file together with the earlier HCRD release.

## Deliberately excluded artifacts

Four E33 response-matrix shards (`observations_shard_0.csv` through
`observations_shard_3.csv`, about 45 MiB in total) are omitted because they are
deterministically regenerated from the frozen seeds. Their SHA-256 digests are
preserved in
`results/hct/shape_contrast_hybrid_e33_confirmation/pre_manifest.json`.
All derived HCT/SCI scores, comparator outputs, trial-level evaluation rows,
and summaries are included.

Third-party R runtimes, package sources, and comparator code are not
redistributed. The exact versions and source or DESCRIPTION hashes are frozen
in the E33 and E35 manifests. In particular, the comparison used `Sshaped`
1.2 and `ShapeChange` 1.5; the LIDAR runner also used `SemiPar` 1.0-4.2.
Restoring those dependencies is required only to refit the external R methods,
not to inspect the frozen outputs or regenerate the publication figures.

## Integrity and scope

The simulation rows verify that the implementation behaves as predicted under
the frozen designs; they are not proofs. The mathematical guarantees are
conditional on the assumptions stated in `theory/hct/`, including independent
Gaussian errors and, for the E35 extension, a correct declared variance-ratio
bound. The E37 extension allows arbitrary dependence and unequal variance
inside a run, but the run-level curves must be independent Gaussian replicates
with a common mean and covariance. The current package does not claim general
non-Gaussian or arbitrary between-run dependence validity.
