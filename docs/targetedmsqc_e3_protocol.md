# E3 prospective protocol: TargetedMSQC run-transfer confirmation

Status: **frozen before any HCRD score or outcome comparison was computed**.

Protocol identifier: `hcrd-e3-targetedmsqc-v1`  
Freeze date: 2026-08-25  
Source repository: <https://github.com/shadieshghi/TargetedMSQC>  
Frozen source commit: `5439ba9f102d043241ccb8207e7ec9f1a35ebc63`

## 1. Confirmatory question

Does the already selected full eight-level HCRD representation add useful
information to the 52 published TargetedMSQC quality metrics when predicting an
expert's `ok` versus `Flag` decision for a transition in a previously unseen
LC--MS/MS run from the same targeted panel?

This is a deliberately different analytical regime from E1 and E2: targeted
proteomics, light/heavy transition pairs, four `.wiff` runs, manually annotated
transition quality, and the official TargetedMSQC comparator features.  It tests
same-panel **new-run transfer**.  It does not claim transfer to unseen peptide
panels, matrices, laboratories, or instrument families.

## 2. Data, unit, and deterministic matching

The only data source is the CSF panel distributed at the frozen commit:

- `Training/CSF_Biomarkers_training_annotated.csv` supplies expert labels;
- `Features/features.csv` supplies the published numerical QC metrics;
- `Peak_boundary/CSF_Biomarkers.csv` supplies peak-group boundaries;
- `Chromatograms/CSF_Biomarkers.tsv` supplies light/heavy chromatograms.

The analysis unit is one annotated transition in one run, identified by
`FileName`, `PeptideModifiedSequence`, `PrecursorCharge`, `FragmentIon`, and
`ProductCharge`.  A row is eligible only if it has one official feature row, one
peak boundary, and both a light and a heavy chromatogram.  Duplicate or
ambiguous keys and rows with fewer than eight samples in either required window
are excluded by deterministic rules and counted.  Eligibility is independent
of the expert outcome.  No row is selected using HCRD values.

Primary label: `ok = 1`, `Flag = 0`.  Average precision (AP) therefore measures
reliable approval of acceptable transitions.  Flag-positive AP is reported only
as a secondary operational view.

## 3. Frozen signal windows and representations

For each isotope chromatogram, let `[a,b]` be the published peak-group boundary
and `w=b-a`.

- short window: `[a,b]`;
- context window: `[a-w,b+w]`, clipped to the available chromatogram.

The frozen E1 feature extractor is applied separately to light and heavy
signals and their feature vectors are concatenated in that order.  The full
eight-level representation contains the resampled signal, all decomposition
channels, convex-lobe geometry, and area/energy-like descriptors; area is not
used as the main representation.

The candidate feature sets are fixed as follows:

| Name | Contents | Role |
|---|---|---|
| `qc52` | all 52 numerical TargetedMSQC metrics | primary comparator |
| `qc52_raw` | `qc52` plus light/heavy raw short/context vectors | signal-access control |
| `qc52_hcrd1` | `qc52` plus one-level HCRD | depth ablation |
| `qc52_hcrd8` | `qc52` plus full eight-level HCRD | **primary method** |
| `qc52_geometry` | `qc52` plus multilevel lobe geometry | secondary mechanism ablation |
| `qc52_area` | `qc52` plus multilevel area/energy descriptors | secondary mass ablation |

`qc52_hcrd8` was selected before this dataset was scored because the complete
multilevel representation was the stable positive configuration in E1/E2.
Neither geometry nor area may replace it as the primary method after inspection
of E3 results.

## 4. Learner and run-transfer validation

Every feature set uses the identical, untuned pipeline:

1. median imputation learned on the training runs;
2. standardization learned on the training runs;
3. L2 logistic regression with `C=1`, `class_weight="balanced"`,
   `solver="liblinear"`, `max_iter=5000`, and seed `20260825`.

The primary validation is leave-one-run-out cross-validation over the four
annotated `.wiff` files.  All preprocessing and fitting occur inside each fold.
The four held-out prediction blocks are pooled exactly once for the primary AP.
Per-run AP is also retained.  Because peptides recur across runs, this protocol
estimates transfer to a new acquisition of the same panel rather than new-peptide
generalization.

A secondary five-fold peptide-group cross-validation uses the lexicographically
sorted peptide list assigned round-robin to folds.  It estimates new-peptide
transfer across the same four runs, but it is not confirmatory because run-level
acquisition signatures can recur between training and test sets.

## 5. Confirmatory estimand and inference

The primary estimand is

`AP(qc52_hcrd8) - AP(qc52)`

on pooled leave-one-run-out predictions.  Its 95% interval and two-sided
bootstrap p-value use 10,000 paired two-way cluster bootstrap replicates.  Each
replicate independently resamples the four runs and the eligible peptides, and
the product of the two cluster multiplicities is used as the row weight.  This
preserves pairing between methods and recognizes both dependence axes.  The
small number of run clusters is an explicit inferential limitation; the
per-run direction is therefore a co-primary robustness check, not a replacement
for the interval.

Prospective success requires all of:

1. pooled AP difference `qc52_hcrd8 - qc52 > 0`;
2. the two-way cluster-bootstrap 95% lower bound is above zero;
3. `qc52_hcrd8` has higher AP than `qc52` in at least three of four held-out runs;
4. pooled `AP(qc52_hcrd8) > AP(qc52_hcrd1)`.

This is one confirmatory contrast, so no multiplicity correction is applied to
it.  All other comparisons are exploratory and labelled as such.

## 6. Predeclared secondary analyses

- pooled and per-run ROC AUC;
- flag-positive AP;
- peptide-group cross-validation;
- `qc52_raw`, one-level, geometry-only, and area-only ablations;
- performance within score-independent expert-note strata:
  morphology-related flags (`shoulder`, `jagged`, `bimodal`, `tailing`, or
  `high background`) versus ratio/order-related flags (`inconsistent`);
- counts and reasons for all deterministic exclusions.

These analyses may explain *where* the representation helps or fails, but they
cannot rescue a failed primary endpoint.

## 7. Leakage guards and interpretation limits

- labels are used only to fit training folds and evaluate held-out predictions;
- official-feature imputation and scaling are fold-local;
- no hyperparameter, HCRD depth, signal window, representation, or subgroup is
  selected using E3 outcomes;
- no transition from a held-out run enters that fold's training set;
- the source commit, file hashes, software versions, exact exclusions,
  predictions, and bootstrap seed are recorded in the result artifact.

A positive result would be evidence that multilevel convex-lobe morphology adds
information beyond a strong domain-specific QC system in its intended
same-panel run-transfer setting.  A null result would be informative: it would
bound HCRD's scope and test whether the positive E1/E2 effect depended on the
earlier feature-finding regime.
