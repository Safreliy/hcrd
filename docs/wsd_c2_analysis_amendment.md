# WSD C2 analysis export amendment

The C2 runner froze two distinct rules before HCRD execution:

1. the primary comparator is the largest mean primary VUS-PR among official
   wrappers complete on all 71 primary series; and
2. comparator freezing is allowed once at least four wrappers are complete.

Four wrappers completed all 86 series: POLY, SR, Sub-PCA, and Sub-IForest.
MMPAD, KShapeAD, MatrixProfile, and SAND were incomplete; SAND additionally
raised an official-wrapper sampling error on short training regions.  Their
partial metrics and failures remain in the archive and are not counted as
losses.  `comparator_frozen.json` selected POLY before the first WSD HCRD score.

The original export function unnecessarily required all eight wrappers to be
complete even though this contradicted its own frozen eligibility rule.  The
post-execution exporter `experiments/analyze_wsd_complete_baselines.py` changes
no endpoint, population, comparator, seed, statistical test, or multiplicity
rule.  It imports those functions from the hash-frozen runner and exports only
the four methods recorded as eligible in `comparator_frozen.json`.
