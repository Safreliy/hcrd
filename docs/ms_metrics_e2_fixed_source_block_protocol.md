# E2 fixed-source target RT-block sensitivity

Protocol fixed on 2026-08-26 after the original E2 analysis and the external
audit, and before computing results from this sensitivity.

## Estimand

The primary E2 question is conditional: given the source-study model fitted
under the locked pipeline, does it transfer to the other study without target
refitting? The original paired bootstrap resampled target mass features as if
they were independent. This analysis keeps the fitted source models fixed but
replaces target-feature resampling by retention-time (RT) block resampling.
It therefore tests the stated conditional estimand under a transparent model
of local dependence among overlapping or coeluting features.

The public labels contain no compound or adduct identifier. RT blocks are a
dependence sensitivity, not a claim of chemical independence.

## Design

For each target study, assign every unambiguous feature to

`floor(midpoint_retention_time / block_width)`.

For each transfer direction and bootstrap replicate:

1. sample the nonempty target RT blocks with replacement, drawing as many
   blocks as were observed;
2. include every feature in each sampled block, with multiplicity;
3. compute average precision for the saved qscore and HCRD-8+Q source models;
4. record the paired difference HCRD-8+Q minus qscore.

The source model, target scores, representation, feature filter, and learner
are identical to E2. The same target resample is paired across methods. Use
10,000 replicates for each of 30-, 60-, and 120-second block widths, master
seed `20260826`, and percentile 95% intervals. The 60-second design is primary
within this sensitivity; the other widths check scale dependence.

## Interpretation

A positive interval supports the conditional no-refit transfer claim after
allowing arbitrary dependence within local RT blocks. This result does not
propagate source-study sampling or model-fitting uncertainty; those are
addressed separately by the source-refit RT-block analysis. Neither analysis
recovers unavailable compound/adduct clusters or represents new laboratories.
