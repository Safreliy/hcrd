# qscore implementation sensitivity

Protocol fixed on 2026-08-26 after the E2 fidelity concern was raised and
before computing this sensitivity result.

## Question

Does the E2 HCRD gain depend on the particular independent qscore
reimplementation or on using only its two median summaries?

## Fixed variants

Recompute qscore from the same global mass-feature boxes using the published
beta-shape and residual-SNR formula. Evaluate four fixed summaries:

1. `Q-current`: median SNR and correlation from EICs with at least 8 points;
2. `Q-min5`: the same two medians with the authors' at-least-5-points cutoff;
3. `Q-author5`: median SNR, median correlation, maximum SNR, maximum
   correlation, and second-largest correlation;
4. `Q-multi7`: median, 90th percentile, and maximum of SNR and correlation,
   plus the available-file fraction.

For every variant compare qscore alone with HCRD-8 features plus exactly the
same qscore summary. Use the unchanged standardized, class-balanced L2
logistic learner, both fixed transfer directions, and 10,000 paired target
feature bootstrap replicates with seed `20260826`.

On Falkor, report Pearson and Spearman correlations against every compatible
author-supplied column in `features_extracted.csv`. Author qscore outputs are
not available for MESOSCOPE; no bit-for-bit author-baseline claim is made for
that study.

## Interpretation

The analysis is a post-frozen implementation sensitivity. It tests formula
cutoff and aggregation choices while retaining a common global-box extraction.
It cannot reproduce the authors' unavailable per-detected-peak MESOSCOPE
boundaries.

