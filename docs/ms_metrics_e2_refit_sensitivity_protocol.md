# E2 source-refit and acquisition-file sensitivity

Protocol fixed on 2026-08-26 after the original frozen E2 and matched-capacity
analyses and before computing any result from this sensitivity analysis.

## Scope

The original E2 intervals resampled target mass features conditional on each
fitted source model. This secondary analysis asks whether its AP contrasts are
robust to source-model refitting, local retention-time dependence, and removal
of acquisition-file groups. It does not replace or retroactively alter the
pre-specified E2 endpoint.

The public label schema has no compound or adduct identifier. Consequently,
no result is described as a compound-cluster bootstrap. Retention-time blocks
are a transparent dependence sensitivity for overlapping/coeluting mass
features, not a claim of chemical independence.

## Part A: paired source-refit RT-block bootstrap

For each study, assign every unambiguous mass feature to

`floor(midpoint_retention_time / block_width)`.

For each replicate and transfer direction:

1. resample the nonempty source blocks with replacement and concatenate all
   member features;
2. refit the unchanged standardized, class-balanced L2 logistic learner;
3. resample the nonempty target blocks independently with replacement;
4. compute target AP on the resampled target features.

The same source and target resamples are paired across qscore and HCRD-8+Q.
Use 1000 replicates at the primary 60-second block width and 300 replicates at
each 30- and 120-second sensitivity width, master seed `20260826`, and ten
workers unless overridden. Report percentile 95% intervals for HCRD-8+Q minus
qscore. The equal-dimensional Gaussian comparison remains the separately
reported matched-capacity analysis; it is not a co-primary endpoint here.

The feature representations are fixed in Part A; the fitted source pipeline is
not. This isolates model- and mass-feature-population uncertainty.

## Part B: acquisition-file delete-group full representation refit

Within each study, permute the acquisition files once with seed `20260826` and
split them into ten groups as evenly as possible. For fold `g`, remove group
`g` from both studies and recompute from the saved per-file cubes:

- median qscore;
- HCRD-8 median, 90th percentile, maximum, and availability fraction;
- the appended recomputed qscore values.

Refit the unchanged learner in both transfer directions and report AP and the
HCRD-8+Q minus qscore contrast. This is a deterministic delete-group
sensitivity, not a bootstrap confidence interval. Report the full fold range
and whether the contrast retains its sign in every fold. The reference runner
evaluates five folds concurrently by default; results and fold ordering are
independent of worker count.

## Interpretation

Part A addresses source refitting and local target/source dependence. Part B
checks whether the central result is driven by a small acquisition-file group
while repeating representation aggregation and model fitting. Neither part
captures unidentified compound/adduct clusters or sampling from additional
laboratories.
