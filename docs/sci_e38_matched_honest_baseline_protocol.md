# E38r1 protocol: SCI versus a conservative pointwise-band baseline

The first E38 execution completed the statistical calculations and wrote the
two CSV files, but stopped while formatting the Markdown report because one
keyword was passed twice to `str.format`. That incomplete output is preserved
under `results/sci/matched_honest_baseline_e38_failed_report_generation/`.
Revision r1 changes only the report-formatting line and output identifier. The
statistical configuration and all frozen checks below are unchanged. The text
below includes a post-audit clarification of what the frozen PBP code computes;
the original protocol is archived with the frozen sources.

## Purpose

The `ShapeChange` bootstrap is an important published comparator, but it uses
a smoother model than SCI. E38 adds a second comparator with a finite-sample
guarantee under the same known-scale Gaussian observation model.

## Comparator

The pointwise-band baseline (PBP) first builds simultaneous Bonferroni
intervals for all sampled mean values. It separately computes the longest
prefix compatible with a discretely convex vector in the box and the earliest
suffix compatible with a discretely concave vector in the box. These two
witness vectors need not join at one transition value. PBP is therefore a
conservative discrete split relaxation, not an exact projection onto the same
continuous function class as SCI.

This baseline is related to the general confidence-region approach of Davies,
Kovac and Meise (2009), but it is our simpler implementation. It must not be
called their official algorithm. Its finite-sample coverage follows because
the true sampled mean belongs to the pointwise box with probability at least
`1-alpha`. The guarantee applies on the domain passed to PBP. The frozen E38
comparison uses `[x_1,x_n]` for both the target and both methods; the public
function can instead receive a wider declared `[A,B]` and then extends fully
concave or fully convex boundary cuts to `A` or `B`. The comparison is against
this deliberately simple conservative baseline, not against the strongest
possible same-band projection.

## Frozen experiment

- the four published Feng et al. signals: cusp, onset, jump, and logistic;
- uniform and `Beta(4,8)` quantile designs;
- `n` equal to 500 and 1,000;
- 200 fresh Gaussian responses in each of 16 cells;
- known noise standard deviation 0.1;
- confidence level 95%;
- seed 20262211;
- SCI separation multipliers 1, 2, and 4;
- the same observed design range for both methods;
- up to four worker processes, with a fixed independent seed for each cell.

## Frozen checks

1. All 16 cells are retained.
2. Coverage is at least 0.90 for each method in every cell. This is a coarse
   implementation alarm, not the nominal target.
3. Among each method's nonempty outputs, SCI median width is not more than 0.01
   larger than PBP median width in any cell.
4. SCI reduces that conditional median width by at least 10% in at least eight
   cells.
5. Weak logistic cells remain in the table even if both methods return almost
   the full design range.

The experiment compares information at matched finite-sample validity. It
does not establish a universal width ordering between the two methods. A
post-audit uncertainty supplement reports Wilson intervals for coverage and
paired bootstrap intervals for the median-width reduction without changing
the frozen trial data.

## Post-audit target and width checks

The frozen driver checked coverage of the generating point `m0=0.3`. A
deterministic post-audit analysis computes the full design-identified target
once per cell and checks the saved interval endpoints. Point and full-target
coverage agree in all 6,400 saved method rows, so no reported coverage value
changes.

The original width comparison uses each method's nonempty outputs. SCI empty
rates range from 0 to 0.040 and PBP empty rates from 0 to 0.005. A sensitivity
analysis on trials where both methods are nonempty retains the 19.4%--75.7%
reduction range over the 12 informative cells. The Beta-design jump cell at
`n=500` changes from 40.0% to 33.3%; the main conclusion is unchanged. The
complete results and hashes are in `identified_target_and_width_*` under the
E38r1 result directory.
