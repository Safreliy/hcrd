# E38r1 protocol: SCI versus an honest confidence-region projection

The first E38 execution completed the statistical calculations and wrote the
two CSV files, but stopped while formatting the Markdown report because one
keyword was passed twice to `str.format`. That incomplete output is preserved
under `results/sci/matched_honest_baseline_e38_failed_report_generation/`.
Revision r1 changes only the report-formatting line and output identifier. The
statistical configuration and all frozen checks below are unchanged.

## Purpose

The `ShapeChange` bootstrap is an important published comparator, but it uses
a smoother model than SCI. E38 adds a second comparator with a finite-sample
guarantee for the same sampled convex-to-concave shape class.

## Comparator

The pointwise-band projection (PBP) baseline first builds simultaneous
Bonferroni intervals for all sampled mean values. For each possible split of
the ordered design, a linear program checks whether some vector inside this
confidence box is discretely convex before the split and discretely concave
after it. The output is the range of all feasible splits.

This baseline is related to the general confidence-region approach of Davies,
Kovac and Meise (2009), but it is our simpler implementation. It must not be
called their official algorithm. Its finite-sample coverage follows because
the true sampled mean belongs to the pointwise box with probability at least
`1-alpha`.

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
3. SCI median width is not more than 0.01 larger than PBP median width in any
   cell.
4. SCI reduces median width by at least 10% in at least eight cells.
5. Weak logistic cells remain in the table even if both methods return almost
   the full design range.

The experiment compares information at matched finite-sample validity. It
does not establish a universal width ordering between the two methods.
