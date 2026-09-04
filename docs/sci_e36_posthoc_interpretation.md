# E36 result note: why the empty-set gate failed

This note was written after the frozen E36 evaluation. The frozen protocol and
`all_pass: false` decision remain unchanged. A later external audit found one
trial whose exclusion boundaries were equal. Strict sign inversion makes that
trial empty, not a nonempty zero-width interval. The corrected row and manifest
record this change; coverage and all headline ranges are unchanged.

## Result

Known-scale SCI covered the true transition in 97.70% to 98.04% of the 5,000
responses in each of the 16 benchmark cells. Simultaneous contrast coverage
was 95.24% to 96.18%. Both pre-specified 0.94 implementation thresholds passed.

The gate named `zero_empty_sets` failed. Empty-set probabilities ranged from
0.02% to 2.28%. The largest value occurred for the jump signal with the
`Beta(4,8)` design at `n=1000`. Its 95% Wilson interval was 1.90% to 2.73%.

## Why this is not a contradiction

SCI can be empty when the observed data contain simultaneous evidence that no
single convex-to-concave transition is compatible with the response. Even
when the model is true, a confidence procedure can make this error on its
failure event. The theorem states

`P{SCI is empty} <= alpha`

when the design-level transition target is nonempty. It does not state that SCI
is never empty. Here `alpha=0.05`, and every observed empty-set frequency was below
0.05. The upper end of every cellwise 95% Wilson interval was also below 0.05.

Thus the zero-empty requirement was stricter than the theorem. It remains a
failed frozen diagnostic and must not be relabelled as passed after seeing the
data. A future independently frozen audit should test the theorem-aligned
condition, such as an upper confidence bound on the empty-set probability,
rather than require zero failures in 5,000 trials.

## Publication wording

Use this sentence:

> The pre-specified zero-empty diagnostic failed because SCI was empty in up
> to 2.28% of trials. This is below the 5% theorem allowance and represents a
> confidence-set failure event, not a contradiction of the coverage result.

Do not write that every E36 gate passed.
