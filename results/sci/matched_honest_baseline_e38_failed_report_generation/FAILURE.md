# Incomplete E38 execution

The frozen statistical calculation completed and wrote `trial_scores.csv` and
`summary.csv`. Report generation then failed because the signal name was
passed twice to Python's `str.format`. No manifest was written.

The frozen configuration and partial outputs are retained for transparency.
E38r1 fixes only that formatting call, uses a new driver hash and output
directory, and reruns the same deterministic statistical calculation.
