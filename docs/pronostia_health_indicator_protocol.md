# Frozen PRONOSTIA health-indicator confirmation H1

Status: prospectively fixed before downloading or inspecting PRONOSTIA signal
outcomes in this project.

Date fixed: 2026-08-24.

## Motivation and independence

XJTU-SY is exhausted for method development. It showed that direct RUL
regression does not benefit from HCRD polygon mass, but the horizontally sensed
RMS-envelope level-3 mass had positive Spearman association with life progress
on all 15 bearings. H1 transfers that one exact coordinate without reselection
to the independent NASA FEMTO/PRONOSTIA run-to-failure benchmark.

## Frozen indicator

For every vibration acquisition:

1. use the horizontal acceleration channel only;
2. partition its samples into 256 consecutive blocks and take block RMS;
3. compute a six-level centred sparse HCRD hierarchy with numerical tolerances
   `atol=1e-12`, `rtol=64*eps`;
4. define the HCRD indicator as `log1p` of exact level-3 polygon mass.

If a signal length is not divisible by 256, `numpy.array_split` defines the
blocks. "Level 3" is one-based; a hierarchy terminating earlier contributes
zero. No orientation, smoothing, level, channel, or feature is selected on
PRONOSTIA.

## Task and comparators

The task is unsupervised health-indicator construction, not RUL regression.
For every complete run-to-failure bearing, evaluate association with normalized
life progress using Spearman trendability. Fixed comparators are horizontal raw
RMS, variance, kurtosis, and 0--1 kHz spectral band power. The primary contrast
is HCRD minus raw RMS in absolute Spearman trendability, macro-averaged across
bearings. Secondary outcomes are positive-direction consistency, median
trendability, and the other fixed comparators.

Uncertainty uses a paired bearing bootstrap. H1 succeeds only if (a) the HCRD
indicator has positive correlation on every eligible complete trajectory and
(b) the 95% interval for its absolute-trendability difference versus raw RMS is
above zero. Comparison with published methods is contextual unless their code
and exact preprocessing can be reproduced.

## Data and exclusions

Use the public FEMTO Bearing dataset from the NASA Prognostics Data Repository.
Only trajectories documented as complete run-to-failure acquisitions are
eligible. Truncated challenge-test trajectories, temperature-only files,
corrupted acquisitions, and bearings with fewer than 20 acquisitions are
excluded by rule and listed in metadata. Raw data are never committed.

This protocol is a genuine independent confirmation only for the single frozen
HCRD indicator above. Any PRONOSTIA-driven fusion, level change, smoothing, or
fault-onset rule is a new development stage and requires another dataset.

## Recorded outcome

H1 failed both success conditions on 17 complete trajectories and 24,889
acquisitions. The frozen HCRD indicator was positively associated with life
progress on 9/17 bearings, not all 17. Its median absolute Spearman
trendability was `0.4485`, versus `0.5175` for horizontal RMS. The mean paired
difference in absolute trendability was `-0.08384`, with a 95% bearing-bootstrap
interval `[-0.17333,+0.00309]`; 8/17 bearings improved. The XJTU all-positive
observation therefore did not transfer and is dataset-specific evidence, not a
general health-indicator property.
