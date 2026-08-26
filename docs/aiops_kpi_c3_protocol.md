# C3: independent AIOps2018 KPI confirmation

## Design

The public 2018 AIOps KPI Anomaly Detection repository contains two distinct
labelled releases.  Its preliminary `train.csv` has 26 KPIs; the finals
`phase2_train.csv` has 29 different KPI identifiers.  The data are real
production KPIs from large Internet companies and were labelled by domain
experts.

The sparse-local-transient class is inherited unchanged from WSD C2:

\[
R/n\le0.005,\qquad M/n\le0.01,
\]

where (R) is the longest anomaly run and (M) is the anomalous-sample count.
This definition existed before any AIOps label was inspected.  It selects
15/26 preliminary KPIs and 13/29 phase-2 KPIs.

## Development and locked transfer

The existing fixed `hcrd_temporal_candidate_scores` family was evaluated only
on the 15 preliminary-class KPIs.  Mean per-series AUC-PR selected `a2_area_sr`:

1. compute the eight-level HCRD area-density representation;
2. robustly fuse levels by the fixed maximum rule;
3. regard that geometric mass density as a new time series;
4. apply the fixed spectral-residual operator with amplitude window 100.

This candidate reached development mean AUC-PR 0.260335 versus 0.240314 for
direct HCRD.  It is frozen before any phase-2 score.  This is precisely a test
of whether temporal analysis of the HCRD polygon-mass series adds value.

## Confirmation endpoint and leakage gate

The primary endpoint is the paired mean per-series AUC-PR difference between
the selected HCRD-area-SR candidate and the identical spectral-residual
operator applied directly to the raw KPI, on the 13 phase-2 KPIs in the fixed
class.  This comparator isolates the value of the HCRD representation rather
than merely the value of spectral residual.  Direct HCRD, raw absolute
deviation, AUC-ROC, and all 29 phase-2 KPIs are secondary.

The phase order is enforced in code:

1. freeze hashes, identities, class membership, candidate, comparator, metric,
   and inference before any phase-2 score;
2. evaluate raw-signal baselines and write a completion gate;
3. refuse HCRD execution until the gate exists;
4. analyse only the complete one-to-one merge.

The primary comparison reports a 50,000-draw paired bootstrap interval, an
exact (2^{13}) sign-flip randomisation test for the paired mean under the
symmetry null, paired t-test, Wilcoxon test, exact sign test, and
wins/ties/losses.  No runtime claim is made.

## Boundary

This is a genuine cross-dataset and independent-release transfer, but it has
only 13 primary KPIs.  A positive result confirms value over the domain-aligned
raw spectral-residual comparator; it does not by itself establish superiority
over every modern KPI detector.  A negative result falsifies the proposed
hybrid transfer and must be retained.
