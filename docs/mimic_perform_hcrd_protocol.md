# Frozen MIMIC PERform replication protocol

Status: frozen before any reference-beat extraction or HCRD scoring.

Author: Saveliy Baturin, Independent Researcher

## Purpose

This is an independent replication of the PPGopt result on the standard
PPG-beats MIMIC PERform benchmark. The official Training and Testing datasets
contain 200 disjoint ten-minute records each (100 adults and 100 neonates per
split), sampled at 125 Hz. The official Testing data remain locked until a
single rule has been selected on Training.

The primary unproved hypothesis is that a learned trajectory through the full
HCRD hierarchy identifies physiological pulse waves across age groups more
reliably than single-scale peak prominence.

## Immutable split

Within each adult (`a`) and neonate (`n`) stratum of the official Training set,
records are sorted by SHA-256 of their record ID. The first 80 in each stratum
are development and the remaining 20 validation. All 200 official Testing
records are confirmation. The exact assignment is stored in
`data/manifests/mimic_perform_records.json` before outcome generation.

## Reference ECG beats and quality

Two independent Python implementations detect R peaks: WFDB XQRS and
NeuroKit's default ECG detector. Missing ECG samples are linearly interpolated.
An XQRS beat is retained only when a NeuroKit beat lies within 150 ms.

Following the logic of PPG-beats, the sample-level ECG quality around a retained
XQRS beat is high only when that beat and its adjacent XQRS beats all agree with
NeuroKit. The high-quality interval extends from the previous to the next XQRS
beat. Non-finite samples and flat extrema lasting more than 200 ms are invalid.
Results report the fraction of signal retained and detector agreement. A
sensitivity analysis uses NeuroKit alone without the consensus mask.

As in PPG-beats, ECG reference beats are shifted to compensate pulse-transit
delay separately for each evaluated PPG detector. For successive blocks of 300
ECG beats, test lags from -10 to +10 s in 20 ms increments and choose the lag
maximizing the number of reference beats within 150 ms of a PPG detection;
ties choose the smallest absolute lag. This alignment is part of the benchmark
metric and is applied identically to every method.

## Representations and models

PPG conditioning and HCRD extraction are unchanged from the frozen PPGopt
experiment: zero-phase fourth-order 0.5--15 Hz Butterworth bandpass, robust
MAD scaling, 30 s HCRD windows with 2 s halos, eight hierarchy levels, complete
140-feature geometry bank, and the separately declared seven-feature local
morphology block. Area, signed area, quadratic energy, and triangle mass are
individual coordinates rather than replacements for the hierarchy.

Because neonate rates can exceed 200 bpm, event NMS is 200 ms for every MIMIC
record. No age/group label is supplied to the classifier.

The PPGopt-selected model families and capacities are transferred without a
new hyperparameter search:

1. geometry-only HGB: learning rate 0.1, 31 leaves, 200 iterations, L2=1;
2. hybrid HGB: learning rate 0.05, 15 leaves, 200 iterations, L2=1;
3. geometry logistic ablation: standardized features, C=0.1.

All use balanced class weights. Models are fit on 160 development records.
Validation selects only an event-probability threshold from
`{0.05, 0.10, ..., 0.95}` by median per-record F1, then micro-F1, precision,
and the higher threshold. The primary model is the best validation model, with
ties resolved in favor of geometry-only and then logistic regression.

## Baselines

- P0: `find_peaks` on the identical conditioned PPG. Development chooses a
  refractory-rate limit in `{180, 200, 240, 300}` bpm and prominence in
  `{0.1, 0.2, 0.35, 0.5, 0.75, 1.0}`.
- HeartPy 1.2.7 with 30--300 bpm bounds.
- Deterministic HCRD persistence P1 with minimum persistence in `{1,...,5}`.
- MSPTDfast v2 published benchmark: median record F1 0.968 on the same official
  Testing dataset. It is a contextual target until its MATLAB implementation is
  run under the identical locally generated ECG reference.

P0 is used only to estimate pulse-transit alignment when assigning development
candidate labels. Each aligned reference is assigned to at most one nearest
candidate within 150 ms.

## Metrics and success rules

The PPG-beats primary summary is median per-record F1 with interquartile range.
Also report micro-F1, precision, recall, adult/neonate strata, mean timing error,
false positives per minute, reference-detector agreement, and execution time.

A strong replication requires the locked HCRD primary model to:

1. exceed locally run P0 and HeartPy in both median record F1 and micro-F1;
2. exceed median F1 0.968, the published MSPTDfast-v2 result;
3. be no more than 0.01 F1 below the best local baseline in either adults or
   neonates;
4. retain a material geometry-only advantage over deterministic persistence;
5. obtain a positive 95% paired record-bootstrap interval against the strongest
   local baseline.

Failure is reported as a domain limitation or trade-off. No parameter change
after opening official Testing is confirmatory.

## Pre-outcome representation-control amendment M2

Before fitting any MIMIC PPG model, a mass-only diagnostic control was added to
answer whether polygon/triangle area and quadratic energy alone explain the
result. It uses the frozen geometry-HGB capacity (learning rate 0.1, 31 leaves,
200 iterations, L2=1) but receives only the per-level polygon area, signed
polygon area, quadratic energy, and triangle area coordinates together with
their cross-level decay, total, and maximizing-level summaries. It omits
support/persistence, component duration and boundaries, amplitude trajectories,
shape, offsets, and negative-neighbour context. Its threshold is selected on
validation by the same rule. This diagnostic is not eligible to become the
primary model. A full-geometry advantage over this control is evidence that the
decomposition hierarchy, rather than mass alone, is carrying useful signal.

## Pre-outcome reference-validity amendment M1

Consensus extraction on development data produced one record with no
high-quality reference beat. Before any PPG-detector score was calculated, the
minimum evaluable-record rule was fixed at three aligned consensus beats.
Records below this limit are excluded from both training and performance
summaries, with their count reported for every phase. No threshold based on PPG
detection performance or HCRD output is used for this exclusion.
