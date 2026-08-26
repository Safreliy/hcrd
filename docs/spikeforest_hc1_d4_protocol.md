# D4 protocol: real paired neural-spike detection

Status on 2026-08-25: population and session split fixed before waveform or
detector-score inspection; detector development is allowed only on the
development sessions.  Confirmation remains locked until a second freeze.

## Why this task

The WSD C2 structural audit did not validate sparse anomaly duration as a
sufficient class rule.  Its only clear association was the predeclared
exploratory shape-concentration descriptor.  Extracellular action potentials
are real, compact, high-concentration transient waveforms.  In paired HC-1
recordings, intracellular spikes provide physical event times independent of
the extracellular detector being evaluated.

## Fixed source and population

- Source: official CRCNS HC-1 public S3 mirror.
- Recording index and ground-truth preparation: official SpikeForest
  `PAIRED_CRCNS_HC1` index.
- Population: the 15 smallest HC-1 session archives in the 2026-08-25 public
  S3 listing that contain at least one indexed paired SpikeForest recording.
  Archive size is an access constraint and was chosen without waveforms,
  labels, or method scores.
- Fixed roots: `d13521`, `d5331`, `d13921`, `d14531`, `d15121`, `d7211`,
  `d7111`, `d5611`, `d13711`, `d18811`, `d18712`, `d7212`, `d6111`,
  `d14921`, `d12821`.

## Locked split

For a session root, compute SHA-256 of its UTF-8 name.  An even first byte is
development; an odd first byte is confirmation.  This yields:

- development: `d13921`, `d14531`, `d15121`, `d18811`, `d5331`, `d6111`,
  `d7111`, `d7212` (20 indexed recordings);
- confirmation: `d12821`, `d13521`, `d13711`, `d14921`, `d18712`, `d5611`,
  `d7211` (14 indexed recordings).

No confirmation waveform, ground-truth time, detector score, or result may be
read before freezing preprocessing, channel policy, tolerances, HCRD score,
comparator family, primary metric, and aggregation level.  Development may be
used to choose them.  Inference must aggregate at independent session-root
level, not pretend windows or spikes are independent replicates.

## Candidate comparators to freeze after development

At minimum: robust amplitude threshold, nonlinear energy operator, continuous
wavelet detector, and a training-half matched-template detector.  All methods
must receive identical bandpass input and refractory/event matching.  The
primary endpoint should be event-level average precision or the area under a
precision--recall curve over thresholds, with a fixed temporal tolerance.

The scientific claim, if confirmed, will be limited to training-free or
classical spike detection.  Spike sorting, unit identity, and superiority over
modern multichannel sorters are outside scope.
