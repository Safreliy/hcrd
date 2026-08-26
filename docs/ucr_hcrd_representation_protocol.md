# UCR morphology representation benchmark (U1)

Status: frozen before any U1 classifier result was computed.

## Question

Does the complete HCRD hierarchy provide useful supervised information that is
lost by a raw signal or by a conventional wavelet hierarchy?  This protocol is
deliberately about the decomposition and its collection of signed structures.
Polygon mass and quadratic energy are auxiliary ablations, not the proposed
method by themselves.

## Benchmark population

The population is the 112 equal-length, no-missing-value univariate problems
from the 2018 UCR Time Series Classification Archive, as listed by aeon 1.5.0.
The official default train/test split is retained.  A problem is eligible when
all of the following label-independent resource rules hold:

- time-series length is between 16 and 2048 samples;
- train plus test contains at most 10,000 cases;
- train plus test contains at most 10,000,000 scalar observations;
- both splits contain at least two classes and the training split has at least
  20 cases.

Every eligible problem is assigned without looking at its values or labels:

```text
discovery     if first_byte(SHA256(UTF8(dataset_name))) < 128
confirmation  otherwise
```

The preparation script writes the exact population, exclusions, shapes and
assignment to `data/manifests/ucr_u1_manifest.json`.  The confirmation outcomes
must not be computed until the discovery analysis has written and hashed a
single geometric subgroup rule.

## Three executable signal models

### M0: raw morphology

Definition: median-centre and RMS-scale each series, then expose it as one
channel to MiniRocketMultivariate.

Demonstration: this is the strong representation-free control.  The tunable
parameter is the fixed MiniRocket kernel count, 9,996 effective features from a
request of 10,000 kernels.  Expected observation: high accuracy on many UCR
problems, so HCRD must add information rather than beat a weak classifier.
False intuition: a weak linear classifier on ten hand summaries is not a fair
raw-signal baseline.  Unproved claim: none; this is an empirical control.

### M1: multichannel decomposition

Definition: expose five fine-to-coarse HCRD detail channels followed by the
final trend.  The trend always remains the sixth channel.  If a finite
hierarchy ends early, aeon's per-channel variance guard cannot accept the
resulting exactly-zero detail.  Such a channel alone receives a fixed,
label-independent `1e-4` linear carrier that is subtracted from another
channel, so exact reconstruction is preserved.  The db4 wavelet control
uses five reconstructed detail channels plus its approximation, so channel
count and downstream learner match HCRD.

Demonstration: the channels reconstruct the normalized input exactly.  The
tunable parameter is the fixed depth, five.  Expected observation: tasks whose
class information is organized by signed chord structures should become more
linearly accessible after the same random convolutional map.  False intuition:
more channels alone prove a better representation; the equal-channel wavelet
control is required.  Unproved claim: HCRD is superior on any UCR subgroup.

### M2: structure collection and hybrid

Definition: retain a fixed-size description of the *set* of structures at each
level: signed counts, distribution quantiles, spatial-bin occupancy and signed
mass, and the largest structures with position, support, sign, amplitude,
polygon mass, quadratic energy and shape factor.  A RidgeClassifierCV is fitted
to this bank alone and to its concatenation with M1 MiniRocket features.

Demonstration: unlike a single total area, two signals with equal total mass but
different numbers, positions, signs or support lengths map to different
vectors.  Tunable parameters are fixed at eight spatial bins and eight ranked
structures per level.  Expected observation: the bank may help when location
and organization of convex/concave events determine the class.  False
intuition: polygon area is the decomposition; it is only one coordinate of the
bank.  Unproved claim: the fixed vector is a sufficient statistic for HCRD.

## Fixed representations and learner

The primary comparison is:

1. `raw_minirocket` (M0);
2. `wavelet_minirocket` (five db4 details plus approximation);
3. `hcrd_minirocket` (five HCRD details plus trend);
4. `hcrd_structure` (the full structure bank);
5. `hcrd_hybrid` (HCRD MiniRocket features plus the structure bank).

`hcrd_energy` is a diagnostic ablation containing only the level-wise aggregate
block.  It cannot support a claim about the full method.

All MiniRocket transforms request 10,000 kernels, use random state 0 and see
only the official training split during fitting.  Every head is
`StandardScaler(with_mean=False)` followed by `RidgeClassifierCV` with alphas
`10**linspace(-3, 3, 10)`.  No test labels are used for tuning.

## Outcomes and multiplicity

The primary per-problem outcome is default-split test accuracy, matching UCR
practice.  Balanced accuracy and macro-F1 are secondary.  At the collection
level report mean rank, mean accuracy, wins/ties/losses, paired dataset-level
bootstrap intervals and a two-sided Wilcoxon signed-rank test.  Discovery
inference is explicitly exploratory.

The discovery stage may search only one-dimensional thresholds of training-only
HCRD geometry descriptors.  The selected rule must include at least ten
discovery problems, maximize mean `hcrd_hybrid - max(raw, wavelet)` accuracy,
and be written verbatim with its SHA-256 digest before confirmation begins.
Only its performance on the untouched confirmation problems is confirmatory.

## Runtime and parallelism

Independent series are decomposed by a process pool.  A hierarchy within one
series remains sequential.  Dataset-level work is serial to avoid nested CPU
oversubscription; MiniRocket uses the same worker count after HCRD extraction.
Wall time, extraction time and model time are recorded separately.

## Discovery amendment U1-D2 (written after U1-D1, before confirmation)

U1-D1 showed that passing all channels to one multivariate MiniRocket is not a
generally competitive HCRD realization.  A mechanistic concern is that random
channel combinations erase level identity and that coarse and fine detail
channels have very different scales.  U1-D2 therefore tests a candidate that
was not part of the confirmatory specification:

- median-centre each nonempty component and RMS-scale it when its RMS exceeds
  `1e-3`;
- fit one univariate MiniRocket per component;
- split the 10,000-kernel budget equally across the six components (1,680
  requested kernels per component, the nearest multiple of 84);
- concatenate the six feature blocks and use the same scaled RidgeClassifierCV;
- apply the identical construction to the six db4 wavelet channels;
- separately fit a fixed ExtraTrees model and a fixed LightGBM model to the
  complete structure bank.

Raw-plus-HCRD concatenation is recorded as a higher-cost hybrid and may not be
compared as an equal-budget representation.  All U1-D2 model choices are
exploratory.  One candidate and one geometric subgroup rule, if any, must be
frozen before confirmation.

## Discovery amendment U1-D3 (written after the four-problem D2 diagnostic)

The componentwise convolution candidate did not resolve the main deficit on
the four diagnostic problems.  In contrast, nonlinear tree learners on the
structure bank improved two of the four without access to the raw series.
U1-D3 therefore evaluates the structure bank with ExtraTrees.  Its
`min_samples_leaf` in `{1, 2, 4}` and `max_features` in `{sqrt, 0.5}` are chosen
by stratified cross-validation on the official training split only (up to five
folds, limited by the smallest class).  A fixed LightGBM model is retained as a
control.  This amendment remains discovery-only.

## Discovery amendment U1-D4 (training-only gate)

Because no marginal geometry threshold identifies all useful tasks, U1-D4
uses a practical representation gate.  On each official training split, raw
MiniRocket and wavelet MiniRocket heads are scored by stratified CV after an
unsupervised full-training transform.  The ExtraTrees score is its training
GridSearchCV score from U1-D3.  A single margin for selecting HCRD over the
better conventional representation is chosen on discovery outcomes, hashed,
and then applied unchanged to confirmation.  Datasets whose smallest training
class has one case default to raw because internal stratified CV is undefined.
