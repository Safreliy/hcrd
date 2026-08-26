# HCRD polygon mass as a temporal object

This note separates exact consequences of the decomposition from working
hypotheses that require experiments. It accompanies protocol A1; it does not
use A1 evaluation labels.

## Exact starting point

Write the finite HCRD reconstruction as

`y(t) = r_L(t) + sum_l d_l(t)`.

For level `l`, define the nonnegative polygon-mass density

`a_l(t) = |d_l(t)|`.

Because each detail structure is the piecewise-linear gap between an input
baseline and its retained chord,

`integral a_l(t) dt = sum_s polygon_area(l, s)`.

This is an exact identity, checked independently in the test suite on irregular
grids. It is an L1 geometric mass identity, not a Parseval energy identity:
HCRD details are not asserted to be orthogonal.

## Model 1: finite multiscale time series

**Object.** For sampled times `t_1, ..., t_n`, collect the rows into the
nonnegative matrix `A in R_+^(L x n)`, `A[l, i] = a_l(t_i)`. A scalar area
series is a fixed projection such as `M_i = sum_l A[l, i]`; retaining `A`
preserves the hierarchy.

**Small demonstration.** Decompose a periodic carrier, then add one triangular
pulse. The carrier should form a repeated pattern in the columns of `A`; the
pulse should create a localized excess at the levels whose structure durations
match the pulse width.

```python
A = multiscale_area_density(signal, max_levels=8)
M = A.sum(axis=0)
```

**Parameter to vary.** Number of retained levels `L` (4, 8, complete).

**Expected observation.** Fine levels respond to sharp local curvature;
coarser levels retain longer shape departures. Eight fixed levels should often
provide a useful linear-time practical compromise.

**False intuition exposed.** A large total area is not automatically an
anomaly: a normal high-amplitude oscillation can have more polygon mass than an
abnormal low-amplitude shape change.

**Still unproved.** No distribution-free theorem says anomalous samples have
larger `M_i`, or identifies one universally optimal level depth.

## Model 2: mass flow on the hierarchy-time graph

**Object.** Use vertices `(l, i)` on a rectangular level-time graph. Horizontal
edges connect adjacent times and vertical edges connect adjacent hierarchy
levels. When `M_i > 0`, define a scale distribution

`p_l(i) = A[l, i] / M_i`.

One inexpensive temporal characteristic is the total-variation flux

`F_i = 0.5 * ||p(i) - p(i-1)||_1`,

which is always in `[0, 1]`. It reports redistribution of geometric mass across
scales rather than merely a change of amplitude.

**Small demonstration.** Compare an amplitude-only rescaling of a fixed pulse
with a pulse whose width suddenly contracts. After normalisation, the first
should leave `p` approximately unchanged, whereas the second should move mass
toward finer levels and increase `F`.

```python
p = A / np.maximum(A.sum(axis=0, keepdims=True), eps)
flux = 0.5 * np.abs(np.diff(p, axis=1)).sum(axis=0)
```

**Parameter to vary.** Total variation versus a distance that respects level
separation, such as one-dimensional Wasserstein distance on the level path.

**Expected observation.** Scale flux should complement total mass for abrupt
morphological transitions and degradation onsets.

**False intuition exposed.** Large flux need not mean failure: chirps, regime
changes, or legitimate nonstationarity also move mass between scales.

**Still unproved.** Stability of `F` depends on stability of both knot selection
and the normalisation near `M_i = 0`; a robust or persistence-thresholded HCRD
may be required for noisy signals.

## Model 3: a low-dimensional geometric trajectory

**Object.** Project each column of `A` to

- total mass `M_i`;
- scale barycentre `B_i = sum_l l p_l(i)`;
- scale entropy `H_i = -sum_l p_l(i) log p_l(i)`.

The signal becomes a curve `z_i = (log(1 + M_i), B_i, H_i)` in three dimensions.
A downstream change-point detector, one-class learner, or sequence model can
analyse this curve while the coordinates remain interpretable.

**Small demonstration.** A stationary periodic signal should trace a repeated
loop or compact cloud. A new transient morphology should make an excursion in
mass, typical scale, entropy, or several coordinates together.

```python
z = np.column_stack([np.log1p(M), barycentre, entropy])
score = downstream_detector.fit_predict(z)
```

**Parameter to vary.** Temporal smoothing or window length used by the
downstream detector.

**Expected observation.** This trajectory may be better for change-point and
health-state tasks than direct RUL regression, because it represents the onset
and redistribution of structure rather than forcing a linear life coordinate.

**False intuition exposed.** Three moments are not a lossless summary: distinct
scale distributions can have equal mass, barycentre, and entropy. The full
matrix `A` remains the primary representation.

**Still unproved.** Useful separation, monotonicity, and early-warning lead time
are empirical properties of a task and cannot be inferred from the geometric
identity alone.

## Evidence boundary

- **Theorem/identity:** row integrals equal exact HCRD polygon masses; affine
  additions vanish from details; vertical rescaling multiplies all densities
  by the absolute scale factor.
- **Executable observation:** the current tests verify conservation and affine
  behaviour numerically on finite signals.
- **Hypothesis:** anomalies or degradation produce separable excursions in
  mass, scale distribution, flux, or the low-dimensional trajectory. TSB-AD A1
  is the first broad confirmation attempt for this hypothesis.

