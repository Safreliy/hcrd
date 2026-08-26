# HCRD geometric mass and energy representation

Status: definitions and executable identities are implemented; predictive
value is an empirical hypothesis until the frozen external study is complete.

## 1. Finite geometric model: one signed structure

For a level input baseline `b_(l-1)` and the chord `b_l` on a retained-knot
interval `[a,b]`, let `d_l = b_(l-1) - b_l`. Define

```text
A_signed = integral d_l(x) dx
M_polygon = integral |d_l(x)| dx
E2 = integral d_l(x)^2 dx
T_triangle = duration * amplitude / 2.
```

All integrals are exact for the piecewise-linear sampled graph. On an exact
convex or concave run, `d_l` has one sign, hence
`M_polygon = |A_signed|`. This is the literal polygonal area between the graph
and its chord. `T_triangle` is exact only for a triangular residual and is an
inexpensive surrogate otherwise.

Parameter to vary: residual shape at fixed duration and amplitude.

Expected observation: `M_polygon / (duration * amplitude)` distinguishes a
sharp impact from a broad deformation even when their peaks match.

False intuition to avoid: the triangle area is not a universal approximation
bound. A piecewise-linear residual can be much sharper or flatter than a
triangle.

## 2. Combinatorial model: a measure on the hierarchy

Each structure is an atom indexed by `(level, left, right, sign)`, carrying
mass `M_polygon`. Per level we compute total positive/negative mass,
concentration, peak amplitude, duration, quadratic energy, and a weighted shape
factor. The fixed-length feature vector pads missing terminal levels with zero.

Parameter to vary: maximum hierarchy depth.

Expected observation: fine levels describe short impacts; coarse levels
describe broad deviations and background deformation. Which levels matter is
task dependent and must be evaluated by an ablation, not inferred from the
picture.

False intuition to avoid: HCRD levels are not mutually orthogonal frequency
bands. Quadratic energies do not obey a Parseval identity and must not be
summed as though they were wavelet energies.

## 3. Geometric time model: evolution of the measure

For repeated measurements `y_t`, the hierarchy produces a sequence of finite
measures `mu_t`. Candidate health indicators include total mass, mass by
level/sign, concentration, and their causal differences or rolling slopes.

Parameter to vary: history length used by the downstream predictor.

Expected observation for degrading bearings: impacts first increase fine-level
area/concentration; broader damage may later move mass toward coarser levels.
This is a falsifiable prediction, not yet a theorem.

False intuition to avoid: monotone physical degradation need not make every
HCRD quantity monotone. Load, speed, sensor mounting, noise, and competing
fault modes can change the hierarchy.

## Proof-backed transformation laws

For a vertical scaling `y -> c y` plus any affine trend:

- knot sets are unchanged at zero tolerance for `c != 0`;
- polygon and triangle mass scale by `|c|`;
- quadratic energy scales by `c^2`;
- affine trends contribute zero detail;
- under `x -> alpha x + beta`, both area quantities scale by `|alpha|`.

These identities are covered by executable tests in `tests/test_energy.py`.
The implementation integrates the sparse hierarchy on previous-level knots,
without constructing every dense length-`n` detail.

## Open theory

- Stability bounds for polygon mass when the hard knot set changes.
- Approximation bounds between triangle and polygon mass under explicit shape
  assumptions.
- Whether a normalized mass distribution defines a useful hierarchy metric.
- Statistical behavior under stationary noise and impulsive alternatives.
- Conditions under which mass migration across levels detects a change point.
