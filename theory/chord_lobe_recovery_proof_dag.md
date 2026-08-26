# Finite-sample chord-lobe recovery

## First-failure review

First failure: the earlier recovery corollary assumed that the desired points
were already the first HCRD knot set.

Why it matters: this proves reconstruction conditional on selection, but does
not give a generative condition under which noisy HCRD selects the points.

Precise question: which observable curvature pattern makes the centred knot
walk recover every chord boundary with finite-sample probability at least
`1-delta`?

Repair: require alternating active curvature blocks, a single sampled zero at
each boundary, a two-segment minimum lobe width, and an active-curvature margin
strictly greater than twice a simultaneous noise threshold.

Next likely risk: confusing exact knot recovery with exact latent-baseline
recovery when the retained knot ordinates themselves contain noise.

## Claim signature

On a uniform grid, let `f=b+q`, where `b` is affine between declared knots,
`q` vanishes at those knots, the population curvature is zero exactly at every
interior knot, and the remaining curvatures form alternating one-sign blocks
with magnitude at least `gamma`. Under iid Gaussian sample noise, HCRD with
threshold `tau` recovers all first-level knots when `gamma>2 tau`, on the
simultaneous curvature event. A second maximum-noise event bounds the chord
baseline error by `epsilon` and the detail error by `2 epsilon`.

## Proof DAG

```text
A1 uniform grid + iid Gaussian noise
  -> L1 simultaneous curvature error <= tau (probability 1-delta/2)
  -> L2 population zeros remain inactive
  -> L3 active curvature signs and status are preserved when gamma>2 tau

A2 alternating blocks + one sampled zero + gap >= 2
  + L2 + L3
  -> L4 centred walk selects exactly each declared boundary

A3 q=0 at knots + b affine on every knot interval
  + L4
  -> L5 estimated baseline minus b is the chord interpolant of knot noise

A1
  -> L6 maximum sample noise <= epsilon (probability 1-delta/2)

L5 + L6
  -> C1 baseline sup error <= epsilon and MSE <= epsilon^2
  -> C2 detail sup error <= 2 epsilon and MSE <= 4 epsilon^2

L1 + L6 + union bound
  -> theorem probability >= 1-delta
```

## Counterexample ledger

Claim: equal alternating parabolic lobes have exact sampled boundary recovery.

Removed hypothesis: equal transition amplitude, hence the isolated zero
curvature at the join.

Candidate: two four-interval lobes with amplitudes `1` and `1/4`.

Verification: the noiseless divided curvatures are
`(-.5,-.5,-.5,.5625,.125,.125,.125)` and the centred rule returns knots
`(0,3,8)` instead of `(0,4,8)`.

Conclusion failure: the intended join is displaced by one sample.

Status: CTR.

Minimal repaired statement: require a single inactive sampled curvature
between opposite active blocks, or replace exact boundary recovery by a
margin-dependent localization result for unsampled crossings.

Other necessary boundaries:

- If a lobe has fewer than two segments, the strict-coarsening clamp can alter
  the transition.
- If background curvature exceeds or cancels lobe curvature, sign blocks need
  not be visible.
- At `gamma=2 tau`, deterministic equality can leave an active curvature at
  the inactive threshold, so the theorem uses a strict inequality.
- Same-sample estimation of `sigma` is not covered by the fixed-threshold
  Gaussian probability statement.
