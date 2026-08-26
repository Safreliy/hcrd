# Finite-sample chord-lobe recovery

## Selection obligation

Reconstruction conditional on a supplied knot set does not itself show that
noisy HCRD selects that set.  A generative selection result therefore assumes
alternating active curvature blocks, a sampled join, a two-segment minimum
lobe width, and an active-curvature margin relative to a simultaneous noise
threshold.  Exact knot recovery is then separated from latent-baseline error:
the retained knot ordinates still contain sample noise, which is controlled by
a second maximum-noise event.

## Approximate-join selection

Weakening exact joins to `|kappa| <= eta` does not necessarily make an observed
join inactive at the noise-only threshold `tau`.  Inactivity is nevertheless
unnecessary for the centred minimum-curvature rule. Under
`gamma > eta + 2 tau`, an observed join is strictly smaller than both adjacent
active curvatures. If its sign matches the left block, the walk selects it as
the smaller left side of the next transition; if its sign matches the right
block, it is the smaller right side of the preceding transition. If it is
inactive, the isolated-inactive rule selects it directly. Therefore every
threshold from `tau` through `eta + tau` is valid, including the original
noise-only implementation.

The algorithm does not need `eta`, but a data-derived recovery certificate
requires a separate simultaneous upper-confidence bound for it.

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

A4 approximate joins satisfy |kappa| <= eta
  + L1
  -> L7 observed joins have magnitude <= eta+tau

A5 gamma > eta + 2 tau
  + L1
  -> L8 active curvatures exceed eta+tau, retain their signs,
        and are strictly larger than observed joins

L7 + L8 + centred minimum-magnitude transition rule + A2 + A3 + L6
  -> approximate-join corollary with the same recovery and error bounds
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
between opposite active blocks. For joins bounded by `eta`, require
`gamma > eta + 2 tau`; both the noise-only threshold `tau` and the conservative
join-inactivating threshold `eta + tau` then recover the join. Otherwise replace
exact boundary recovery by a separately defined localization result.

Other necessary boundaries:

- If a lobe has fewer than two segments, the strict-coarsening clamp can alter
  the transition.
- If background curvature exceeds or cancels lobe curvature, sign blocks need
  not be visible.
- At `gamma=2 tau`, deterministic equality can leave an active curvature at
  the inactive threshold, so the theorem uses a strict inequality.
- Same-sample estimation of `sigma` is not covered by the fixed-threshold
  Gaussian probability statement.
