# E35 protocol: HCT with unknown bounded heteroskedasticity

## Scientific question

Can HCT turn a state-of-the-art S-shaped point estimate into an honest
inflection-location confidence set when independent Gaussian errors have
unknown, unequal variances?

The practically meaningful target is the entire set of locations compatible
with a convex-to-concave mean shape, including nonsmooth and non-identifiable
signals.  The variance class is indexed by the declared sensitivity parameter
\(\kappa=\max_i\sigma_i^2/\bar\sigma^2\).

## Method fixed before simulation

- contrast family: dyadic block sizes and separation multipliers `(1, 2, 4)`;
- total error probability `alpha = 0.05`;
- variance-envelope allocation `eta = 0.01`;
- contrast allocation `0.04`;
- fixed consecutive residual blocks of size two or three;
- variance envelope from Theorem E35.1;
- candidate point estimator: `Sshaped::sshapedreg` where external comparison
  is used; projection into the HCT set is post-processing only.

## Simulation confirmation

- fresh seed `20262011`;
- sample sizes `n in {500, 1000}`;
- uniform and `Beta(4,8)` quantile designs;
- the four published Feng et al. signals, including the cusp, onset, jump and
  weak logistic cases;
- `200` repetitions per cell;
- Gaussian noise with average variance `0.1^2`;
- variance profiles:
  - `constant`: unit relative variance;
  - `linear`: relative variance proportional to `0.35 + 1.30*x`;
  - `plume_peak`: relative variance proportional to
    `0.25 + 2.75*exp(-((x-0.58)/0.12)^2)`;
- each profile is normalized to mean one and its exact finite-design
  max-to-mean ratio is supplied as \(\kappa\).

The method is compared with an oracle HCT band that knows every
\(\sigma_i\).  The oracle is a width reference, not an implementable
competitor.

### Frozen gates

- HCT target coverage at least `0.93` in every cell;
- variance-envelope coverage at least `0.93` in every cell;
- zero empty confidence sets on the simultaneous oracle-band event;
- all `48` method/profile/design/sample-size/signal cells retained;
- all weak-signal cells retained rather than removed as failures.

## LIDAR illustration

The `SemiPar::lidar` data are used because Feng et al. used them to estimate
the centre of an atmospheric mercury plume.  We fit `Sshaped::sshapedreg` to
`-logratio`, turning the physically expected decreasing inverse-S curve into
an increasing convex-to-concave curve.  We report:

- the S-shaped LSE point estimate;
- the nominal residual-bootstrap interval from `ShapeChange` as a descriptive
  comparator, explicitly not as a heteroskedastic guarantee;
- HCT intervals in metres for predeclared `kappa in {1, 1.5, 2, 3, 4}`.

This real-data analysis is a model-sensitivity illustration, not a coverage
experiment.  HCT's finite-sample interpretation additionally requires
independent Gaussian errors and a correct upper bound on \(\kappa\).

