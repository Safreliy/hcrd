# E4 protocol: multiscale lobe recovery on independent real LC--MS backgrounds

Status: **frozen before running HCRD or any comparator on an injected case**.  
Protocol: `hcrd-e4-neatms-real-background-v1`  
Freeze date: 2026-08-25

## Research question and scope

Can the core multilevel HCRD decomposition recover a slowly varying baseline
after one or several nested chord lobes are added to real LC--MS acquisition
backgrounds, more accurately than strong classical baseline estimators that
are given an oracle choice of tuning parameter?

This is a controlled semi-synthetic recovery experiment with exact injected
ground truth. It is not manual peak-quality classification and cannot be
reported as performance on natural unknown peaks. Its purpose is to test the
specific lobe class motivated by the theorem while retaining real
heteroskedastic, correlated instrument backgrounds.

## Frozen source and background population

Source: NeatMS LCMS data, Dataset 1, Zenodo record
<https://doi.org/10.5281/zenodo.3973172>, CC BY 4.0.

- archive: `dataset1.zip`, 159,079,084 bytes;
- MD5: `47a63b1bcba15d9b5ce6c6e4b6d5537e`;
- twenty readable `sample1.mzML`--`sample20.mzML` files are used;
- the 6,148-byte `sample21.mzML` placeholder is excluded by file validity and
  size before any injected signal or method score exists.

For every run, retain MS1 scan time and the recorded total-ion current (TIC).
Take the central 1,032 scans and split them into eight consecutive blocks of
129 samples. Within a block, normalize TIC by subtracting its median and
dividing by `q95-q05` (a zero scale makes the block ineligible). Compute the
median absolute deviation from the straight chord joining the two endpoints.
Retain the three lowest-deviation blocks per run, breaking ties by block index.
This score-independent rule gives exactly 60 backgrounds and explicitly
selects the approximately affine-background class.

## Fixed injected structures

Sample locations are `x=0,...,128`. A triangular lobe is zero at its two
declared support endpoints and has unit height at its declared apex.
Asymmetry is assigned by SHA-256 of `(protocol, file, block, component)` into
the fixed set `{0.35, 0.50, 0.65}`; it is independent of every method.

Two cases are generated on each frozen background:

1. `single`: support `[32,96]`, amplitude `1.0`;
2. `nested`: the sum of a broad `[16,112]` lobe of amplitude `1.0`, a
   `[40,88]` lobe of amplitude `0.6`, and a `[56,72]` lobe of amplitude `0.3`.

The observed signal is `background + injected_detail`. No random measurement
noise is added: the real TIC block is the complete nuisance process. The exact
background and injected component remain available only for evaluation and
oracle tuning of the comparison methods.

## Frozen methods

- `HCRD-L1`: the first minimum-curvature HCRD baseline;
- `HCRD-L8`: the final trend after at most eight HCRD levels; this is the
  primary method and has no fitted parameter;
- Gaussian smoothing, oracle `sigma` in `{1,2,4,8,16,32}`;
- Savitzky--Golay, oracle `(window, polynomial)` over windows
  `{9,17,33,65,97}` and polynomials `{2,3}` where valid;
- grey-scale morphological opening, oracle size in `{9,17,33,65,97}`;
- asymmetric least squares (AsLS), oracle `lambda` in
  `{1e2,1e3,1e4,1e5,1e6}` and `p` in `{0.001,0.01,0.05}`, 20 iterations;
- orthogonal wavelet low-pass reconstruction with `sym4`, oracle retained
  approximation level in every feasible level from 1 through 5.

“Oracle” means the parameter minimizing baseline MSE separately for every
case using the known injected ground truth. This intentionally advantages the
classical comparator and must not be described as a deployable selection rule.

## Endpoints, inference, and pre-specified success

The primary population is the 60 `nested` cases. The primary loss is mean
squared error between estimated and true normalized background. Detail MSE is
secondary. The 60 `single` cases are a secondary theorem-aligned analysis.

For `HCRD-L8` versus `HCRD-L1` and each of the five oracle comparator families,
report paired mean MSE difference, win/tie/loss counts, exact sign test, and a
50,000-replicate paired background bootstrap. Familywise inference uses
Bonferroni two-sided intervals/tests over the six primary comparisons
(`alpha=0.05/6`).

Pre-specified E4 success requires all of:

1. HCRD-L8 has the lowest mean nested-case baseline MSE of every fixed method;
2. every simultaneous upper confidence bound for
   `MSE(HCRD-L8)-MSE(comparator)` is below zero;
3. HCRD-L8 beats HCRD-L1 under the same simultaneous rule, demonstrating a
   genuine benefit from multiple levels;
4. HCRD-L8 wins on at least 45 of 60 nested backgrounds against the strongest
   classical oracle by mean MSE.

No secondary single-lobe result, detail metric, runtime result, alternative
background selection, or post-hoc method may rescue a failed primary endpoint.

## Planned independent replication

If E4 succeeds, the unchanged extraction, injections, methods, and endpoint
will be frozen on NeatMS Dataset 2 before its archive is downloaded or scored.
That replication is required before a multi-source real-background superiority
claim. If E4 fails, Dataset 2 is not used to search tuning parameters; the
failure instead refines the boundary of the chord-background class.
