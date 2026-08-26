# Direct ITD comparison

This supplementary comparison was specified after the main synthetic
benchmark and is not part of the original frozen family. It uses 50 fresh
noiseless draws from the exact
uniform-grid chord-lobe class, the same latent-baseline MSE endpoint, centred
HCRD, and the final baseline returned by PySDKit 0.4.54 with `N_max=10`.

The run is reproduced with:

```bash
python experiments/run_external_comparison.py --trials 50 --noise 0 \
  --suite exact --output results/external_comparison_itd_c2
```

All configured external methods are evaluated on the same draws. Pairwise
uncertainty uses the runner's paired bootstrap and two-sided exact sign test;
Holm adjustment covers the seven HCRD comparisons in this run.
