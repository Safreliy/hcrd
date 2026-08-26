# Direct HCRD--ITD comparison

On 50 fixed noiseless draws from the exact chord-lobe class, centred HCRD
recovered the latent baseline to numerical zero. The final baseline returned
by PySDKit 0.4.54 ITD (`N_max=10`) had mean MSE 0.1473 and median MSE 0.1085.

The paired HCRD-minus-ITD mean difference was -0.1473 (95% bootstrap interval
[-0.1871, -0.1085]); HCRD won all 50 trials and the Holm-adjusted two-sided
exact sign-test value was 1.24e-14. This is a class-specific baseline-recovery
comparison and does not address ITD's native time--frequency objectives.

Reproduction:

```bash
python experiments/run_external_comparison.py --trials 50 --noise 0 \
  --suite exact --output results/external_comparison_itd_c2
```

See `docs/itd_direct_comparison_protocol.md` and
`results/external_comparison_itd_c2/`.
