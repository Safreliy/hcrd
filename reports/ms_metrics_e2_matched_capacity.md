# E2 matched-capacity sensitivity analysis

HCRD-8+Q and the Gaussian-derivative control both contain 948 per-file and
2,847 aggregated variables. They use the same 75 raw waveform inputs, source
and target populations, standardized logistic learner, and no-target-refit
transfer protocol.

| Transfer | HCRD-8+Q AP | Gaussian control AP | HCRD minus control (95% CI) | Holm p |
|---|---:|---:|---:|---:|
| Falkor to MESOSCOPE | 0.8955 | **0.9063** | -0.0108 [-0.0594, 0.0405] | 0.7653 |
| MESOSCOPE to Falkor | **0.8990** | 0.7815 | +0.1175 [0.0624, 0.1654] | 0.00040 |

The result rules out a simple feature-count explanation for the
MESOSCOPE-to-Falkor gain, but it does not establish two-direction dominance of
HCRD over a rich conventional scale-space bank. Intervals are paired target-
feature bootstraps conditional on the fitted source models; they exclude
source-refit and compound/adduct/file-cluster uncertainty.

Reproduction:

```bash
python experiments/run_ms_metrics_e2_matched_capacity.py --bootstrap 10000
```

The protocol is in `docs/ms_metrics_e2_matched_capacity.md`; compact results
and predictions are in `results/ms_metrics_e2_matched_capacity/`.
