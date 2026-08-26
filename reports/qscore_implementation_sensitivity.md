# qscore implementation sensitivity

Every row compares HCRD-8 plus a qscore variant with that same qscore variant alone under the unchanged two-direction transfer learner.

| Variant | Width | Falkor to MESOSCOPE | MESOSCOPE to Falkor |
|---|---:|---:|---:|
| Current median, m\geq8 | 2 | 0.1181 [0.0628, 0.1769] | 0.1005 [0.0554, 0.1457] |
| Median, m\geq5 | 2 | 0.1244 [0.0688, 0.1832] | 0.0987 [0.0538, 0.1432] |
| Author-like five-summary, m\geq5 | 5 | 0.1517 [0.0813, 0.2184] | 0.0765 [0.0324, 0.1211] |
| Multisummary seven-variable, m\geq8 | 7 | 0.0847 [0.0320, 0.1407] | 0.0798 [0.0347, 0.1262] |

All eight intervals exclude zero; the largest Holm-adjusted bootstrap value is 0.002000.

## Falkor author-output fidelity

```json
{
  "q_current_med_snr": {
    "author_column": "med_SNR",
    "n": 1277,
    "pearson": 0.9336135896537437,
    "spearman": 0.8945427823719853
  },
  "q_current_med_cor": {
    "author_column": "med_cor",
    "n": 1277,
    "pearson": 0.86688847842734,
    "spearman": 0.8585633772621548
  },
  "q_min5_med_snr": {
    "author_column": "med_SNR",
    "n": 1277,
    "pearson": 0.9328405782361021,
    "spearman": 0.8935898654438609
  },
  "q_min5_med_cor": {
    "author_column": "med_cor",
    "n": 1277,
    "pearson": 0.8644024006496251,
    "spearman": 0.8573137506559276
  },
  "q_author5_max_snr": {
    "author_column": "max_SNR",
    "n": 1277,
    "pearson": 0.8856557824425798,
    "spearman": 0.887283070448789
  },
  "q_author5_max_cor": {
    "author_column": "max_cor",
    "n": 1277,
    "pearson": 0.7084967222638092,
    "spearman": 0.7493460602826706
  },
  "q_author5_second_cor": {
    "author_column": "medtop3_cor",
    "n": 1277,
    "pearson": 0.7890284016589688,
    "spearman": 0.8178948701311131
  }
}
```

Author per-detected-peak qscore outputs are unavailable for MESOSCOPE.
