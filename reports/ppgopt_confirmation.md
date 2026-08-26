# Locked PPGopt confirmation result

Protocol SHA-256: `1e33526deb7e3f24f426c8a2d92506c203e17bcbdee56200b2bde11388cbc782`  
Frozen rule SHA-256: `3c71be6424fddc171d8f75d55359572c25ee6e9cd6d01b62a59a2b3e58a3599c`  
Subjects: S6--S7; 30 recordings; 5,734 valid expert events.

| Method | F1 | Precision | Recall | Mean error, ms |
|---|---:|---:|---:|---:|
| P0 find_peaks | 0.967011 | 0.997774 | 0.938089 | 9.639 |
| HeartPy | 0.972765 | 0.996706 | 0.949948 | 9.354 |
| P1 deterministic HCRD | 0.797963 | 0.868940 | 0.737705 | 14.904 |
| P2 HCRD geometry | 0.983572 | 0.996421 | 0.971050 | 9.336 |
| P2 HCRD hybrid | 0.987237 | 0.996623 | 0.978026 | 9.523 |

The frozen HCRD hybrid reaches **F1 = 0.987237**, improving over
the tuned local-maximum baseline by **0.020226**
and over HeartPy by **0.014472**.
Geometry alone reaches **0.983572**; the seven-feature morphology
block adds 0.003665. In contrast, thresholding
multilevel persistence without learning reaches only 0.797963.

The largest practical gain is under step motion: hybrid HCRD improves over P0
by 0.106713 F1. At rest it is
-0.001220 relative to P0, within the frozen 0.01
non-inferiority margin.

The paired subject-cluster bootstrap interval for hybrid minus P0 is
[0.001159,
0.037522]. This meets the frozen
rule, but there are only two confirmation subjects; replication on an
independent PPG data are still required to establish population-level transfer.

Published all-data optima from Wolling et al. (Karlen 0.958, van Gent 0.970)
are contextual rather than held-out comparisons. The primary direct
comparators above were executed by this repository with the same locked test
subjects and scoring code.
