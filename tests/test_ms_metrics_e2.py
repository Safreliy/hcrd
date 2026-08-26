from __future__ import annotations

import numpy as np

from experiments.run_ms_metrics_e2 import _bootstrap_comparisons, global_window_qscore


def test_global_window_qscore_prefers_clean_unimodal_peak() -> None:
    x = np.linspace(0.0, 1.0, 51)
    clean = x ** 3 * (1.0 - x) ** 4
    rng = np.random.default_rng(4)
    noisy = rng.normal(size=x.size)
    clean_score = global_window_qscore(x, clean)
    noisy_score = global_window_qscore(x, noisy)
    assert np.all(np.isfinite(clean_score))
    assert clean_score[1] > noisy_score[1]


def test_global_window_qscore_rejects_short_trace() -> None:
    score = global_window_qscore(np.arange(4.0), np.arange(4.0))
    assert np.all(np.isnan(score))


def test_e2_bootstrap_reports_primary_domain_and_level_contrasts() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    base = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    scores = {
        "qscore": base,
        "domain_q": base,
        "hcrd_1_q": base,
        "hcrd_8_q": base,
        "hcrd_geometry_q": base,
        "area_only_q": base,
    }
    comparisons = _bootstrap_comparisons(labels, scores, replicates=50)
    assert "hcrd_8_q" in comparisons
    assert "hcrd_8_q_vs_domain_q" in comparisons
    assert "hcrd_8_q_vs_hcrd_1_q" in comparisons
