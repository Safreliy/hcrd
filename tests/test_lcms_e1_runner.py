from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef

from experiments.run_lcms_eic_e1 import (
    _best_mcc_threshold,
    _weighted_ap,
    _weighted_ap_preparation,
)


def test_weighted_ap_matches_sklearn_with_ties() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)
    scores = np.asarray([0.9, 0.7, 0.7, 0.2, 0.2, 0.1])
    weights = np.asarray([1, 2, 3, 4, 1, 2], dtype=float)
    order, starts = _weighted_ap_preparation(scores)
    observed = _weighted_ap(labels, weights, order, starts)
    expected = average_precision_score(labels, scores, sample_weight=weights)
    assert np.isclose(observed, expected)


def test_best_mcc_threshold_is_exact() -> None:
    labels = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.uint8)
    scores = np.asarray([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    threshold = _best_mcc_threshold(labels, scores)
    candidates = np.unique(scores)
    expected = max(
        candidates,
        key=lambda value: (matthews_corrcoef(labels, scores >= value), value),
    )
    assert threshold == expected
