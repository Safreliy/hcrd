"""Executable witnesses for the transient-detection theorem boundary."""

from __future__ import annotations

import numpy as np

from hcrd import hcrd_area_anomaly_score
from hcrd.anomaly import _robust_positive_surprise


def test_sparse_row_fallback_is_discontinuous() -> None:
    at_boundary = np.zeros((1, 20), dtype=float)
    at_boundary[0, -1] = 1.0
    boundary_score = _robust_positive_surprise(at_boundary)[0, -1]

    epsilon = 1e-6
    nearby = at_boundary.copy()
    nearby[0, -2] = epsilon
    nearby_score = _robust_positive_surprise(nearby)[0, -1]

    assert np.isclose(boundary_score, 1.0)
    assert np.isclose(nearby_score, 10.0 / epsilon)


def test_row_normalisation_erases_cross_level_amplitude() -> None:
    row = np.asarray([0.0] * 18 + [0.25, 1.0])
    densities = np.vstack([row, 1e-12 * row])
    surprise = _robust_positive_surprise(densities)
    np.testing.assert_allclose(surprise[0], surprise[1], rtol=1e-12, atol=1e-12)


def test_pure_impulse_need_not_peak_at_impulse_sample() -> None:
    signal = np.asarray([0.0, 0.0, 1.0, 0.0])
    score = hcrd_area_anomaly_score(signal, max_levels=8, aggregation="max")
    assert score[1] > score[2]
    np.testing.assert_allclose(score[[1, 2]], [10.0 / 7.0, 1.25])
