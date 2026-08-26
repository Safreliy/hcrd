from __future__ import annotations

import numpy as np
import pytest

from hcrd import buffered_crossfit_folds, gaussian_contrast_pivot


def test_buffered_folds_score_every_index_and_separate_guides() -> None:
    folds = buffered_crossfit_folds(23, block_size=5, dependence_lag=2)
    scored = np.concatenate([fold.scoring_indices for fold in folds])
    np.testing.assert_array_equal(scored, np.arange(23))
    for fold in folds:
        if fold.guide_indices.size:
            distances = np.abs(
                fold.guide_indices[:, None] - fold.scoring_indices[None, :]
            )
            assert np.min(distances) > 2
        assert not np.intersect1d(fold.scoring_indices, fold.buffer_indices).size
        assert not np.intersect1d(fold.scoring_indices, fold.guide_indices).size


def test_gaussian_contrast_pivot_uses_full_covariance() -> None:
    values = np.asarray([1.0, 2.0, 4.0])
    contrast = np.asarray([-0.5, -0.5, 1.0])
    covariance = np.asarray(
        [[1.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 1.0]]
    )
    result = gaussian_contrast_pivot(values, contrast, covariance)
    assert result.estimate == pytest.approx(2.5)
    assert result.standard_error == pytest.approx(
        np.sqrt(contrast @ covariance @ contrast)
    )
    assert 0.0 < result.two_sided_p_value < 1.0


def test_known_covariance_pivot_is_null_uniform_in_monte_carlo() -> None:
    rng = np.random.default_rng(20260826)
    covariance = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    contrast = np.asarray([1.0, -1.0])
    draws = rng.multivariate_normal(np.zeros(2), covariance, size=10000)
    p_values = np.asarray(
        [
            gaussian_contrast_pivot(row, contrast, covariance).two_sided_p_value
            for row in draws
        ]
    )
    assert abs(np.mean(p_values <= 0.05) - 0.05) < 0.01


def test_dependent_inference_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        buffered_crossfit_folds(10, block_size=0, dependence_lag=1)
    with pytest.raises(ValueError, match="symmetric"):
        gaussian_contrast_pivot([1.0, 2.0], [1.0, -1.0], [[1.0, 1.0], [0.0, 1.0]])
