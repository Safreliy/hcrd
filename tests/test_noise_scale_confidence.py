import numpy as np
import pytest
from scipy.stats import chi2

from hcrd.noise_scale_confidence import (
    consecutive_block_design,
    gaussian_block_upper_scale,
    gaussian_projection_upper_scale,
)


def test_consecutive_block_design_is_balanced_and_full_rank() -> None:
    design = consecutive_block_design(11, 3)
    np.testing.assert_allclose(design.sum(axis=1), 1.0)
    assert np.linalg.matrix_rank(design) == 3
    assert sorted(design.sum(axis=0).astype(int)) == [3, 4, 4]


def test_projection_scale_bound_matches_residual_formula() -> None:
    y = np.array([0.0, 2.0, 1.0, 5.0, 4.0, 6.0])
    design = consecutive_block_design(y.size, 2)
    result = gaussian_projection_upper_scale(
        y, design, failure_probability=0.05
    )
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    rss = float(np.sum((y - fitted) ** 2))
    assert result.nuisance_rank == 2
    assert result.residual_degrees_of_freedom == 4
    assert result.residual_sum_squares == pytest.approx(rss)
    assert result.upper_scale == pytest.approx(np.sqrt(rss / chi2.ppf(0.05, 4)))


def test_scale_bound_is_invariant_to_nuisance_mean() -> None:
    design = consecutive_block_design(12, 3)
    y = np.linspace(-1.0, 1.0, 12)
    shift = design @ np.array([10.0, -4.0, 7.0])
    first = gaussian_projection_upper_scale(y, design, failure_probability=0.1)
    second = gaussian_projection_upper_scale(
        y + shift, design, failure_probability=0.1
    )
    assert second.residual_sum_squares == pytest.approx(first.residual_sum_squares)
    assert second.upper_scale == pytest.approx(first.upper_scale)


def test_fast_block_bound_matches_general_projection() -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(size=31)
    design = consecutive_block_design(y.size, 7)
    general = gaussian_projection_upper_scale(y, design, failure_probability=0.02)
    fast = gaussian_block_upper_scale(y, 7, failure_probability=0.02)
    assert fast.residual_sum_squares == pytest.approx(general.residual_sum_squares)
    assert fast.upper_scale == pytest.approx(general.upper_scale)
    assert fast.residual_degrees_of_freedom == general.residual_degrees_of_freedom


@pytest.mark.parametrize(
    "y,design,eta",
    [
        ([1.0], [[1.0]], 0.05),
        ([1.0, 2.0], [[1.0], [1.0]], 0.0),
        ([1.0, 2.0], [[1.0], [np.nan]], 0.05),
        ([1.0, 2.0], np.eye(2), 0.05),
    ],
)
def test_projection_scale_bound_rejects_invalid_inputs(y, design, eta) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        gaussian_projection_upper_scale(y, design, failure_probability=eta)
