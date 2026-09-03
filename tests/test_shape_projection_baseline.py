import numpy as np
import pytest

from shapecontrast import gaussian_pointwise_shape_projection


def test_exact_s_shape_retains_transition_with_zero_noise() -> None:
    x = np.linspace(0.0, 1.0, 41)
    mean = -(x - 0.5) ** 3
    result = gaussian_pointwise_shape_projection(
        x, mean, noise_scale=0.0, alpha=0.05
    )
    assert result.interval is not None
    assert result.left <= 0.5 <= result.right
    assert result.minimum_feasible_cut is not None
    assert result.maximum_feasible_cut is not None


def test_projection_handles_irregular_design_and_jump() -> None:
    x = np.linspace(0.02, 0.98, 60) ** 1.7
    mean = x + (x >= 0.35)
    result = gaussian_pointwise_shape_projection(
        x, mean, noise_scale=0.0, alpha=0.05
    )
    assert result.interval is not None
    assert result.left <= 0.35 <= result.right


def test_projection_can_reject_global_s_shape() -> None:
    x = np.linspace(0.0, 1.0, 20)
    alternating = np.tile([0.0, 1.0], 10)
    result = gaussian_pointwise_shape_projection(
        x, alternating, noise_scale=0.0, alpha=0.05
    )
    assert result.empty
    assert result.interval is None


@pytest.mark.parametrize(
    "x, y, sigma, alpha",
    [
        ([0.0, 0.5, 0.4], [0.0, 0.0, 0.0], 1.0, 0.05),
        ([0.0, 0.5, 1.0], [0.0, 0.0], 1.0, 0.05),
        ([0.0, 0.5, 1.0], [0.0, 0.0, 0.0], -1.0, 0.05),
        ([0.0, 0.5, 1.0], [0.0, 0.0, 0.0], 1.0, 0.0),
    ],
)
def test_projection_rejects_invalid_inputs(
    x: list[float], y: list[float], sigma: float, alpha: float
) -> None:
    with pytest.raises(ValueError):
        gaussian_pointwise_shape_projection(
            x, y, noise_scale=sigma, alpha=alpha
        )
