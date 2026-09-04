import numpy as np
import pytest

from shapecontrast import (
    design_identified_transition_set,
    gaussian_pointwise_shape_projection,
)


def test_exact_s_shape_retains_transition_with_zero_noise() -> None:
    x = np.linspace(0.0, 1.0, 41)
    mean = -((x - 0.5) ** 3)
    result = gaussian_pointwise_shape_projection(x, mean, noise_scale=0.0, alpha=0.05)
    assert result.interval is not None
    assert result.left <= 0.5 <= result.right
    assert result.minimum_feasible_cut is not None
    assert result.maximum_feasible_cut is not None


def test_projection_handles_irregular_design_and_jump() -> None:
    x = np.linspace(0.02, 0.98, 60) ** 1.7
    mean = x + (x >= 0.35)
    result = gaussian_pointwise_shape_projection(x, mean, noise_scale=0.0, alpha=0.05)
    assert result.interval is not None
    assert result.left <= 0.35 <= result.right


def test_boundary_cuts_cover_the_declared_domain() -> None:
    x = np.linspace(0.1, 0.9, 17)
    affine_mean = 2.0 * x - 1.0
    result = gaussian_pointwise_shape_projection(
        x,
        affine_mean,
        noise_scale=0.0,
        alpha=0.05,
        domain=(0.0, 1.0),
    )

    assert result.interval == (0.0, 1.0)


def test_zero_noise_projection_contains_full_design_identified_target() -> None:
    x = np.array([0.1, 0.24, 0.41, 0.63, 0.9])
    mean = np.array([-0.2, 0.1, 0.32, 0.45, 0.49])
    domain = (0.0, 1.0)
    target = design_identified_transition_set(x, mean, domain=domain)
    result = gaussian_pointwise_shape_projection(
        x, mean, noise_scale=0.0, alpha=0.05, domain=domain
    )

    assert target.hull is not None
    assert result.interval is not None
    assert result.left <= target.hull[0]
    assert target.hull[1] <= result.right


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
        gaussian_pointwise_shape_projection(x, y, noise_scale=sigma, alpha=alpha)


@pytest.mark.parametrize(
    "domain",
    [
        (0.1, 1.0),
        (0.0, 0.9),
        (1.0, 0.0),
        (float("nan"), 1.0),
    ],
)
def test_projection_rejects_invalid_domain(domain: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        gaussian_pointwise_shape_projection(
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 0.0],
            noise_scale=1.0,
            alpha=0.05,
            domain=domain,
        )
