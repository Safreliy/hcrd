import numpy as np
import pytest
from scipy.stats import t

from shapecontrast import (
    build_shape_contrast_family,
    invert_s_shaped_inflection,
    replicated_t_shape_band,
)


def test_replicated_band_matches_direct_student_calculation() -> None:
    x = np.linspace(0.0, 1.0, 16)
    family = build_shape_contrast_family(x, block_sizes=(1, 2))
    rng = np.random.default_rng(12)
    common = rng.normal(size=(9, 1))
    curves = x[None, :] ** 3 + common + rng.normal(scale=0.1, size=(9, 16))

    band = replicated_t_shape_band(family, curves, alpha=0.05)
    scores = family.means_many(curves)
    critical = t.ppf(1.0 - 0.05 / (2.0 * family.contrast_count), 8)

    np.testing.assert_allclose(band.estimate, np.mean(scores, axis=0))
    np.testing.assert_allclose(
        band.radius, critical * np.std(scores, axis=0, ddof=1) / 3.0
    )
    assert band.replicate_count == 9
    assert band.degrees_of_freedom == 8


def test_replicated_band_allows_cross_design_dependence() -> None:
    x = np.linspace(0.0, 1.0, 12)
    family = build_shape_contrast_family(x, block_sizes=(1,))
    rng = np.random.default_rng(34)
    shared = rng.normal(scale=2.0, size=(20, 1))
    curves = -(x - 0.5) ** 3 + shared

    band = replicated_t_shape_band(family, curves, alpha=0.05)
    np.testing.assert_allclose(band.radius, 0.0, atol=1e-14)
    result = invert_s_shaped_inflection(family, band, domain=(0.0, 1.0))
    assert result.interval is not None


@pytest.mark.parametrize(
    "curves, alpha, message",
    [
        (np.ones((1, 8)), 0.05, "at least two"),
        (np.ones((3, 7)), 0.05, "one replicate"),
        (np.ones((3, 8)), 1.0, "alpha"),
    ],
)
def test_replicated_band_rejects_invalid_input(
    curves: np.ndarray, alpha: float, message: str
) -> None:
    family = build_shape_contrast_family(np.linspace(0.0, 1.0, 8))
    with pytest.raises(ValueError, match=message):
        replicated_t_shape_band(family, curves, alpha=alpha)
