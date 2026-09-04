from dataclasses import asdict

import numpy as np
import pytest

import shapecontrast as sci
from hcrd.shape_inflection_confidence import (
    build_shape_contrast_family as build_dense_family,
)
from hcrd.shape_inflection_confidence import (
    calibrate_gaussian_shape_contrast_max as calibrate_dense,
)
from hcrd.shape_inflection_confidence import (
    gaussian_bonferroni_shape_band as dense_band,
)
from hcrd.shape_inflection_confidence import (
    invert_s_shaped_inflection as invert_dense,
)


@pytest.mark.parametrize(
    "x",
    [
        np.arange(1, 98, dtype=float) / 98.0,
        np.linspace(0.01, 0.99, 97) ** 1.4,
    ],
)
def test_matrix_free_family_matches_frozen_dense_implementation(x):
    rng = np.random.default_rng(20260904)
    y = np.sin(4.0 * x) + rng.normal(0.0, 0.05, size=x.size)
    dense = build_dense_family(x, separation_multipliers=(1, 2, 4))
    compact = sci.build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )

    assert compact.contrast_count == dense.contrast_count
    np.testing.assert_array_equal(compact.start_index, dense.start_index)
    np.testing.assert_array_equal(compact.block_size, dense.block_size)
    np.testing.assert_array_equal(compact.separation, dense.separation)
    np.testing.assert_allclose(compact.support_left, dense.support_left)
    np.testing.assert_allclose(compact.support_right, dense.support_right)
    np.testing.assert_allclose(compact.weight_l2, dense.weight_l2, atol=2e-15)
    np.testing.assert_allclose(compact.means(y), dense.means(y), atol=2e-13)


def test_matrix_free_band_and_inversion_match_frozen_implementation():
    rng = np.random.default_rng(7)
    x = np.arange(1, 501, dtype=float) / 501.0
    y = x - (x - 0.5) ** 3 + rng.normal(0.0, 0.01, size=x.size)
    dense = build_dense_family(x, separation_multipliers=(1, 2, 4))
    compact = sci.build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    old_band = dense_band(dense, y, noise_scale=0.01, alpha=0.05)
    new_band = sci.gaussian_bonferroni_shape_band(
        compact, y, noise_scale=0.01, alpha=0.05
    )
    old_set = invert_dense(dense, old_band, domain=(0.0, 1.0))
    new_set = sci.invert_s_shaped_inflection(
        compact, new_band, domain=(0.0, 1.0)
    )

    np.testing.assert_allclose(new_band.estimate, old_band.estimate, atol=2e-13)
    np.testing.assert_allclose(new_band.radius, old_band.radius, atol=2e-15)
    assert asdict(new_set) == asdict(old_set)


def test_matrix_free_inversion_treats_touching_boundaries_as_empty():
    x = np.arange(1, 8, dtype=float) / 8.0
    family = sci.build_shape_contrast_family(x, block_sizes=(1,))
    positive_index = int(np.flatnonzero(family.start_index == 2)[0])
    negative_index = int(np.flatnonzero(family.start_index == 0)[0])
    estimate = np.zeros(family.contrast_count)
    estimate[positive_index] = 2.0
    estimate[negative_index] = -2.0
    band = sci.ShapeContrastBand(
        estimate=estimate,
        radius=np.ones(family.contrast_count),
        critical_value=1.0,
        alpha=0.05,
        noise_scale=1.0,
    )

    confidence_set = sci.invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )
    assert confidence_set.left == confidence_set.right
    assert confidence_set.empty


def test_matrix_free_gaussian_calibration_matches_dense_version():
    x = np.linspace(0.01, 0.99, 41) ** 1.3
    dense = build_dense_family(x, separation_multipliers=(1, 2))
    compact = sci.build_shape_contrast_family(
        x, separation_multipliers=(1, 2)
    )
    old = calibrate_dense(
        dense,
        alpha=0.10,
        calibration_failure_probability=0.01,
        simulations=400,
        seed=19,
        chunk_size=37,
    )
    new = sci.calibrate_gaussian_shape_contrast_max(
        compact,
        alpha=0.10,
        calibration_failure_probability=0.01,
        simulations=400,
        seed=19,
        chunk_size=37,
    )
    assert new.order_statistic_rank == old.order_statistic_rank
    assert new.critical_value == pytest.approx(old.critical_value, abs=2e-13)


def test_exact_contrast_variances_match_dense_weights():
    x = np.linspace(0.01, 0.99, 73) ** 1.7
    variances = np.linspace(0.4, 1.6, x.size)
    dense = build_dense_family(x, separation_multipliers=(1, 2, 4))
    compact = sci.build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    expected = (dense.operator**2) @ variances
    np.testing.assert_allclose(
        compact.contrast_variances(variances), expected, atol=2e-14
    )


def test_batched_evaluation_matches_one_curve_at_a_time():
    rng = np.random.default_rng(91)
    x = np.linspace(0.01, 0.99, 79) ** 1.2
    responses = rng.normal(size=(7, x.size))
    family = sci.build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    expected = np.vstack([family.means(row) for row in responses])
    np.testing.assert_allclose(family.means_many(responses), expected, atol=2e-14)


def test_large_uniform_family_has_compact_storage_and_finite_output():
    n = 100_000
    x = np.arange(1, n + 1, dtype=float) / (n + 1)
    family = sci.build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    estimates = family.means(np.sin(2.0 * np.pi * x))

    assert family.contrast_count > 5 * n
    assert family.stored_bytes < 60 * n
    assert not hasattr(family, "operator")
    assert estimates.shape == (family.contrast_count,)
    assert np.all(np.isfinite(estimates))


def test_standalone_scale_helpers_are_available():
    y = np.linspace(0.0, 1.0, 100) + 0.01 * np.sin(np.arange(100))
    iid = sci.gaussian_block_upper_scale(
        y, 50, failure_probability=0.01
    )
    varying = sci.gaussian_heteroskedastic_upper_envelope(
        y, max_to_mean_variance_ratio=2.0, failure_probability=0.05
    )
    assert iid.upper_scale > 0.0
    assert varying.upper_scale > 0.0


def test_standalone_projection_scale_accepts_no_nuisance_columns():
    y = np.array([1.0, -2.0, 0.5, 3.0])
    result = sci.gaussian_projection_upper_scale(
        y, np.empty((y.size, 0)), failure_probability=0.05
    )
    assert result.nuisance_rank == 0
    assert result.residual_sum_squares == pytest.approx(float(y @ y))
