import numpy as np

from hcrd.shape_inflection_confidence import (
    ShapeContrastBand,
    build_shape_contrast_family,
    calibrate_gaussian_shape_contrast_max,
    dyadic_block_sizes,
    gaussian_bonferroni_shape_band,
    gaussian_calibrated_shape_band,
    invert_s_shaped_inflection,
)


def test_dyadic_block_sizes_stop_when_three_blocks_no_longer_fit():
    assert dyadic_block_sizes(25) == (1, 2, 4, 8)


def test_chord_family_has_correct_sign_on_irregular_convex_and_concave_data():
    x = np.array([0.0, 0.04, 0.11, 0.23, 0.39, 0.58, 0.76, 0.9, 1.0])
    family = build_shape_contrast_family(x)
    convex = family.means(x**2)
    concave = family.means(2.0 * x - x**2)
    assert np.all(convex >= -1e-14)
    assert np.all(concave <= 1e-14)


def test_separated_chord_blocks_remain_shape_valid():
    x = np.linspace(0.0, 1.0, 65) ** 1.4
    family = build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    assert np.any(family.separation > family.block_size)
    assert np.all(family.means(np.exp(x)) >= -1e-14)
    assert np.all(family.means(-np.exp(x)) <= 1e-14)


def test_affine_signal_returns_the_full_domain_without_noise():
    x = np.linspace(0.02, 0.98, 49)
    family = build_shape_contrast_family(x)
    band = gaussian_bonferroni_shape_band(
        family, 1.0 + 2.0 * x, noise_scale=0.0, alpha=0.05
    )
    confidence_set = invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )
    assert confidence_set.interval == (0.0, 1.0)


def test_noise_free_s_shape_produces_nontrivial_set_containing_root():
    x = np.arange(1, 201, dtype=float) / 201.0
    truth = x - (x - 0.5) ** 3
    family = build_shape_contrast_family(x)
    band = gaussian_bonferroni_shape_band(
        family, truth, noise_scale=0.0, alpha=0.05
    )
    confidence_set = invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )
    assert not confidence_set.empty
    assert confidence_set.left < 0.5 < confidence_set.right
    assert confidence_set.width < 0.25


def test_inversion_uses_support_extrema_of_certified_signs():
    x = np.arange(1, 13, dtype=float) / 13.0
    family = build_shape_contrast_family(x, block_sizes=(1,))
    estimate = np.zeros(family.contrast_count)
    lower = np.full(family.contrast_count, -1.0)
    upper = np.full(family.contrast_count, 1.0)
    lower[2] = 0.1
    estimate[2] = 0.2
    upper[-2] = -0.1
    estimate[-2] = -0.2
    band = ShapeContrastBand(
        estimate=estimate,
        lower=lower,
        upper=upper,
        radius=np.ones(family.contrast_count),
        critical_value=1.0,
        alpha=0.05,
        noise_scale=1.0,
    )
    confidence_set = invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )
    assert confidence_set.left == family.support_left[2]
    assert confidence_set.right == family.support_right[-2]


def test_touching_exclusion_boundaries_leave_no_candidate_point():
    x = np.arange(1, 8, dtype=float) / 8.0
    family = build_shape_contrast_family(x, block_sizes=(1,))
    positive_index = int(np.flatnonzero(family.start_index == 2)[0])
    negative_index = int(np.flatnonzero(family.start_index == 0)[0])
    assert family.support_left[positive_index] == family.support_right[negative_index]

    estimate = np.zeros(family.contrast_count)
    lower = np.full(family.contrast_count, -1.0)
    upper = np.full(family.contrast_count, 1.0)
    lower[positive_index] = 0.1
    upper[negative_index] = -0.1
    band = ShapeContrastBand(
        estimate=estimate,
        lower=lower,
        upper=upper,
        radius=np.ones(family.contrast_count),
        critical_value=1.0,
        alpha=0.05,
        noise_scale=1.0,
    )

    confidence_set = invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )
    assert confidence_set.left == confidence_set.right
    assert confidence_set.empty
    assert confidence_set.interval is None


def test_projection_cannot_increase_error_when_truth_is_inside_set():
    x = np.arange(1, 201, dtype=float) / 201.0
    truth = x - (x - 0.5) ** 3
    family = build_shape_contrast_family(x)
    band = gaussian_bonferroni_shape_band(
        family, truth, noise_scale=0.0, alpha=0.05
    )
    confidence_set = invert_s_shaped_inflection(
        family, band, domain=(0.0, 1.0)
    )
    candidate = 0.05
    projected = confidence_set.project(candidate)
    assert projected is not None
    assert abs(projected - 0.5) <= abs(candidate - 0.5)


def test_joint_gaussian_calibration_is_reproducible_and_applicable():
    x = np.arange(1, 65, dtype=float) / 65.0
    family = build_shape_contrast_family(
        x, separation_multipliers=(1, 2, 4)
    )
    first = calibrate_gaussian_shape_contrast_max(
        family,
        alpha=0.05,
        calibration_failure_probability=0.005,
        simulations=1000,
        seed=1234,
    )
    second = calibrate_gaussian_shape_contrast_max(
        family,
        alpha=0.05,
        calibration_failure_probability=0.005,
        simulations=1000,
        seed=1234,
    )
    assert first == second
    band = gaussian_calibrated_shape_band(
        family,
        x - (x - 0.5) ** 3,
        noise_scale=0.1,
        calibration=first,
    )
    assert band.estimate.shape == (family.contrast_count,)
    assert np.all(band.radius > 0.0)
