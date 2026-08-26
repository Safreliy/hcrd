import numpy as np

from hcrd.core import decompose, find_convexity_knots
from hcrd.robust import (
    adaptive_gaussian_guided_decompose,
    estimate_noise_sigma,
    familywise_curvature_threshold,
    gaussian_guided_decompose,
    robust_decompose,
    stability_certificate,
)


def test_global_discontinuity_counterexample():
    epsilon = 1e-8
    plus = np.array([-2.0, 0.0, 2.0 + epsilon, -2.0])
    minus = np.array([-2.0, 0.0, 2.0 - epsilon, -2.0])
    plus_baseline = decompose(
        plus, boundary_rule="legacy", atol=0.0, rtol=0.0, max_levels=1
    ).levels[0].baseline
    minus_baseline = decompose(
        minus, boundary_rule="legacy", atol=0.0, rtol=0.0, max_levels=1
    ).levels[0].baseline
    np.testing.assert_allclose(np.max(np.abs(plus - minus)), 2.0 * epsilon, rtol=1e-8)
    assert np.max(np.abs(plus_baseline - minus_baseline)) > 3.9


def test_sign_cell_certificate_preserves_knots_and_is_locally_nonexpansive():
    signal = np.array([0.0, 1.0, -0.5, 2.0, -1.0, 0.25])
    radius = stability_certificate(signal)
    assert radius > 0
    perturbation = np.array([0.2, -0.5, 0.1, 0.7, -0.3, 0.4]) * (0.5 * radius)
    first = decompose(signal, atol=0.0, rtol=0.0, max_levels=1).levels[0]
    second = decompose(
        signal + perturbation, atol=0.0, rtol=0.0, max_levels=1
    ).levels[0]
    np.testing.assert_array_equal(first.knots, second.knots)
    assert np.max(np.abs(second.baseline - first.baseline)) <= np.max(
        np.abs(perturbation)
    ) + 1e-14
    assert np.max(np.abs(second.detail - first.detail)) <= 2.0 * np.max(
        np.abs(perturbation)
    ) + 1e-14


def test_signs_alone_do_not_certify_centred_magnitude_tie():
    signal = np.array([0.0, 0.0, 1.0, 3.0, 4.0])
    epsilon = 1e-7
    changed = np.array([0.0, 0.0, 1.0, 3.0 - epsilon, 4.0 - 2.0 * epsilon])
    np.testing.assert_array_equal(np.sign(np.diff(np.diff(signal))), [1.0, 1.0, -1.0])
    np.testing.assert_array_equal(
        np.sign(np.diff(np.diff(changed))), [1.0, 1.0, -1.0]
    )
    assert stability_certificate(signal) == 0.0
    first = find_convexity_knots(signal, atol=0.0, rtol=0.0)
    second = find_convexity_knots(changed, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(first, [0, 3, 4])
    np.testing.assert_array_equal(second, [0, 2, 4])


def test_centred_certificate_preserves_sign_and_magnitude_decisions():
    signal = np.array([0.0, 0.0, 1.0, 2.5, 2.75, 2.0])
    radius = stability_certificate(signal)
    assert radius > 0.0
    perturbation = np.array([0.3, -0.2, 0.1, -0.4, 0.2, -0.1]) * radius
    first = find_convexity_knots(signal, atol=0.0, rtol=0.0)
    second = find_convexity_knots(
        signal + perturbation, atol=0.0, rtol=0.0
    )
    np.testing.assert_array_equal(first, second)


def test_noise_estimator_has_correct_order_for_gaussian_noise():
    rng = np.random.default_rng(101)
    sigma = 0.2
    estimates = [estimate_noise_sigma(rng.normal(0.0, sigma, size=4096)) for _ in range(20)]
    assert abs(np.median(estimates) - sigma) < 0.02


def test_robust_threshold_suppresses_small_curvature_changes():
    rng = np.random.default_rng(103)
    signal = 0.01 * rng.normal(size=513)
    raw_knots = find_convexity_knots(signal, atol=0.0, rtol=0.0)
    robust = robust_decompose(signal, z_score=3.5)
    robust_knots = robust.decomposition.levels[0].knots
    assert robust_knots.size < raw_knots.size


def test_familywise_threshold_scales_with_sigma_and_sample_count():
    first = familywise_curvature_threshold(100, 0.1, delta=0.05)
    second = familywise_curvature_threshold(1000, 0.1, delta=0.05)
    doubled = familywise_curvature_threshold(100, 0.2, delta=0.05)
    assert second > first
    np.testing.assert_allclose(doubled, 2.0 * first)


def test_guided_variant_retains_exact_reconstruction():
    rng = np.random.default_rng(109)
    signal = rng.normal(size=257)
    result = gaussian_guided_decompose(signal, smoothing_sigma=2.0)
    np.testing.assert_allclose(result.reconstruct(), signal, atol=2e-13)


def test_adaptive_guided_variant_retains_exact_reconstruction():
    rng = np.random.default_rng(113)
    signal = rng.normal(size=257)
    result = adaptive_gaussian_guided_decompose(signal)
    np.testing.assert_allclose(result.guided.reconstruct(), signal, atol=2e-13)
