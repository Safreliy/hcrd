"""Tests for the common-scale noise-certified HCRD score."""

from __future__ import annotations

import numpy as np

from hcrd import (
    certified_hcrd_area_score,
    certify_hcrd_hierarchy_margin,
    hcrd_hierarchy_decision_radius,
    stochastic_hierarchy_agreement_lower_bound,
    decompose_sparse,
)


def test_affine_signal_inside_supnorm_ball_has_zero_score() -> None:
    n = 101
    x = np.linspace(0.0, 10.0, n)
    sigma = 0.2
    delta = 0.05
    epsilon = sigma * np.sqrt(2.0 * np.log(2.0 * n / delta))
    perturbation = 0.9 * epsilon * np.sin(np.arange(n))
    signal = 3.0 - 0.7 * x + perturbation
    result = certified_hcrd_area_score(
        signal,
        x=x,
        noise_sigma=sigma,
        confidence_delta=delta,
        max_levels=8,
    )
    np.testing.assert_array_equal(result.score, np.zeros(n))


def test_common_scale_score_preserves_vertical_units() -> None:
    signal = np.zeros(41)
    signal[20] = 10.0
    first = certified_hcrd_area_score(signal, noise_sigma=0.01)
    second = certified_hcrd_area_score(3.0 * signal, noise_sigma=0.03)
    np.testing.assert_allclose(second.score, 3.0 * first.score)
    np.testing.assert_allclose(second.densities, 3.0 * first.densities)


def test_larger_curvature_multiplier_keeps_affine_null_certificate() -> None:
    n = 61
    x = np.linspace(-2.0, 4.0, n)
    sigma = 0.1
    delta = 0.1
    epsilon = sigma * np.sqrt(2.0 * np.log(2.0 * n / delta))
    signal = 1.5 + 0.25 * x + 0.95 * epsilon * np.cos(np.arange(n))
    result = certified_hcrd_area_score(
        signal,
        x=x,
        noise_sigma=sigma,
        confidence_delta=delta,
        curvature_multiplier=2.5,
    )
    assert result.curvature_multiplier == 2.5
    np.testing.assert_array_equal(result.score, np.zeros(n))


def test_parameter_validation() -> None:
    with np.testing.assert_raises(ValueError):
        certified_hcrd_area_score([0.0, 1.0], noise_sigma=-1.0)
    with np.testing.assert_raises(ValueError):
        certified_hcrd_area_score(
            [0.0, 1.0], noise_sigma=1.0, confidence_delta=1.0
        )
    with np.testing.assert_raises(ValueError):
        certified_hcrd_area_score(
            [0.0, 1.0], noise_sigma=1.0, curvature_multiplier=0.99
        )


def test_affine_reference_has_certified_hierarchy_margin() -> None:
    x = np.linspace(0.0, 2.0, 51)
    reference = 2.0 - 0.5 * x
    certificate = certify_hcrd_hierarchy_margin(
        reference,
        x=x,
        noise_sigma=0.1,
        confidence_delta=0.05,
        curvature_multiplier=2.0,
    )
    assert certificate.certified
    assert certificate.levels_checked == 1
    np.testing.assert_array_equal(certificate.knot_sets[0], [0, 50])


def test_equal_transition_magnitudes_are_not_overcertified() -> None:
    reference = np.asarray([0.0, 1.0, 3.0, 4.0, 3.0, 1.0, 0.0])
    certificate = certify_hcrd_hierarchy_margin(
        reference,
        noise_sigma=1e-3,
        confidence_delta=0.05,
    )
    # Symmetric opposite-curvature comparisons have no strict magnitude margin.
    assert not certificate.certified


def test_decision_radius_exposes_strict_multilevel_input_ball() -> None:
    reference = np.r_[np.zeros(10), [1.0, 3.0, 7.0, 4.0, 1.0], np.zeros(10)]
    radius = hcrd_hierarchy_decision_radius(
        reference, curvature_tolerance=0.1, max_levels=8
    )
    assert radius.input_radius == 0.025
    assert radius.levels_checked == 3
    rng = np.random.default_rng(20260826)
    for _ in range(100):
        perturbation = rng.uniform(
            -0.99 * radius.input_radius,
            0.99 * radius.input_radius,
            size=reference.size,
        )
        observed = decompose_sparse(
            reference + perturbation,
            atol=radius.curvature_tolerance,
            rtol=0.0,
            max_levels=8,
        )
        assert all(
            np.array_equal(expected, actual)
            for expected, actual in zip(
                radius.knot_sets, observed.knot_sets, strict=True
            )
        )


def test_tied_transition_has_zero_decision_radius() -> None:
    reference = np.asarray([0.0, 1.0, 3.0, 4.0, 3.0, 1.0, 0.0])
    radius = hcrd_hierarchy_decision_radius(
        reference, curvature_tolerance=0.1
    )
    assert radius.input_radius == 0.0
    assert radius.transition_comparison_margin_min == 0.0


def test_stochastic_margin_bound_uses_both_small_ball_and_noise_terms() -> None:
    strong = stochastic_hierarchy_agreement_lower_bound(
        sample_count=100,
        noise_sigma=0.01,
        radius=0.05,
        margin_failure_probability=0.02,
    )
    weak = stochastic_hierarchy_agreement_lower_bound(
        sample_count=100,
        noise_sigma=0.015,
        radius=0.05,
        margin_failure_probability=0.02,
    )
    assert 0.0 < weak < strong < 1.0
    assert stochastic_hierarchy_agreement_lower_bound(
        sample_count=100,
        noise_sigma=0.0,
        radius=0.05,
        margin_failure_probability=0.02,
    ) == 0.98


def test_certified_nonaffine_hierarchy_agrees_inside_input_ball() -> None:
    reference = np.r_[np.zeros(10), [1.0, 3.0, 7.0, 4.0, 1.0], np.zeros(10)]
    certificate = certify_hcrd_hierarchy_margin(
        reference,
        noise_sigma=1e-3,
        confidence_delta=0.05,
        curvature_multiplier=2.0,
    )
    assert certificate.certified
    rng = np.random.default_rng(11)
    for _ in range(100):
        noise = rng.uniform(
            -certificate.input_radius,
            certificate.input_radius,
            size=reference.size,
        )
        noisy = decompose_sparse(
            reference + noise,
            atol=certificate.curvature_tolerance,
            rtol=0.0,
            max_levels=8,
        )
        assert all(
            np.array_equal(expected, observed)
            for expected, observed in zip(
                certificate.knot_sets, noisy.knot_sets, strict=True
            )
        )
