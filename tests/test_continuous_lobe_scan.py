"""Tests for the continuous-class entropy and sieve scan bridge."""

from math import log, sqrt
from statistics import NormalDist

import numpy as np
import pytest

from hcrd.continuous_lobe_scan import (
    asymmetric_triangular_lobes,
    affine_residual_subspace_rank,
    continuous_scan_detection_threshold,
    continuous_scan_localization_sufficient_norm,
    continuous_scan_power_sufficient_norm,
    parameter_entropy_integral_upper,
    scan_lobe_sieve,
    signed_entropy_integral_upper,
    sieve_power_sufficient_norm,
    subspace_scan_detection_threshold,
    subspace_scan_localization_sufficient_norm,
    subspace_scan_power_sufficient_norm,
    triangular_lobe_lipschitz_certificate,
)


def test_asymmetric_triangular_lobes_have_geometric_support_and_apex():
    x = np.linspace(0.0, 1.0, 101)
    parameters = np.array([[0.5, 0.6, 0.25], [0.5, 0.6, 0.75]])
    lobes = asymmetric_triangular_lobes(x, parameters)
    assert lobes.shape == (2, x.size)
    assert np.all(lobes[:, (x < 0.2) | (x > 0.8)] == 0.0)
    assert np.argmax(lobes[0]) < np.argmax(lobes[1])
    assert np.max(lobes[0]) == pytest.approx(1.0)
    assert np.max(lobes[1]) == pytest.approx(1.0)


def test_sieve_scan_recovers_a_fixed_grid_lobe():
    x = np.linspace(0.0, 1.0, 129)
    parameters = np.array(
        [
            [0.35, 0.30, 0.50],
            [0.50, 0.30, 0.50],
            [0.65, 0.30, 0.50],
        ]
    )
    raw = asymmetric_triangular_lobes(x, parameters)
    from hcrd import residualized_lobe_dictionary

    dictionary = residualized_lobe_dictionary(raw, x=x)
    observation = 8.0 * dictionary[1]
    result = scan_lobe_sieve(
        observation,
        raw,
        parameters,
        noise_sigma=1.0,
        canonical_mesh_radius=0.1,
        alpha=0.05,
        x=x,
    )
    assert result.rejected
    assert result.selected_index == 1
    assert np.array_equal(result.selected_parameter, parameters[1])
    assert result.continuous_signal_factor == pytest.approx(0.995)


def test_sieve_approximation_cost_is_explicit_and_monotone():
    exact = sieve_power_sufficient_norm(50, 0.05, 0.2, 0.0)
    coarse = sieve_power_sufficient_norm(50, 0.05, 0.2, 0.5)
    assert coarse > exact
    with pytest.raises(ValueError):
        sieve_power_sufficient_norm(50, 0.05, 0.2, sqrt(2.0))


def test_subspace_scan_has_rank_aware_level_and_exact_pointwise_power():
    x = np.linspace(0.0, 1.0, 129)
    rank = affine_residual_subspace_rank(x)
    assert rank == 127
    threshold = subspace_scan_detection_threshold(rank, 0.05)
    expected = sqrt(rank + 2.0 * sqrt(rank * log(20.0)) + 2.0 * log(20.0))
    assert threshold == pytest.approx(expected)
    sufficient = subspace_scan_power_sufficient_norm(rank, 0.05, 0.2)
    z = NormalDist().inv_cdf(0.8)
    assert sufficient == pytest.approx(threshold + z)
    assert NormalDist().cdf(sufficient - threshold) == pytest.approx(0.8)
    assert sufficient < 14.0
    localization = subspace_scan_localization_sufficient_norm(
        rank, 0.1, identifiability_gap=0.25
    )
    assert localization > sufficient
    with pytest.raises(ValueError):
        subspace_scan_localization_sufficient_norm(rank, 0.1, 0.0)


def test_subspace_level_bound_covers_gaussian_norms():
    rank = 17
    alpha = 0.05
    threshold = subspace_scan_detection_threshold(rank, alpha)
    rng = np.random.default_rng(20260826)
    squared_norms = rng.chisquare(rank, size=20000)
    observed = np.mean(squared_norms > threshold**2)
    assert observed <= alpha


def test_parameter_entropy_bound_and_continuous_thresholds_are_ordered():
    entropy = parameter_entropy_integral_upper(
        [1.0, 0.5, 0.4], template_lipschitz=2.0
    )
    assert entropy > 0.0
    signed = signed_entropy_integral_upper(entropy, canonical_diameter=2.0)
    assert signed > entropy
    threshold = continuous_scan_detection_threshold(entropy, 0.05)
    power = continuous_scan_power_sufficient_norm(entropy, 0.05, 0.2)
    localization = continuous_scan_localization_sufficient_norm(
        signed, 0.1, identifiability_gap=0.25
    )
    assert power > threshold
    assert localization > power


def test_singleton_family_has_zero_entropy_but_nonzero_gaussian_threshold():
    entropy = parameter_entropy_integral_upper([0.0, 0.0], 3.0)
    assert entropy == 0.0
    assert continuous_scan_detection_threshold(entropy, 0.05) == pytest.approx(
        sqrt(2.0 * np.log(20.0))
    )


def test_duplicate_templates_expose_localization_identifiability_failure():
    x = np.linspace(0.0, 1.0, 65)
    raw = asymmetric_triangular_lobes(x, [[0.5, 0.4, 0.5]])
    duplicated = np.vstack([raw, raw])
    from hcrd import residualized_lobe_dictionary

    dictionary = residualized_lobe_dictionary(duplicated, x=x)
    assert np.dot(dictionary[0], dictionary[1]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        continuous_scan_localization_sufficient_norm(1.0, 0.1, 0.0)


def test_interior_triangular_family_has_analytic_lipschitz_certificate():
    x = np.linspace(0.0, 1.0, 129)
    certificate = triangular_lobe_lipschitz_certificate(
        x,
        center_bounds=(0.35, 0.65),
        width_bounds=(0.20, 0.40),
        apex_fraction_bounds=(0.25, 0.75),
    )
    assert certificate.residual_norm_lower > 0.0
    assert certificate.normalized_template_lipschitz > 0.0
    assert certificate.canonical_diameter_upper <= 2.0
    assert certificate.entropy_integral_upper > 0.0

    rng = np.random.default_rng(20260825)
    parameters = np.c_[
        rng.uniform(0.35, 0.65, 100),
        rng.uniform(0.20, 0.40, 100),
        rng.uniform(0.25, 0.75, 100),
    ]
    raw = asymmetric_triangular_lobes(x, parameters)
    from hcrd import residualized_lobe_dictionary

    dictionary = residualized_lobe_dictionary(raw, x=x)
    design = np.c_[np.ones(x.size), x]
    fitted = (design @ np.linalg.lstsq(design, raw.T, rcond=None)[0]).T
    projection_residual = raw - fitted
    assert (
        np.min(np.linalg.norm(projection_residual, axis=1)) + 1e-12
        >= certificate.residual_norm_lower
    )
    for left, right in zip(range(0, 50), range(50, 100), strict=True):
        observed = np.linalg.norm(dictionary[left] - dictionary[right])
        upper = certificate.normalized_template_lipschitz * np.linalg.norm(
            parameters[left] - parameters[right]
        )
        assert observed <= upper + 1e-12


def test_triangular_certificate_rejects_boundary_and_coarse_grid_failures():
    x = np.linspace(0.0, 1.0, 17)
    with pytest.raises(ValueError, match="first sample"):
        triangular_lobe_lipschitz_certificate(
            x,
            center_bounds=(0.20, 0.60),
            width_bounds=(0.20, 0.40),
            apex_fraction_bounds=(0.25, 0.75),
        )
    with pytest.raises(ValueError, match="too coarse"):
        triangular_lobe_lipschitz_certificate(
            x,
            center_bounds=(0.30, 0.70),
            width_bounds=(0.04, 0.06),
            apex_fraction_bounds=(0.45, 0.55),
        )
