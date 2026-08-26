import numpy as np

from hcrd.core import discrete_curvature, find_convexity_knots
from hcrd.recovery import (
    alternating_parabolic_chord_lobes,
    amplitude_for_curvature_ratio,
    finite_sample_recovery_thresholds,
)


def test_parabolic_class_has_isolated_zero_curvature_transitions():
    signal = alternating_parabolic_chord_lobes(
        seed=3,
        lobes=4,
        samples_per_lobe=8,
        amplitude=2.5,
        noise_sigma=0.0,
        spacing=0.25,
    )
    curvature = discrete_curvature(signal.observed, signal.x)
    boundary_positions = signal.knots[1:-1]
    np.testing.assert_allclose(curvature[boundary_positions - 1], 0.0, atol=1e-13)
    active = np.ones(curvature.size, dtype=bool)
    active[boundary_positions - 1] = False
    np.testing.assert_allclose(
        np.abs(curvature[active]), signal.curvature_magnitude, atol=1e-12
    )


def test_noiseless_parabolic_class_recovers_all_lobe_boundaries():
    signal = alternating_parabolic_chord_lobes(
        seed=7,
        lobes=6,
        samples_per_lobe=5,
        amplitude=1.25,
        noise_sigma=0.0,
    )
    knots = find_convexity_knots(
        signal.observed, signal.x, atol=0.0, rtol=0.0
    )
    np.testing.assert_array_equal(knots, signal.knots)


def test_recovery_threshold_parameterization_hits_requested_ratio():
    n = 65
    m = 8
    threshold, radius = finite_sample_recovery_thresholds(
        n, 0.3, delta=0.05, spacing=0.5
    )
    amplitude = amplitude_for_curvature_ratio(
        2.25, threshold, m, spacing=0.5
    )
    gamma = 8.0 * amplitude / (m**2 * 0.5)
    assert np.isclose(gamma / threshold, 2.25)
    assert threshold > 0
    assert radius > 0


def test_equal_transition_amplitude_is_not_cosmetic():
    # Removing the isolated zero-curvature transition can move the centred
    # boundary by one sample even without noise.
    m = 4
    local = np.arange(m + 1, dtype=float) / m
    shape = 4.0 * local * (1.0 - local)
    detail = np.zeros(2 * m + 1)
    detail[: m + 1] += shape
    detail[m:] -= 0.25 * shape
    knots = find_convexity_knots(detail, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(knots, [0, 3, 8])
