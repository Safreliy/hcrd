"""Compact-impulse score and event-metric checks."""

from __future__ import annotations

import numpy as np

from hcrd import (
    event_average_precision,
    hcrd_concentration_anomaly_score,
    intracellular_spike_times,
)


def test_intracellular_derivative_recovers_separated_events() -> None:
    rng = np.random.default_rng(3)
    signal = rng.normal(0.0, 0.01, 2000)
    truth = np.asarray([300, 900, 1500])
    for index in truth:
        signal[index : index + 3] += [0.0, 3.0, 0.0]
    detected = intracellular_spike_times(
        signal,
        sampling_rate=1000.0,
        threshold_sigma=10.0,
        refractory_seconds=0.005,
    )
    assert detected.size == truth.size
    assert np.all(np.abs(detected - truth) <= 2)


def test_event_average_precision_is_one_for_exact_isolated_peaks() -> None:
    score = np.zeros(100)
    truth = np.asarray([20, 50, 80])
    score[truth] = [3.0, 2.0, 1.0]
    assert event_average_precision(
        score,
        truth,
        tolerance_samples=1,
        refractory_samples=3,
    ) == 1.0


def test_concentration_score_is_affine_invariant() -> None:
    signal = np.zeros(81)
    signal[39:42] = [1.0, 5.0, 1.0]
    first = hcrd_concentration_anomaly_score(signal)
    second = hcrd_concentration_anomaly_score(7.0 * signal + np.arange(81) * 0.2)
    np.testing.assert_allclose(first, second, atol=1e-10, rtol=1e-10)
