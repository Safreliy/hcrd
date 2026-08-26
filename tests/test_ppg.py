import numpy as np

from hcrd.ppg import (
    artifact_mask,
    hcrd_candidate_bank,
    mask_events,
    match_event_cardinality,
    match_event_pairs,
    match_events,
    robust_bandpass,
    suppress_events,
)


def test_artifact_intervals_are_inclusive():
    intervals = np.asarray([[2, 4], [8, 8]])
    assert artifact_mask(10, intervals).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert mask_events([1, 2, 4, 5, 8, 9], intervals).tolist() == [1, 5, 9]


def test_event_matching_maximizes_cardinality_before_timing_error():
    result = match_events([10, 20], [6, 12], tolerance_samples=8)
    assert result.true_positive == 2
    assert result.false_positive == 0
    assert result.false_negative == 0
    assert result.absolute_errors.tolist() == [4, 8]
    assert match_event_pairs([10, 20], [6, 12], 8).tolist() == [[0, 0], [1, 1]]
    assert match_event_cardinality([10, 20], [6, 12], 8) == 2


def test_event_matching_reports_errors_and_f1():
    result = match_events([10, 30, 50], [9, 31, 80], tolerance_samples=3)
    assert (result.true_positive, result.false_positive, result.false_negative) == (2, 1, 1)
    assert np.isclose(result.f1, 2.0 / 3.0)
    assert sorted(result.absolute_errors.tolist()) == [1, 1]


def test_score_ordered_suppression_is_deterministic():
    selected = suppress_events(
        [100, 120, 300, 320],
        [0.6, 0.9, 0.5, 0.5],
        minimum_distance_samples=50,
        threshold=0.5,
    )
    assert selected.tolist() == [120, 300]


def test_multilevel_candidate_bank_is_finite_and_contains_pulse_candidates():
    sampling_frequency = 100.0
    time = np.arange(0.0, 12.0, 1.0 / sampling_frequency)
    signal = sum(
        np.exp(-0.5 * ((time - center) / 0.08) ** 2)
        for center in np.arange(1.0, 12.0, 1.0)
    )
    bank = hcrd_candidate_bank(
        signal,
        sampling_frequency,
        max_levels=5,
        window_seconds=4.0,
        halo_seconds=0.5,
    )
    assert bank.geometry.shape == (bank.positions.size, 5 * 15 + 20)
    assert bank.morphology.shape == (bank.positions.size, 7)
    assert np.all(np.isfinite(bank.geometry))
    assert np.all(np.isfinite(bank.morphology))
    assert bank.positions.size >= 8
    expected = np.arange(1.0, 12.0, 1.0) * sampling_frequency
    assert all(np.min(np.abs(bank.positions - item)) <= 5 for item in expected)


def test_bandpass_interpolates_isolated_missing_sensor_samples():
    sampling_frequency = 100.0
    time = np.arange(1000) / sampling_frequency
    signal = np.sin(2.0 * np.pi * 1.2 * time)
    signal[::100] = np.nan
    filtered = robust_bandpass(signal, sampling_frequency)
    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))
