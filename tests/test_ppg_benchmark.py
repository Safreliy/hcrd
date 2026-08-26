import numpy as np

from hcrd.ppg_benchmark import align_ecg_to_ppg, flatline_mask, interpolate_nonfinite


def test_nonfinite_interpolation_and_flatline_mask():
    signal = np.arange(100, dtype=float)
    signal[10] = np.nan
    signal[40:70] = 5.0
    interpolated = interpolate_nonfinite(signal)
    assert np.all(np.isfinite(interpolated))
    mask = flatline_mask(signal, sampling_frequency=100.0, minimum_duration_seconds=0.2)
    assert mask[10]
    assert np.all(mask[40:70])
    assert not mask[30]


def test_piecewise_alignment_recovers_known_lag():
    reference = np.arange(100, 2100, 100)
    prediction = reference + 23
    aligned, lags = align_ecg_to_ppg(
        reference,
        prediction,
        sampling_frequency=100.0,
        maximum_lag_seconds=1.0,
        lag_increment_seconds=0.01,
        tolerance_seconds=0.05,
    )
    assert np.all(lags == 19)  # smallest lag within the strict < 50 ms window
    assert np.all(np.abs(aligned - prediction) < 5)


def test_piecewise_alignment_expands_last_nominal_block_like_ppg_beats():
    reference = np.arange(100, 80100, 100)
    prediction = reference.copy()
    prediction[:300] += 10
    prediction[300:600] += 20
    prediction[600:] += 30
    _, lags = align_ecg_to_ppg(
        reference,
        prediction,
        sampling_frequency=100.0,
        maximum_lag_seconds=0.5,
        lag_increment_seconds=0.01,
        tolerance_seconds=0.02,
    )
    assert np.all(lags[:300] == 9)
    assert np.all(lags[300:] == 19)
