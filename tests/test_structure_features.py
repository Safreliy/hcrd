import numpy as np

from hcrd.structure_features import (
    geometry_feature_names,
    hcrd_representation,
    hcrd_representation_batch,
    structure_feature_names,
    wavelet_channel_batch,
)


def test_full_channels_reconstruct_normalised_signal_and_keep_trend_last():
    x = np.linspace(0.0, 1.0, 96)
    signal = 0.4 * x + np.sin(8.0 * np.pi * x) + 0.2 * np.cos(15.0 * np.pi * x)
    result = hcrd_representation(signal, max_levels=5)
    expected = signal - np.median(signal)
    expected /= np.sqrt(np.mean(expected**2))
    np.testing.assert_allclose(np.sum(result.channels, axis=0), expected, atol=1e-12)
    assert result.channels.shape == (6, signal.size)


def test_structure_bank_has_stable_finite_layout_and_is_scale_invariant():
    signal = np.asarray([0, 1, 3, 1, 0, -2, -1, 0] * 4, dtype=float)
    first = hcrd_representation(signal, max_levels=3, spatial_bins=4, top_k=2)
    second = hcrd_representation(7.2 * signal + 31.0, max_levels=3, spatial_bins=4, top_k=2)
    assert first.structure_features.size == len(
        structure_feature_names(max_levels=3, spatial_bins=4, top_k=2)
    )
    assert first.geometry_features.size == len(geometry_feature_names(max_levels=3))
    assert np.all(np.isfinite(first.structure_features))
    np.testing.assert_allclose(first.structure_features, second.structure_features, atol=1e-11)


def test_parallel_batch_and_wavelet_control_shapes():
    signals = [
        np.sin(np.linspace(0, 6 * np.pi, 64) + phase) for phase in (0.0, 0.4, 0.8)
    ]
    channels, features, geometry = hcrd_representation_batch(
        signals, max_levels=3, spatial_bins=4, top_k=2, workers=1
    )
    wavelet = wavelet_channel_batch(signals, max_levels=3)
    assert channels.shape == wavelet.shape == (3, 4, 64)
    assert features.shape[0] == geometry.shape[0] == 3


def test_early_hierarchy_channels_pass_aeon_variance_guard_without_losing_reconstruction():
    signal = np.r_[np.linspace(0.0, 1.0, 32), np.linspace(1.0, 0.0, 32)]
    result = hcrd_representation(signal, max_levels=5)
    expected = signal - np.median(signal)
    expected /= np.sqrt(np.mean(expected**2))
    assert np.all(np.std(result.channels, axis=1) > 1e-7)
    np.testing.assert_allclose(np.sum(result.channels, axis=0), expected, atol=1e-12)
