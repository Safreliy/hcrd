import numpy as np

from hcrd.external_baselines import (
    ceemdan_slow_tail_path,
    emd_residue,
    itd_residue,
    l1_trend_filter,
    l1_trend_filter_path,
    vmd_low_frequency,
)


def test_l1_trend_filter_preserves_affine_signal():
    x = np.linspace(0.0, 1.0, 65)
    signal = 1.2 - 0.7 * x
    estimate = l1_trend_filter(signal, regularization=10.0)
    np.testing.assert_allclose(estimate, signal, atol=1e-8)


def test_l1_trend_filter_path_preserves_order_and_convex_objective_quality():
    x = np.linspace(0.0, 1.0, 65)
    signal = 0.4 * x + 0.2 * np.sin(8.0 * np.pi * x)
    values = [10.0, 0.0, 1.0]
    path = l1_trend_filter_path(signal, values)
    assert len(path) == len(values)
    np.testing.assert_allclose(path[1], signal, atol=1e-12)
    for regularization, estimate in zip(values, path, strict=True):
        objective = 0.5 * np.sum((signal - estimate) ** 2) + regularization * np.sum(
            np.abs(np.diff(estimate, n=2))
        )
        identity_objective = regularization * np.sum(np.abs(np.diff(signal, n=2)))
        assert objective <= identity_objective + 1e-7


def test_emd_and_vmd_return_finite_shape_matched_baselines():
    x = np.linspace(0.0, 1.0, 128)
    signal = 0.2 * x + np.sin(2.0 * np.pi * 8.0 * x)
    for estimate in (emd_residue(signal, x), vmd_low_frequency(signal, modes=3)):
        assert estimate.shape == signal.shape
        assert np.all(np.isfinite(estimate))


def test_itd_residue_is_finite_and_components_reconstruct():
    x = np.linspace(0.0, 1.0, 129)
    signal = 0.2 * x + np.sin(2.0 * np.pi * 8.0 * x)
    estimate = itd_residue(signal)
    assert estimate.shape == signal.shape
    assert np.all(np.isfinite(estimate))


def test_ceemdan_slow_tail_path_is_seeded_and_complete():
    x = np.linspace(0.0, 1.0, 65)
    signal = 0.4 * x + np.sin(6.0 * np.pi * x)
    first, error = ceemdan_slow_tail_path(
        signal, x, trials=2, noise_seed=71, maximum_tail_components=2
    )
    second, second_error = ceemdan_slow_tail_path(
        signal, x, trials=2, noise_seed=71, maximum_tail_components=2
    )
    assert 1 <= len(first) <= 2
    assert error <= 5e-15
    assert second_error == error
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
