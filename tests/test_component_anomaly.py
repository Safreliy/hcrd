import numpy as np
import pytest

from hcrd import (
    hcrd_component_iforest_scores,
    multiscale_area_density,
    multiscale_detail_series,
)


def test_signed_components_have_the_area_density_as_their_absolute_value():
    rng = np.random.default_rng(49)
    signal = rng.normal(size=129)
    details = multiscale_detail_series(signal, max_levels=6)
    area = multiscale_area_density(signal, max_levels=6)
    np.testing.assert_allclose(np.abs(details), area, atol=1e-14)


def test_component_forest_candidate_family_is_reproducible_and_aligned():
    x = np.linspace(0.0, 1.0, 161)
    signal = np.sin(16.0 * np.pi * x)
    signal[120:127] += np.hanning(7)
    first = hcrd_component_iforest_scores(signal, train_size=80, max_levels=5)
    second = hcrd_component_iforest_scores(signal, train_size=80, max_levels=5)
    assert len(first) == 11
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].shape == signal.shape
        assert np.all(np.isfinite(first[name]))
        np.testing.assert_allclose(first[name], second[name], atol=0.0, rtol=0.0)


def test_component_forest_validates_training_prefix():
    with pytest.raises(ValueError, match="train_size"):
        hcrd_component_iforest_scores(np.arange(10.0), train_size=3)

