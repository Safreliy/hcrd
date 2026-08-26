import numpy as np

from hcrd.features import (
    COMPONENT_FEATURE_NAMES,
    gaussian_pyramid_components,
    guided_hcrd_components,
    hcrd_components,
    normalise_window,
    representation_features,
)


def test_component_representations_reconstruct_normalised_signal():
    rng = np.random.default_rng(41)
    signal = rng.normal(size=256)
    expected = normalise_window(signal)
    for factory in (gaussian_pyramid_components, hcrd_components, guided_hcrd_components):
        components = factory(signal, n_components=5)
        assert len(components) == 5
        assert np.allclose(np.sum(components, axis=0), expected, atol=1e-10)


def test_feature_vector_is_fixed_length_and_finite():
    x = np.linspace(0.0, 6.0 * np.pi, 256)
    signal = np.sin(x) + 0.2 * np.cos(4.0 * x)
    for representation in ("raw", "gaussian_pyramid", "hcrd", "hcrd_guided"):
        features = representation_features(signal, representation, n_components=5)
        assert features.shape == (5 * len(COMPONENT_FEATURE_NAMES),)
        assert np.all(np.isfinite(features))
