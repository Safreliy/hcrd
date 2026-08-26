import numpy as np

from hcrd import (
    decompose_sparse,
    level_energies,
    multiscale_area_density,
    multiscale_energy_feature_names,
    multiscale_energy_features,
    sparse_structure_energies,
)


def test_exact_triangle_has_expected_geometric_and_quadratic_energy():
    x = np.asarray([0.0, 1.0, 2.0])
    signal = np.asarray([0.0, 1.0, 0.0])
    structures = sparse_structure_energies(
        decompose_sparse(signal, x, atol=0.0, rtol=0.0)
    )
    item = structures[0][0]
    assert np.isclose(item.polygon_area, 1.0)
    assert np.isclose(item.signed_polygon_area, 1.0)
    assert np.isclose(item.quadratic_energy, 2.0 / 3.0)
    assert np.isclose(item.triangle_area, 1.0)
    assert np.isclose(item.shape_factor, 0.5)
    assert np.isclose(item.quadratic_shape_factor, 0.75)


def test_affine_trend_is_invisible_and_vertical_scaling_is_predictable():
    x = np.linspace(-2.0, 3.0, 65)
    signal = np.sin(2.0 * x) + 0.2 * np.cos(5.0 * x)
    original = level_energies(decompose_sparse(signal, x, atol=0.0, rtol=0.0))
    scale = -2.5
    transformed = level_energies(
        decompose_sparse(scale * signal + 1.7 * x - 0.4, x, atol=0.0, rtol=0.0)
    )
    assert len(original) == len(transformed)
    for first, second in zip(original, transformed, strict=True):
        assert np.isclose(second.polygon_area, abs(scale) * first.polygon_area)
        assert np.isclose(
            second.quadratic_energy, scale**2 * first.quadratic_energy
        )
        assert np.isclose(second.peak_amplitude, abs(scale) * first.peak_amplitude)


def test_sparse_polygon_areas_match_dense_structure_areas():
    rng = np.random.default_rng(91)
    x = np.cumsum(rng.uniform(0.1, 1.5, size=129))
    sparse = decompose_sparse(rng.normal(size=x.size), x)
    dense = sparse.materialize()
    sparse_levels = sparse_structure_energies(sparse)
    for dense_level, sparse_level in zip(dense.levels, sparse_levels, strict=True):
        for dense_structure, sparse_structure in zip(
            dense_level.structures, sparse_level, strict=True
        ):
            assert np.isclose(
                dense_structure.signed_area,
                sparse_structure.signed_polygon_area,
                atol=1e-11,
            )


def test_fixed_length_energy_features_are_finite():
    x = np.linspace(0.0, 1.0, 257)
    signal = np.sin(8.0 * np.pi * x) + 0.3 * np.sin(22.0 * np.pi * x)
    features = multiscale_energy_features(signal, x, max_levels=5)
    names = multiscale_energy_feature_names(5)
    assert features.shape == (len(names),)
    assert np.all(np.isfinite(features))
    assert len(set(names)) == len(names)


def test_area_density_integrates_to_exact_level_polygon_mass():
    rng = np.random.default_rng(812)
    x = np.cumsum(rng.uniform(0.2, 1.2, size=193))
    signal = rng.normal(size=x.size)
    hierarchy = decompose_sparse(signal, x, max_levels=6)
    density = multiscale_area_density(signal, x, max_levels=6)
    summaries = level_energies(hierarchy)

    assert density.shape == (hierarchy.depth, signal.size)
    assert np.all(density >= 0.0)
    for row, summary in zip(density, summaries, strict=True):
        assert np.isclose(
            np.trapezoid(row, x), summary.polygon_area, atol=1e-10
        )


def test_area_density_is_zero_for_an_affine_signal():
    x = np.linspace(-3.0, 4.0, 101)
    density = multiscale_area_density(2.5 * x - 0.7, x)
    np.testing.assert_allclose(density, 0.0, atol=1e-12)
