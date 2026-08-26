import math

import numpy as np

from hcrd.core import (
    decompose,
    decompose_sparse,
    find_convexity_knots,
    total_variation,
)


def test_exact_reconstruction_for_both_boundary_rules():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=257)
    for rule in ("legacy", "minimum_curvature"):
        result = decompose(signal, boundary_rule=rule)
        np.testing.assert_allclose(result.reconstruct(), signal, rtol=0, atol=2e-13)


def test_legacy_known_alternating_example():
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    knots = find_convexity_knots(
        signal, boundary_rule="legacy", atol=0.0, rtol=0.0
    )
    np.testing.assert_array_equal(knots, [0, 2, 4])


def test_minimum_curvature_removes_sine_phase_bias():
    x = np.linspace(0.0, 4.0 * np.pi, 65)
    signal = np.sin(x)
    knots = find_convexity_knots(signal, x, boundary_rule="minimum_curvature")
    np.testing.assert_allclose(x[knots] / np.pi, [0, 1, 2, 3, 4], atol=1e-12)


def test_affine_equivariance_of_details_and_knots():
    rng = np.random.default_rng(11)
    x = np.linspace(-2.0, 3.0, 129)
    signal = rng.normal(size=x.size)
    transformed = 2.7 * signal - 1.2 + 0.8 * x
    original = decompose(signal, x, atol=0.0, rtol=0.0)
    changed = decompose(transformed, x, atol=0.0, rtol=0.0)
    assert original.depth == changed.depth
    for first, second in zip(original.levels, changed.levels, strict=True):
        np.testing.assert_array_equal(first.knots, second.knots)
        np.testing.assert_allclose(second.detail, 2.7 * first.detail, atol=2e-12)


def test_range_and_total_variation_do_not_increase():
    rng = np.random.default_rng(17)
    signal = rng.normal(size=200)
    result = decompose(signal)
    for level in result.levels:
        assert np.min(level.baseline) >= np.min(level.input_baseline) - 1e-12
        assert np.max(level.baseline) <= np.max(level.input_baseline) + 1e-12
        assert total_variation(level.baseline) <= total_variation(level.input_baseline) + 1e-12


def test_knot_sets_are_nested_and_depth_is_logarithmic():
    rng = np.random.default_rng(23)
    signal = rng.normal(size=513)
    result = decompose(signal)
    previous = np.arange(signal.size)
    for level in result.levels:
        assert set(level.knots).issubset(set(previous))
        previous = level.knots
    assert result.depth <= max(1, math.ceil(math.log2(signal.size - 1)))


def test_nonuniform_affine_signal_has_zero_detail():
    x = np.array([0.0, 0.1, 0.4, 1.3, 2.0, 5.0])
    signal = 3.0 - 1.7 * x
    result = decompose(signal, x)
    assert result.depth == 1
    np.testing.assert_allclose(result.levels[0].detail, 0.0, atol=1e-14)


def test_structure_sign_agrees_with_convex_chord_geometry():
    convex = np.arange(9, dtype=float) ** 2
    result = decompose(convex, atol=0.0, rtol=0.0)
    assert result.levels[0].structures[0].sign == -1
    assert np.all(result.levels[0].detail <= 1e-12)


def test_sparse_hierarchy_materializes_exact_dense_result_on_irregular_grid():
    rng = np.random.default_rng(31)
    x = np.cumsum(rng.uniform(0.2, 1.7, size=257))
    signal = rng.normal(size=x.size)
    sparse = decompose_sparse(signal, x, atol=0.0, rtol=0.0)
    dense = sparse.materialize()
    np.testing.assert_allclose(dense.reconstruct(), signal, rtol=0.0, atol=1e-14)
    assert dense.depth == sparse.depth
    for sparse_level, dense_level in zip(sparse.levels, dense.levels, strict=True):
        np.testing.assert_array_equal(sparse_level.knots, dense_level.knots)
        np.testing.assert_array_equal(
            dense_level.baseline[sparse_level.knots], signal[sparse_level.knots]
        )


def test_sparse_centred_storage_obeys_linear_halving_bound():
    rng = np.random.default_rng(37)
    for length in (2, 3, 17, 129, 513):
        sparse = decompose_sparse(rng.normal(size=length))
        # Sum_j (N_j + 1), with N_j intervals, is below 2N_0 plus one
        # endpoint entry per implemented level.
        assert sparse.stored_knot_count <= 2 * (length - 1) + sparse.depth
