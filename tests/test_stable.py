import numpy as np

from hcrd.core import discrete_curvature
from hcrd.stable import (
    certified_curvature_signs,
    certified_proximal_guided_decompose,
    certified_quadratic_guided_decompose,
    curvature_perturbation_bounds,
    proximal_curvature_split,
    quadratic_curvature_split,
)


def test_proximal_split_and_guided_hierarchy_reconstruct_exactly():
    rng = np.random.default_rng(211)
    signal = rng.normal(size=97)
    split = proximal_curvature_split(signal, regularization=1.3)
    guided = certified_proximal_guided_decompose(
        signal,
        regularization=1.3,
        input_perturbation_radius=0.05,
    )
    np.testing.assert_allclose(split.reconstruct(), signal, atol=2e-13)
    np.testing.assert_allclose(guided.reconstruct(), signal, atol=2e-13)


def test_proximal_guide_and_residual_are_numerically_nonexpansive():
    rng = np.random.default_rng(223)
    for _ in range(8):
        first = rng.normal(size=65)
        second = first + 0.2 * rng.normal(size=65)
        first_split = proximal_curvature_split(first, regularization=0.8)
        second_split = proximal_curvature_split(second, regularization=0.8)
        input_distance = np.linalg.norm(first - second)
        assert np.linalg.norm(first_split.guide - second_split.guide) <= (
            input_distance + 2e-6
        )
        assert np.linalg.norm(first_split.residual - second_split.residual) <= (
            input_distance + 2e-6
        )


def test_proximal_guide_is_affine_equivariant_on_irregular_grid():
    rng = np.random.default_rng(227)
    x = np.cumsum(rng.uniform(0.7, 1.4, size=71))
    signal = rng.normal(size=x.size)
    affine = -0.7 + 0.08 * x
    first = proximal_curvature_split(signal, x, regularization=1.1)
    second = proximal_curvature_split(signal + affine, x, regularization=1.1)
    np.testing.assert_allclose(second.guide, first.guide + affine, atol=2e-6)
    np.testing.assert_allclose(second.residual, first.residual, atol=2e-6)


def test_irregular_grid_curvature_certificate_survives_bounded_error():
    rng = np.random.default_rng(229)
    x = np.cumsum(rng.uniform(0.5, 1.8, size=83))
    signal = rng.normal(size=x.size)
    radius = 0.03
    bounds = curvature_perturbation_bounds(x, radius)
    signs = certified_curvature_signs(signal, radius, x)
    assert np.all(signs[np.abs(discrete_curvature(signal, x)) <= bounds] == 0)
    for _ in range(30):
        error = rng.uniform(-radius, radius, size=signal.size)
        perturbed_signs = np.sign(discrete_curvature(signal + error, x)).astype(np.int8)
        certified = signs != 0
        np.testing.assert_array_equal(perturbed_signs[certified], signs[certified])


def test_input_l2_radius_certifies_proximal_guide_curvature_signs():
    rng = np.random.default_rng(233)
    signal = rng.normal(size=61)
    radius = 0.04
    first = proximal_curvature_split(signal, regularization=0.4)
    signs = certified_curvature_signs(first.guide, radius, first.x)
    for _ in range(12):
        direction = rng.normal(size=signal.size)
        error = direction / np.linalg.norm(direction) * (0.95 * radius)
        second = proximal_curvature_split(signal + error, regularization=0.4)
        second_signs = np.sign(discrete_curvature(second.guide, second.x)).astype(np.int8)
        certified = signs != 0
        np.testing.assert_array_equal(second_signs[certified], signs[certified])


def test_quadratic_split_is_exact_nonexpansive_and_affine_equivariant():
    rng = np.random.default_rng(239)
    x = np.cumsum(rng.uniform(0.8, 1.3, size=79))
    first = rng.normal(size=x.size)
    second = first + 0.3 * rng.normal(size=x.size)
    affine = 0.4 - 0.03 * x
    first_split = quadratic_curvature_split(first, x, regularization=0.7)
    second_split = quadratic_curvature_split(second, x, regularization=0.7)
    translated = quadratic_curvature_split(first + affine, x, regularization=0.7)
    np.testing.assert_allclose(first_split.reconstruct(), first, atol=2e-13)
    input_distance = np.linalg.norm(first - second)
    assert np.linalg.norm(first_split.guide - second_split.guide) <= input_distance
    assert np.linalg.norm(first_split.residual - second_split.residual) <= input_distance
    np.testing.assert_allclose(translated.guide, first_split.guide + affine, atol=2e-12)
    np.testing.assert_allclose(translated.residual, first_split.residual, atol=2e-12)


def test_quadratic_guided_hierarchy_reconstructs_exactly():
    rng = np.random.default_rng(241)
    signal = rng.normal(size=101)
    result = certified_quadratic_guided_decompose(
        signal,
        regularization=0.1,
        input_perturbation_radius=0.02,
    )
    np.testing.assert_allclose(result.reconstruct(), signal, atol=2e-13)
