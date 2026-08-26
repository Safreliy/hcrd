"""Tests for replicate-split HCRD structure inference."""

from __future__ import annotations

import numpy as np

from hcrd import (
    chord_area_coefficients,
    infer_hcrd_matched_structures,
    infer_hcrd_structures,
)


def test_chord_area_functional_annihilates_affine_signals() -> None:
    x = np.asarray([0.0, 0.2, 0.8, 1.7, 3.0])
    coefficients = chord_area_coefficients(x)
    np.testing.assert_allclose(coefficients @ np.ones_like(x), 0.0, atol=1e-14)
    np.testing.assert_allclose(coefficients @ x, 0.0, atol=1e-14)


def test_chord_area_matches_trapezoidal_residual() -> None:
    x = np.asarray([0.0, 0.5, 1.0, 2.0])
    y = np.asarray([1.0, 3.0, -1.0, 2.0])
    chord = np.interp(x, x[[0, -1]], y[[0, -1]])
    expected = np.trapezoid(y - chord, x)
    observed = chord_area_coefficients(x) @ y
    np.testing.assert_allclose(observed, expected)


def test_independent_replicate_detects_large_visible_structure() -> None:
    x = np.arange(33, dtype=float)
    signal = np.zeros(33)
    signal[8:17] = np.asarray([0, 2, 4, 6, 8, 6, 4, 2, 0])
    structures = infer_hcrd_structures(
        signal,
        signal,
        x=x,
        noise_sigma=0.1,
        familywise_alpha=0.05,
        max_levels=8,
    )
    assert structures
    assert any(item.significant for item in structures)
    assert max(abs(item.z_score) for item in structures) > 10.0


def test_matched_structure_annihilates_affine_scoring_mean() -> None:
    x = np.arange(33, dtype=float)
    guide = np.zeros(33)
    guide[8:17] = np.asarray([0, 2, 4, 6, 8, 6, 4, 2, 0])
    affine = 5.0 - 0.2 * x
    structures = infer_hcrd_matched_structures(
        guide,
        affine,
        x=x,
        noise_sigma=1.0,
        max_levels=8,
    )
    assert structures
    np.testing.assert_allclose(
        [item.z_score for item in structures], 0.0, atol=1e-12
    )


def test_matched_structure_uses_full_guide_shape() -> None:
    signal = np.zeros(65)
    signal[16:33] = np.concatenate(
        [np.linspace(0.0, 10.0, 9), np.linspace(8.75, 0.0, 8)]
    )
    structures = infer_hcrd_matched_structures(
        signal,
        signal,
        noise_sigma=0.2,
        max_levels=8,
    )
    assert any(item.significant for item in structures)
    assert max(abs(item.z_score) for item in structures) > 20.0
