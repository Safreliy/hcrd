"""Tests for the finite HCRD lobe-dictionary theorem implementation."""

from __future__ import annotations

import numpy as np

from hcrd.lobe_scan import (
    fano_localization_error_lower,
    orthogonal_detection_lower_norm,
    residualized_lobe_dictionary,
    scan_detection_threshold,
    scan_lobe_dictionary,
    scan_localization_sufficient_norm,
    scan_power_sufficient_norm,
)


def _templates() -> np.ndarray:
    first = np.zeros(17)
    first[2:9] = [0.0, 1.0, 3.0, 5.0, 3.0, 1.0, 0.0]
    second = np.zeros(17)
    second[9:16] = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0]
    return np.stack([first, second])


def test_residualized_dictionary_annihilates_affine_vectors() -> None:
    x = np.linspace(-1.0, 2.0, 17)
    dictionary = residualized_lobe_dictionary(_templates(), x=x)
    np.testing.assert_allclose(dictionary @ np.ones(17), 0.0, atol=1e-12)
    np.testing.assert_allclose(dictionary @ x, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(dictionary, axis=1), 1.0)


def test_scan_is_affine_invariant_and_localizes_visible_lobe() -> None:
    x = np.arange(17, dtype=float)
    dictionary = residualized_lobe_dictionary(_templates(), x=x)
    observation = 7.0 - 0.3 * x + 10.0 * dictionary[1]
    result = scan_lobe_dictionary(
        observation, _templates(), noise_sigma=1.0, alpha=0.05, x=x
    )
    assert result.rejected
    assert result.selected_index == 1
    np.testing.assert_allclose(result.scores, dictionary @ (10.0 * dictionary[1]))


def test_reported_bounds_have_theorem_order_and_valid_domains() -> None:
    small = scan_detection_threshold(10, 0.05)
    large = scan_detection_threshold(1000, 0.05)
    assert large > small > 0.0
    assert scan_power_sufficient_norm(10, 0.05, 0.2) > small
    assert scan_localization_sufficient_norm(10, 0.1, 0.5) > 0.0
    assert 0.0 < orthogonal_detection_lower_norm(10, 0.05, 0.2) < large
    assert 0.0 < fano_localization_error_lower(1000, 1.0) < 1.0
