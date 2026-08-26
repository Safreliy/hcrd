"""Finite-dictionary inference for complete HCRD lobe shapes."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LobeScanResult:
    """Result of an affine-residualized one-sided lobe-dictionary scan."""

    scores: NDArray[np.float64]
    selected_index: int
    statistic: float
    threshold: float
    rejected: bool
    templates: NDArray[np.float64]


def residualized_lobe_dictionary(
    templates: ArrayLike, *, x: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Remove affine components and normalize row-wise lobe templates.

    Orientations are preserved.  Include both signs as separate rows when the
    direction of a lobe is unknown.
    """

    values = np.asarray(templates, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("templates must have shape (M, n) with n >= 3")
    grid = np.arange(values.shape[1], dtype=float) if x is None else np.asarray(x, dtype=float)
    if grid.shape != (values.shape[1],) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be strictly increasing and match template length")
    design = np.column_stack([np.ones(grid.size), grid])
    coefficients = np.linalg.lstsq(design, values.T, rcond=None)[0]
    residuals = values - (design @ coefficients).T
    norms = np.linalg.norm(residuals, axis=1)
    if np.any(norms <= np.finfo(float).eps * max(1.0, np.linalg.norm(values))):
        raise ValueError("every template must contain a non-affine component")
    return residuals / norms[:, None]


def scan_detection_threshold(template_count: int, alpha: float) -> float:
    """Union-bound level-alpha threshold from the scan theorem."""

    if template_count < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("template_count >= 1 and 0 < alpha < 1 are required")
    return sqrt(2.0 * log(template_count / alpha))


def scan_power_sufficient_norm(
    template_count: int, alpha: float, beta: float
) -> float:
    """Sufficient standardized signal norm for miss probability at most beta."""

    if not 0.0 < beta < 1.0:
        raise ValueError("0 < beta < 1 is required")
    return scan_detection_threshold(template_count, alpha) + sqrt(2.0 * log(1.0 / beta))


def scan_localization_sufficient_norm(
    template_count: int, delta: float, maximum_coherence: float
) -> float:
    """Sufficient standardized norm for correct scan-argmax localization."""

    if template_count < 2 or not 0.0 < delta < 1.0:
        raise ValueError("template_count >= 2 and 0 < delta < 1 are required")
    if maximum_coherence >= 1.0:
        raise ValueError("maximum_coherence must be below one")
    return 2.0 * sqrt(2.0 * log(2.0 * template_count / delta)) / (
        1.0 - maximum_coherence
    )


def orthogonal_detection_lower_norm(
    template_count: int, alpha: float, beta: float
) -> float:
    """Largest norm covered by the chi-squared minimax impossibility bound."""

    if template_count < 1 or not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("valid template_count, alpha, and beta are required")
    if alpha + beta >= 1.0:
        raise ValueError("the lower bound requires alpha + beta < 1")
    return sqrt(log(1.0 + 4.0 * template_count * (1.0 - alpha - beta) ** 2))


def fano_localization_error_lower(template_count: int, standardized_norm: float) -> float:
    """Fano lower bound on mean index error for an orthonormal dictionary."""

    if template_count < 2 or standardized_norm < 0.0:
        raise ValueError("template_count >= 2 and nonnegative norm are required")
    return max(0.0, 1.0 - (standardized_norm**2 + log(2.0)) / log(template_count))


def scan_lobe_dictionary(
    observation: ArrayLike,
    templates: ArrayLike,
    *,
    noise_sigma: float,
    alpha: float = 0.05,
    x: ArrayLike | None = None,
) -> LobeScanResult:
    """Scan fixed or independently selected complete lobe shapes.

    This function does not select HCRD knots from ``observation``.  Passing
    same-replicate, data-selected templates would violate the theorem's
    fixed/independent-guide assumption.
    """

    values = np.asarray(observation, dtype=float)
    dictionary = residualized_lobe_dictionary(templates, x=x)
    if values.shape != (dictionary.shape[1],):
        raise ValueError("observation and templates must share their sample length")
    if not np.isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError("noise_sigma must be finite and positive")
    scores = dictionary @ values / noise_sigma
    selected = int(np.argmax(scores))
    threshold = scan_detection_threshold(dictionary.shape[0], alpha)
    statistic = float(scores[selected])
    return LobeScanResult(
        scores=scores,
        selected_index=selected,
        statistic=statistic,
        threshold=threshold,
        rejected=statistic > threshold,
        templates=dictionary,
    )
