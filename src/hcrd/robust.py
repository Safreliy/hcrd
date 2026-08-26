"""Noise-aware HCRD variants and stability certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import BoundaryRule, Decomposition, decompose, discrete_curvature
from .baselines import gaussian_smooth

_NORMAL_MAD = 0.6744897501960817


@dataclass(frozen=True)
class RobustResult:
    decomposition: Decomposition
    estimated_noise_sigma: float
    curvature_threshold: float
    z_score: float


@dataclass(frozen=True)
class GuidedDecomposition:
    """Exact observation = guide residual + HCRD(guide) decomposition."""

    original: np.ndarray
    guide: np.ndarray
    guide_residual: np.ndarray
    decomposition: Decomposition
    smoothing_sigma: float

    def reconstruct(self) -> np.ndarray:
        return self.guide_residual + self.decomposition.reconstruct()


@dataclass(frozen=True)
class AdaptiveGuidedResult:
    guided: GuidedDecomposition
    estimated_noise_sigma: float
    robust_signal_range: float
    noise_ratio: float


def _uniform_locations(signal: ArrayLike, x: ArrayLike | None) -> np.ndarray:
    y = np.asarray(signal, dtype=float)
    locations = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    gaps = np.diff(locations)
    if gaps.size and not np.allclose(gaps, gaps[0], rtol=1e-10, atol=1e-12):
        raise ValueError("automatic noise calibration currently requires a uniform grid")
    return locations


def estimate_noise_sigma(signal: ArrayLike, x: ArrayLike | None = None) -> float:
    """Robustly estimate iid additive noise from second differences.

    For unit-spaced iid noise with standard deviation sigma, the standard
    deviation of the second difference is ``sqrt(6) * sigma``.  Smooth signal
    curvature may bias this estimate upward; this is documented and tested as a
    conservative calibration rather than an oracle estimator.
    """

    y = np.asarray(signal, dtype=float)
    locations = _uniform_locations(y, x)
    if y.size < 3:
        return 0.0
    curvature = discrete_curvature(y, locations)
    centered = curvature - np.median(curvature)
    mad = float(np.median(np.abs(centered)))
    spacing = float(locations[1] - locations[0])
    return mad * spacing / (_NORMAL_MAD * np.sqrt(6.0))


def robust_decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    z_score: float = 3.5,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
) -> RobustResult:
    """Apply HCRD after declaring small curvature statistically insignificant."""

    if z_score < 0:
        raise ValueError("z_score must be nonnegative")
    y = np.asarray(signal, dtype=float)
    locations = _uniform_locations(y, x)
    sigma = estimate_noise_sigma(y, locations)
    spacing = float(locations[1] - locations[0]) if y.size > 1 else 1.0
    curvature_threshold = z_score * np.sqrt(6.0) * sigma / spacing
    result = decompose(
        y,
        locations,
        atol=curvature_threshold,
        rtol=64 * np.finfo(float).eps,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
    )
    return RobustResult(
        decomposition=result,
        estimated_noise_sigma=sigma,
        curvature_threshold=curvature_threshold,
        z_score=z_score,
    )


def familywise_curvature_threshold(
    n: int,
    sigma: float,
    *,
    delta: float = 0.05,
    spacing: float = 1.0,
) -> float:
    """A union-bound threshold for iid Gaussian observation noise.

    With probability at least ``1-delta``, all ``n-2`` divided-slope curvature
    errors are at most the returned value in magnitude.
    """

    if n < 3:
        return 0.0
    if sigma < 0 or spacing <= 0:
        raise ValueError("sigma must be nonnegative and spacing must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    return float(
        sigma
        / spacing
        * np.sqrt(12.0 * np.log(2.0 * (n - 2) / delta))
    )


def certified_robust_decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    sigma: float,
    delta: float = 0.05,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
) -> RobustResult:
    """HCRD with a family-wise Gaussian curvature threshold.

    Unlike :func:`robust_decompose`, this variant assumes that the observation
    noise standard deviation is known or externally estimated and provides a
    finite-sample simultaneous error bound for the first-level curvature signs.
    """

    y = np.asarray(signal, dtype=float)
    locations = _uniform_locations(y, x)
    spacing = float(locations[1] - locations[0]) if y.size > 1 else 1.0
    threshold = familywise_curvature_threshold(
        y.size, sigma, delta=delta, spacing=spacing
    )
    result = decompose(
        y,
        locations,
        atol=threshold,
        rtol=64 * np.finfo(float).eps,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
    )
    return RobustResult(
        decomposition=result,
        estimated_noise_sigma=float(sigma),
        curvature_threshold=threshold,
        z_score=float(threshold * spacing / (np.sqrt(6.0) * sigma)) if sigma else 0.0,
    )


def gaussian_guided_decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    smoothing_sigma: float = 2.0,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
) -> GuidedDecomposition:
    """Use a Gaussian guide for knot discovery without losing reconstruction.

    The guide residual is explicitly retained rather than silently discarded.
    This variant trades exact affine equivariance at the boundaries for much
    greater curvature-sign stability under high-frequency observation noise.
    """

    y = np.asarray(signal, dtype=float)
    locations = _uniform_locations(y, x)
    guide = gaussian_smooth(y, smoothing_sigma)
    result = decompose(
        guide,
        locations,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
        minimum_knot_spacing=max(1, int(np.ceil(2.0 * smoothing_sigma))),
    )
    return GuidedDecomposition(
        original=y.copy(),
        guide=guide,
        guide_residual=y - guide,
        decomposition=result,
        smoothing_sigma=float(smoothing_sigma),
    )


def adaptive_gaussian_guided_decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    sensitivity: float = 300.0,
    minimum_sigma: float = 1.0,
    maximum_sigma: float = 12.0,
    boundary_rule: BoundaryRule = "minimum_curvature",
) -> AdaptiveGuidedResult:
    """Pilot-calibrated guide scale based on a robust noise-to-range ratio.

    This is an empirical modification, not covered by the Gaussian threshold
    theorem.  ``sensitivity`` is expressed in samples and must be revalidated
    when sampling density or signal class changes.
    """

    y = np.asarray(signal, dtype=float)
    locations = _uniform_locations(y, x)
    if sensitivity < 0 or minimum_sigma <= 0 or maximum_sigma < minimum_sigma:
        raise ValueError("invalid adaptive guide parameters")
    noise_sigma = estimate_noise_sigma(y, locations)
    robust_range = float(np.quantile(y, 0.95) - np.quantile(y, 0.05))
    noise_ratio = noise_sigma / max(robust_range, np.finfo(float).eps)
    smoothing_sigma = float(
        np.clip(1.0 + sensitivity * noise_ratio, minimum_sigma, maximum_sigma)
    )
    guided = gaussian_guided_decompose(
        y,
        locations,
        smoothing_sigma=smoothing_sigma,
        boundary_rule=boundary_rule,
    )
    return AdaptiveGuidedResult(
        guided=guided,
        estimated_noise_sigma=noise_sigma,
        robust_signal_range=robust_range,
        noise_ratio=noise_ratio,
    )


def stability_certificate(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    boundary_rule: BoundaryRule = "minimum_curvature",
) -> float:
    """Certified uniform perturbation radius for fixed first-level decisions.

    On a uniform grid, ``gamma*h/4`` preserves nonzero curvature signs.  The
    centred rule also compares curvature magnitudes across sign transitions;
    ``eta*h/8`` preserves those comparisons, where ``eta`` is the smallest
    opposite-neighbour absolute-magnitude gap.  The returned radius is the
    smaller relevant margin.  Equality is not certified.
    """

    y = np.asarray(signal, dtype=float)
    locations = _uniform_locations(y, x)
    if y.size < 3:
        return float("inf")
    curvature = discrete_curvature(y, locations)
    if np.any(curvature == 0):
        return 0.0
    gamma = float(np.min(np.abs(curvature)))
    spacing = float(locations[1] - locations[0])
    sign_radius = gamma * spacing / 4.0
    if boundary_rule == "legacy":
        return sign_radius
    if boundary_rule != "minimum_curvature":
        raise ValueError(f"unknown boundary rule: {boundary_rule}")
    opposite = np.signbit(curvature[:-1]) != np.signbit(curvature[1:])
    if not np.any(opposite):
        return sign_radius
    comparison_margin = float(
        np.min(np.abs(np.abs(curvature[:-1][opposite]) - np.abs(curvature[1:][opposite])))
    )
    comparison_radius = comparison_margin * spacing / 8.0
    return min(sign_radius, comparison_radius)
