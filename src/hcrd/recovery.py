"""Finite-sample chord-lobe recovery helpers.

The generator in this module is the explicit parabolic subclass used to audit
the recovery theorem in the manuscript.  It is intentionally narrow: adjacent
lobes have equal width and amplitude, so their alternating one-sign curvature
blocks meet at a single sampled zero-curvature point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .robust import familywise_curvature_threshold


@dataclass(frozen=True)
class AlternatingChordLobeSignal:
    """A noisy member of the theorem-matched parabolic chord-lobe class."""

    x: NDArray[np.float64]
    observed: NDArray[np.float64]
    baseline: NDArray[np.float64]
    detail: NDArray[np.float64]
    knots: NDArray[np.int64]
    noise: NDArray[np.float64]
    amplitude: float
    curvature_magnitude: float


def finite_sample_recovery_thresholds(
    n: int,
    sigma: float,
    *,
    delta: float = 0.05,
    spacing: float = 1.0,
) -> tuple[float, float]:
    """Return the theorem's curvature threshold and sample-noise radius.

    Half of ``delta`` is allocated to simultaneous curvature control and half
    to the maximum sample error.  The resulting intersection event has
    probability at least ``1-delta`` under iid Gaussian noise.
    """

    if n < 5:
        raise ValueError("the recovery class requires at least five samples")
    if sigma < 0 or spacing <= 0:
        raise ValueError("sigma must be nonnegative and spacing must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    curvature_threshold = familywise_curvature_threshold(
        n, sigma, delta=delta / 2.0, spacing=spacing
    )
    sample_radius = float(
        sigma * np.sqrt(2.0 * np.log(4.0 * n / delta))
    )
    return curvature_threshold, sample_radius


def amplitude_for_curvature_ratio(
    curvature_ratio: float,
    curvature_threshold: float,
    samples_per_lobe: int,
    *,
    spacing: float = 1.0,
) -> float:
    """Convert ``gamma / tau`` to the parabolic lobe amplitude.

    A unit-grid lobe ``4 A u(1-u)`` sampled on ``m`` intervals has divided-
    slope curvature magnitude ``gamma = 8 A / (m^2 h)``.
    """

    if curvature_ratio <= 0 or curvature_threshold < 0:
        raise ValueError("curvature_ratio must be positive and threshold nonnegative")
    if samples_per_lobe < 2 or spacing <= 0:
        raise ValueError("samples_per_lobe >= 2 and positive spacing are required")
    return float(
        curvature_ratio
        * curvature_threshold
        * samples_per_lobe**2
        * spacing
        / 8.0
    )


def alternating_parabolic_chord_lobes(
    *,
    seed: int,
    lobes: int,
    samples_per_lobe: int,
    amplitude: float,
    noise_sigma: float,
    spacing: float = 1.0,
) -> AlternatingChordLobeSignal:
    """Generate alternating equal parabolic lobes about an affine baseline."""

    if lobes < 2 or samples_per_lobe < 2:
        raise ValueError("lobes >= 2 and samples_per_lobe >= 2 are required")
    if amplitude <= 0 or noise_sigma < 0 or spacing <= 0:
        raise ValueError("amplitude and spacing must be positive; noise nonnegative")

    rng = np.random.default_rng(seed)
    n = lobes * samples_per_lobe + 1
    x = spacing * np.arange(n, dtype=float)
    knots = np.arange(0, n, samples_per_lobe, dtype=np.int64)
    intercept, slope = rng.normal(0.0, 0.4, size=2)
    baseline = intercept + slope * x
    detail = np.zeros(n, dtype=float)
    local = np.arange(samples_per_lobe + 1, dtype=float) / samples_per_lobe
    shape = 4.0 * local * (1.0 - local)
    first_sign = 1.0 if rng.random() >= 0.5 else -1.0
    for lobe in range(lobes):
        left = lobe * samples_per_lobe
        right = left + samples_per_lobe + 1
        detail[left:right] += first_sign * (-1.0) ** lobe * amplitude * shape

    noise = rng.normal(0.0, noise_sigma, size=n)
    return AlternatingChordLobeSignal(
        x=x,
        observed=baseline + detail + noise,
        baseline=baseline,
        detail=detail,
        knots=knots,
        noise=noise,
        amplitude=float(amplitude),
        curvature_magnitude=float(
            8.0 * amplitude / (samples_per_lobe**2 * spacing)
        ),
    )
