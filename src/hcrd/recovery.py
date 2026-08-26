"""Finite-sample chord-lobe recovery helpers.

The generators in this module implement the parabolic subclasses used to audit
the exact- and approximate-join recovery theorems in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
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
    amplitudes: NDArray[np.float64]
    curvature_magnitude: float
    join_curvature_bound: float


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


def approximate_join_tolerance(
    join_curvature_bound: float, curvature_noise_radius: float
) -> float:
    """Return the absolute tolerance for approximate sampled joins.

    If population join curvatures are bounded by ``eta`` and all curvature
    errors are bounded by ``tau``, the join-aware HCRD tolerance is
    ``eta + tau``.
    """

    if join_curvature_bound < 0 or curvature_noise_radius < 0:
        raise ValueError("join and noise curvature bounds must be nonnegative")
    return float(join_curvature_bound + curvature_noise_radius)


def amplitudes_for_recovery_ratios(
    *,
    lobes: int,
    samples_per_lobe: int,
    curvature_ratio: float,
    join_ratio: float,
    curvature_threshold: float,
    spacing: float = 1.0,
) -> NDArray[np.float64]:
    """Construct alternating amplitudes with requested ``gamma/tau`` and ``eta/tau``.

    For parabolic lobes of width ``m``, the minimum active curvature is
    ``gamma = 8 A_min / (m^2 h)``.  Alternating amplitudes separated by
    ``Delta A`` give join magnitude
    ``eta = 4 (m-1) Delta A / (m^2 h)``.
    """

    if lobes < 2 or samples_per_lobe < 2:
        raise ValueError("lobes >= 2 and samples_per_lobe >= 2 are required")
    if curvature_ratio <= 0 or join_ratio < 0 or curvature_threshold < 0:
        raise ValueError("curvature ratio must be positive; bounds nonnegative")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    minimum = amplitude_for_curvature_ratio(
        curvature_ratio,
        curvature_threshold,
        samples_per_lobe,
        spacing=spacing,
    )
    difference = (
        join_ratio
        * curvature_threshold
        * samples_per_lobe**2
        * spacing
        / (4.0 * (samples_per_lobe - 1))
    )
    return minimum + difference * (np.arange(lobes, dtype=float) % 2.0)


def alternating_parabolic_chord_lobes(
    *,
    seed: int,
    lobes: int,
    samples_per_lobe: int,
    amplitude: float | None = None,
    amplitudes: ArrayLike | None = None,
    noise_sigma: float,
    spacing: float = 1.0,
) -> AlternatingChordLobeSignal:
    """Generate alternating parabolic lobes about an affine baseline.

    Supply exactly one of a common ``amplitude`` or a length-``lobes`` vector
    ``amplitudes``.  Unequal adjacent values create approximate, rather than
    exactly zero-curvature, sampled joins.
    """

    if lobes < 2 or samples_per_lobe < 2:
        raise ValueError("lobes >= 2 and samples_per_lobe >= 2 are required")
    if (amplitude is None) == (amplitudes is None):
        raise ValueError("supply exactly one of amplitude or amplitudes")
    amplitude_values = (
        np.full(lobes, float(amplitude), dtype=float)
        if amplitudes is None
        else np.asarray(amplitudes, dtype=float)
    )
    if amplitude_values.shape != (lobes,) or np.any(amplitude_values <= 0):
        raise ValueError("amplitudes must contain one positive value per lobe")
    if noise_sigma < 0 or spacing <= 0:
        raise ValueError("spacing must be positive and noise nonnegative")

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
        detail[left:right] += (
            first_sign * (-1.0) ** lobe * amplitude_values[lobe] * shape
        )

    noise = rng.normal(0.0, noise_sigma, size=n)
    return AlternatingChordLobeSignal(
        x=x,
        observed=baseline + detail + noise,
        baseline=baseline,
        detail=detail,
        knots=knots,
        noise=noise,
        amplitude=float(np.min(amplitude_values)),
        amplitudes=amplitude_values,
        curvature_magnitude=float(
            8.0 * np.min(amplitude_values) / (samples_per_lobe**2 * spacing)
        ),
        join_curvature_bound=float(
            4.0
            * (samples_per_lobe - 1)
            * np.max(np.abs(np.diff(amplitude_values)))
            / (samples_per_lobe**2 * spacing)
        ),
    )
