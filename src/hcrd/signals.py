"""Synthetic signal classes with known latent components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticSignal:
    x: NDArray[np.float64]
    observed: NDArray[np.float64]
    baseline: NDArray[np.float64]
    detail: NDArray[np.float64]
    knots: NDArray[np.int64]
    noise: NDArray[np.float64]


def alternating_chord_lobes(
    *,
    seed: int,
    intervals: int = 8,
    samples_per_interval: int = 32,
    noise_sigma: float = 0.0,
    piecewise_baseline: bool = True,
    amplitude_variation: bool = True,
) -> SyntheticSignal:
    """Generate alternating parabolic lobes around a known chord baseline."""

    if intervals < 2 or samples_per_interval < 4:
        raise ValueError("intervals >= 2 and samples_per_interval >= 4 are required")
    rng = np.random.default_rng(seed)
    n = intervals * samples_per_interval + 1
    x = np.linspace(0.0, float(intervals), n)
    knots = np.arange(0, n, samples_per_interval, dtype=np.int64)

    if piecewise_baseline:
        increments = rng.normal(0.0, 0.45, size=intervals)
        knot_values = np.concatenate([[rng.normal(0.0, 0.3)], np.cumsum(increments)])
    else:
        intercept, slope = rng.normal(0.0, 0.5, size=2)
        knot_values = intercept + slope * x[knots]
    baseline = np.interp(x, x[knots], knot_values)

    detail = np.zeros(n, dtype=float)
    amplitudes = (
        rng.uniform(0.7, 1.8, size=intervals)
        if amplitude_variation
        else np.full(intervals, rng.uniform(0.9, 1.5))
    )
    first_sign = 1 if rng.random() >= 0.5 else -1
    for interval in range(intervals):
        left, right = knots[interval], knots[interval + 1]
        q = np.linspace(0.0, 1.0, right - left + 1)
        sign = first_sign * (-1 if interval % 2 else 1)
        detail[left : right + 1] += sign * amplitudes[interval] * 4.0 * q * (1.0 - q)

    noise = rng.normal(0.0, noise_sigma, size=n)
    observed = baseline + detail + noise
    return SyntheticSignal(
        x=x,
        observed=observed,
        baseline=baseline,
        detail=detail,
        knots=knots,
        noise=noise,
    )


def chirp_with_affine_trend(
    *, seed: int, n: int = 1025, noise_sigma: float = 0.0
) -> SyntheticSignal:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    phase = 2.0 * np.pi * (3.0 * x + 12.0 * x**2)
    detail = (0.8 + 0.3 * np.sin(2.0 * np.pi * x)) * np.sin(phase)
    intercept, slope = rng.normal(0.0, 0.5, size=2)
    baseline = intercept + slope * x
    noise = rng.normal(0.0, noise_sigma, size=n)
    return SyntheticSignal(
        x=x,
        observed=baseline + detail + noise,
        baseline=baseline,
        detail=detail,
        knots=np.array([0, n - 1], dtype=np.int64),
        noise=noise,
    )
