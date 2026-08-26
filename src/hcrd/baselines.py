"""Dependency-light comparison baselines used by the preregistered benchmark."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def affine_trend(signal: ArrayLike, x: ArrayLike | None = None) -> NDArray[np.float64]:
    y = np.asarray(signal, dtype=float)
    locations = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    design = np.column_stack([np.ones_like(locations), locations])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coefficients


def moving_average(signal: ArrayLike, window: int) -> NDArray[np.float64]:
    y = np.asarray(signal, dtype=float)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(y, radius, mode="reflect")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def gaussian_smooth(signal: ArrayLike, sigma: float) -> NDArray[np.float64]:
    y = np.asarray(signal, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(np.ceil(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(y, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def fourier_lowpass(signal: ArrayLike, retained_bins: int) -> NDArray[np.float64]:
    y = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(y)
    if retained_bins < 1 or retained_bins > spectrum.size:
        raise ValueError("retained_bins outside valid range")
    filtered = spectrum.copy()
    filtered[retained_bins:] = 0
    return np.fft.irfft(filtered, n=y.size)


def rdp_indices(x: ArrayLike, y: ArrayLike, epsilon: float) -> NDArray[np.int64]:
    """Ramer-Douglas-Peucker polyline simplification."""

    locations = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    if locations.shape != values.shape or values.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of equal shape")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    keep = {0, values.size - 1}
    stack = [(0, values.size - 1)]
    while stack:
        left, right = stack.pop()
        if right <= left + 1:
            continue
        fraction = (locations[left + 1 : right] - locations[left]) / (
            locations[right] - locations[left]
        )
        chord = values[left] + fraction * (values[right] - values[left])
        deviations = np.abs(values[left + 1 : right] - chord)
        local = int(np.argmax(deviations))
        if deviations[local] > epsilon:
            index = left + 1 + local
            keep.add(index)
            stack.append((left, index))
            stack.append((index, right))
    return np.asarray(sorted(keep), dtype=np.int64)


def interpolate_knots(
    x: ArrayLike, y: ArrayLike, knots: ArrayLike
) -> NDArray[np.float64]:
    locations = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    indices = np.asarray(knots, dtype=np.int64)
    return np.interp(locations, locations[indices], values[indices])

