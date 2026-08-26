"""Temporal analysis of the multilevel HCRD polygon-mass series."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from .anomaly import aggregate_area_density
from .energy import multiscale_area_density


def empirical_rank(score: ArrayLike) -> NDArray[np.float64]:
    """Map a finite score to mid-ranks in ``(0, 1]`` without labels."""

    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("score must be a finite one-dimensional array")
    return rankdata(values, method="average") / values.size


def spectral_residual_score(
    signal: ArrayLike, *, amplitude_window: int = 100
) -> NDArray[np.float64]:
    """Return the spectral-residual saliency of one temporal series."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite one-dimensional array")
    if amplitude_window < 1:
        raise ValueError("amplitude_window must be positive")
    scale = float(np.ptp(values))
    if scale == 0.0:
        return np.zeros_like(values)
    normalised = (values - np.min(values)) / scale
    spectrum = np.fft.fft(normalised)
    log_amplitude = np.log(np.maximum(np.abs(spectrum), np.finfo(float).tiny))
    width = min(amplitude_window, values.size)
    left = (width - 1) // 2
    right = width - left - 1
    padded = np.pad(log_amplitude, (left, right), mode="edge")
    smooth = np.convolve(padded, np.full(width, 1.0 / width), mode="valid")
    residual = log_amplitude - smooth
    return np.abs(np.fft.ifft(np.exp(residual + 1j * np.angle(spectrum))))


def hcrd_temporal_candidate_scores(
    signal: ArrayLike,
    *,
    max_levels: int = 8,
    amplitude_window: int = 100,
) -> dict[str, NDArray[np.float64]]:
    """Return the fixed A2 candidate family for one univariate series."""

    values = np.asarray(signal, dtype=float)
    densities = multiscale_area_density(values, max_levels=max_levels)
    direct = aggregate_area_density(densities, aggregation="max")
    raw = np.abs(values - np.median(values))
    area_sr = spectral_residual_score(direct, amplitude_window=amplitude_window)
    level_sr = np.vstack(
        [
            spectral_residual_score(row, amplitude_window=amplitude_window)
            for row in densities
        ]
    )
    level_ranks = np.vstack([empirical_rank(row) for row in level_sr])
    spectrum_max = np.max(level_ranks, axis=0)
    spectrum_mean = np.mean(level_ranks, axis=0)

    rank_direct = empirical_rank(direct)
    rank_raw = empirical_rank(raw)
    rank_area_sr = empirical_rank(area_sr)
    rank_spectrum_max = empirical_rank(spectrum_max)
    output = {
        "a2_direct": direct,
        "a2_raw_abs": raw,
        "a2_fuse_h025_raw": 0.25 * rank_direct + 0.75 * rank_raw,
        "a2_fuse_h050_raw": 0.50 * rank_direct + 0.50 * rank_raw,
        "a2_fuse_h075_raw": 0.75 * rank_direct + 0.25 * rank_raw,
        "a2_area_sr": area_sr,
        "a2_fuse_direct_area_sr": 0.5 * rank_direct + 0.5 * rank_area_sr,
        "a2_spectrum_sr_max": spectrum_max,
        "a2_spectrum_sr_mean": spectrum_mean,
        "a2_fuse_direct_spectrum_sr": (
            0.5 * rank_direct + 0.5 * rank_spectrum_max
        ),
        "a2_fuse_raw_spectrum_sr": 0.5 * rank_raw + 0.5 * rank_spectrum_max,
        "a2_fuse_three": (
            rank_direct + rank_raw + rank_spectrum_max
        ) / 3.0,
    }
    return output

