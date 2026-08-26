"""Compact-impulse scores and event metrics for neural-spike development."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, find_peaks, sosfiltfilt

from .anomaly import aggregate_area_density
from .core import decompose


def spike_bandpass(
    signal: ArrayLike,
    *,
    sampling_rate: float,
    low_hz: float = 300.0,
    high_hz: float = 3000.0,
    order: int = 3,
) -> NDArray[np.float64]:
    """Zero-phase bandpass used identically by all spike detectors."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 32 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite vector of length at least 32")
    nyquist = sampling_rate / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError("bandpass frequencies must lie strictly below Nyquist")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate, output="sos")
    return np.asarray(sosfiltfilt(sos, values), dtype=float)


def intracellular_spike_times(
    intracellular_signal: ArrayLike,
    *,
    sampling_rate: float,
    threshold_sigma: float = 10.0,
    refractory_seconds: float = 0.001,
) -> NDArray[np.int64]:
    """Derive physical event times from an independent intracellular trace.

    This follows the SpikeForest HC-1 description: threshold a differentiated
    intracellular trace at ten robust noise standard deviations.  Absolute
    derivative and a refractory rule merge the two edges of one action
    potential into one event.
    """

    values = np.asarray(intracellular_signal, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("intracellular_signal must be a finite vector")
    if sampling_rate <= 0.0 or threshold_sigma <= 0.0:
        raise ValueError("sampling_rate and threshold_sigma must be positive")
    derivative = np.diff(values, prepend=values[0])
    centred = derivative - np.median(derivative)
    sigma = float(np.median(np.abs(centred)) / 0.6744897501960817)
    if sigma == 0.0:
        raise ValueError("intracellular derivative has zero robust noise scale")
    distance = max(1, int(round(refractory_seconds * sampling_rate)))
    peaks, _ = find_peaks(
        np.abs(centred),
        height=threshold_sigma * sigma,
        distance=distance,
    )
    return peaks.astype(np.int64)


def hcrd_concentration_anomaly_score(
    signal: ArrayLike,
    *,
    max_levels: int | None = 8,
) -> NDArray[np.float64]:
    """Full-hierarchy score emphasizing compact signed chord structures.

    For each HCRD structure, its absolute detail density is multiplied by
    ``amplitude / discrete_polygon_mass``.  This is the structure's shape
    concentration, discovered as an exploratory WSD effect modifier.  Rows are
    then robustly normalized and fused by their pointwise maximum.  Polygon
    mass remains an auxiliary weight; all hierarchy levels and full detail
    shapes remain present in the score.
    """

    hierarchy = decompose(signal, max_levels=max_levels)
    rows = np.zeros((hierarchy.depth, hierarchy.original.size), dtype=float)
    tiny = np.finfo(float).tiny
    for row, level in enumerate(hierarchy.levels):
        density = np.abs(level.detail)
        weighted = np.zeros_like(density)
        for structure in level.structures:
            segment = density[structure.left : structure.right + 1]
            discrete_mass = float(np.sum(segment))
            if discrete_mass <= tiny or structure.amplitude <= 0.0:
                continue
            concentration = structure.amplitude / discrete_mass
            weighted[structure.left : structure.right + 1] = (
                segment * concentration
            )
        rows[row] = weighted
    return aggregate_area_density(rows, aggregation="max")


def robust_multichannel_max(scores: ArrayLike) -> NDArray[np.float64]:
    """Fuse channel rows after fixed robust positive-surprise scaling."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("scores must have shape channels by time")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    nonnegative = values - np.min(values, axis=1, keepdims=True)
    return aggregate_area_density(nonnegative, aggregation="max")


def event_average_precision(
    score: ArrayLike,
    truth: ArrayLike,
    *,
    tolerance_samples: int,
    refractory_samples: int,
) -> float:
    """Event AP from score maxima with one-to-one temporal matching."""

    values = np.asarray(score, dtype=float)
    events = np.asarray(truth, dtype=np.int64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("score must be a finite vector")
    if events.ndim != 1 or events.size == 0 or np.any(np.diff(events) <= 0):
        raise ValueError("truth must be a nonempty strictly increasing vector")
    if events[0] < 0 or events[-1] >= values.size:
        raise ValueError("truth events must lie inside the score")
    if tolerance_samples < 0 or refractory_samples < 1:
        raise ValueError("invalid tolerance or refractory distance")
    candidates, _ = find_peaks(values, distance=refractory_samples)
    order = np.lexsort((candidates, -values[candidates]))
    matched = np.zeros(events.size, dtype=bool)
    true_positives = 0
    precision_sum = 0.0
    for rank, candidate_index in enumerate(order, start=1):
        candidate = int(candidates[candidate_index])
        left = int(np.searchsorted(events, candidate - tolerance_samples, side="left"))
        right = int(np.searchsorted(events, candidate + tolerance_samples, side="right"))
        available = np.flatnonzero(~matched[left:right]) + left
        if available.size == 0:
            continue
        nearest = available[np.argmin(np.abs(events[available] - candidate))]
        matched[nearest] = True
        true_positives += 1
        precision_sum += true_positives / rank
    return float(precision_sum / events.size)


def classical_compact_impulse_scores(
    filtered_signal: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Training-free amplitude and nonlinear-energy comparator scores."""

    values = np.asarray(filtered_signal, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("filtered_signal must be a finite vector")
    amplitude = np.abs(values)
    neo = np.zeros_like(values)
    neo[1:-1] = np.maximum(
        values[1:-1] ** 2 - values[:-2] * values[2:],
        0.0,
    )
    return {"amplitude": amplitude, "neo": neo}
