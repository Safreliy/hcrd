"""Reference-event and alignment utilities for ECG-referenced PPG benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ppg import match_event_pairs


@dataclass(frozen=True)
class ECGReference:
    """Consensus ECG beats and the sample-level high-quality mask."""

    beats: NDArray[np.int64]
    quality_mask: NDArray[np.bool_]
    xqrs_beats: NDArray[np.int64]
    neurokit_beats: NDArray[np.int64]
    agreement_fraction: float


def interpolate_nonfinite(signal: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < 2:
        raise ValueError("signal must contain at least two finite samples")
    return np.interp(np.arange(values.size), finite, values[finite])


def flatline_mask(
    signal: ArrayLike,
    sampling_frequency: float,
    minimum_duration_seconds: float = 0.2,
) -> NDArray[np.bool_]:
    """Mark non-finite samples and constant runs longer than the threshold."""

    values = np.asarray(signal, dtype=float)
    invalid = ~np.isfinite(values)
    finite_values = interpolate_nonfinite(values)
    changes = np.flatnonzero(np.r_[True, np.diff(finite_values) != 0.0, True])
    threshold = int(round(minimum_duration_seconds * sampling_frequency))
    for left, right in zip(changes[:-1], changes[1:], strict=True):
        if right - left > threshold:
            invalid[left:right] = True
    return invalid


def consensus_ecg_reference(
    ecg: ArrayLike,
    sampling_frequency: float,
    *,
    tolerance_seconds: float = 0.15,
) -> ECGReference:
    """Detect high-quality ECG beats by WFDB-XQRS/NeuroKit agreement."""

    import neurokit2 as nk
    from wfdb.processing import xqrs_detect

    original = np.asarray(ecg, dtype=float)
    values = interpolate_nonfinite(original)
    values -= np.median(values)
    scale = 1.4826 * np.median(np.abs(values - np.median(values)))
    if scale > np.finfo(float).eps:
        values /= scale
    xqrs = np.asarray(
        xqrs_detect(values, fs=sampling_frequency, verbose=False), dtype=np.int64
    )
    cleaned = nk.ecg_clean(values, sampling_rate=sampling_frequency, method="neurokit")
    _, information = nk.ecg_peaks(
        cleaned, sampling_rate=sampling_frequency, method="neurokit", correct_artifacts=False
    )
    neurokit = np.asarray(information["ECG_R_Peaks"], dtype=np.int64)
    tolerance = int(round(tolerance_seconds * sampling_frequency))
    pairs = match_event_pairs(neurokit, xqrs, tolerance)
    agreed_xqrs = np.zeros(xqrs.size, dtype=bool)
    if pairs.size:
        agreed_xqrs[pairs[:, 1]] = True
    beat_quality = np.zeros(xqrs.size, dtype=bool)
    for index in range(xqrs.size):
        left = max(0, index - 1)
        right = min(xqrs.size, index + 2)
        beat_quality[index] = bool(np.all(agreed_xqrs[left:right]))
    quality = np.zeros(values.size, dtype=bool)
    for index, high_quality in enumerate(beat_quality):
        if not high_quality:
            continue
        left = 0 if index == 0 else int(xqrs[index - 1] + 1)
        right = values.size if index + 1 == xqrs.size else int(xqrs[index + 1])
        quality[left:right] = True
    quality &= ~flatline_mask(original, sampling_frequency)
    beats = xqrs[agreed_xqrs]
    beats = beats[quality[beats]]
    denominator = max(1, max(xqrs.size, neurokit.size))
    return ECGReference(
        beats=beats,
        quality_mask=quality,
        xqrs_beats=xqrs,
        neurokit_beats=neurokit,
        agreement_fraction=float(pairs.shape[0] / denominator),
    )


def _nearest_distances(
    reference: NDArray[np.int64], prediction: NDArray[np.int64]
) -> NDArray[np.int64]:
    if prediction.size == 0:
        return np.full(reference.size, np.iinfo(np.int64).max, dtype=np.int64)
    insertions = np.searchsorted(prediction, reference)
    left_indices = np.clip(insertions - 1, 0, prediction.size - 1)
    right_indices = np.clip(insertions, 0, prediction.size - 1)
    return np.minimum(
        np.abs(reference - prediction[left_indices]),
        np.abs(reference - prediction[right_indices]),
    )


def align_ecg_to_ppg(
    ecg_beats: ArrayLike,
    ppg_beats: ArrayLike,
    sampling_frequency: float,
    *,
    tolerance_seconds: float = 0.15,
    maximum_lag_seconds: float = 10.0,
    lag_increment_seconds: float = 0.02,
    beats_per_block: int = 300,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Replicate PPG-beats' detector-specific piecewise pulse-transit alignment."""

    reference = np.sort(np.asarray(ecg_beats, dtype=np.int64))
    prediction = np.sort(np.asarray(ppg_beats, dtype=np.int64))
    if reference.size == 0 or prediction.size == 0:
        return reference.copy(), np.zeros(reference.size, dtype=np.int64)
    tolerance = int(round(tolerance_seconds * sampling_frequency))
    lag_seconds = np.arange(
        -maximum_lag_seconds,
        maximum_lag_seconds + 0.5 * lag_increment_seconds,
        lag_increment_seconds,
    )
    lags = np.unique(np.round(lag_seconds * sampling_frequency).astype(np.int64))
    lags = np.asarray(sorted(lags, key=lambda item: (abs(int(item)), int(item))))
    aligned = reference.copy()
    selected_lags = np.zeros(reference.size, dtype=np.int64)
    if reference.size <= beats_per_block:
        block_starts = np.asarray([0], dtype=np.int64)
    else:
        # Match the MATLAB colon expression `1:B:length-B`: the final
        # nominal block starts no later than length-B and is expanded to the
        # end instead of creating a short remainder block.
        block_starts = np.arange(
            0, reference.size - beats_per_block + 1, beats_per_block
        )
    for block_number, start_value in enumerate(block_starts):
        start = int(start_value)
        stop = (
            reference.size
            if block_number + 1 == block_starts.size
            else start + beats_per_block
        )
        block = reference[start:stop]
        shifted = block[None, :] + lags[:, None]
        insertions = np.searchsorted(prediction, shifted)
        left_indices = np.clip(insertions - 1, 0, prediction.size - 1)
        right_indices = np.clip(insertions, 0, prediction.size - 1)
        distances = np.minimum(
            np.abs(shifted - prediction[left_indices]),
            np.abs(shifted - prediction[right_indices]),
        )
        counts = np.count_nonzero(distances < tolerance, axis=1)
        # `lags` is ordered by absolute magnitude, then signed value, so argmax
        # implements the benchmark's minimum-absolute-lag tie rule.
        best_lag = int(lags[int(np.argmax(counts))])
        aligned[start:stop] = block + best_lag
        selected_lags[start:stop] = best_lag
    valid = aligned > 0
    aligned = aligned[valid]
    selected_lags = selected_lags[valid]
    # PPG-beats excludes beats that cease to be monotone when adjacent
    # variable-lag blocks choose different pulse-transit shifts.
    monotone = np.ones(aligned.size, dtype=bool)
    current_max = -1
    for index, value in enumerate(aligned):
        if int(value) > current_max:
            current_max = int(value)
        else:
            monotone[index] = False
    return aligned[monotone], selected_lags[monotone]
