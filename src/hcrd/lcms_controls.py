"""Matched-capacity non-HCRD controls for LC--MS waveform studies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_derivative_control(
    raw_features: ArrayLike,
) -> NDArray[np.float32]:
    """Expand 64 resampled samples into a fixed 948-variable control bank.

    The input's last axis must contain the 75 raw EIC variables used by the
    LC--MS studies: 64 normalized waveform samples followed by 11 scalars.  The
    output adds twelve full Gaussian scale-space channels, eight summaries per
    channel, and nine global scale/spectral summaries.  Its width exactly
    matches the per-file HCRD-8 bank without using HCRD knots or residuals.
    """

    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(raw_features, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 75:
        raise ValueError("raw_features must have 75 variables on its last axis")
    waveform = values[..., :64]
    channels: list[NDArray[np.float64]] = []
    summaries: list[NDArray[np.float64]] = []
    residual_ratios: list[NDArray[np.float64]] = []
    derivative_peaks: list[NDArray[np.float64]] = []
    signal_energy = np.mean(waveform**2, axis=-1)
    denominator = np.maximum(signal_energy, np.finfo(float).eps)

    for scale in (1.0, 2.0, 4.0, 8.0):
        smooth = gaussian_filter1d(waveform, sigma=scale, axis=-1, mode="nearest")
        derivative = gaussian_filter1d(
            waveform, sigma=scale, order=1, axis=-1, mode="nearest"
        )
        curvature = gaussian_filter1d(
            waveform, sigma=scale, order=2, axis=-1, mode="nearest"
        )
        residual_ratios.append(np.mean((waveform - smooth) ** 2, axis=-1) / denominator)
        derivative_peaks.append(np.max(np.abs(derivative), axis=-1))
        for channel in (smooth, derivative, curvature):
            channels.append(channel)
            summaries.append(
                np.stack(
                    [
                        np.mean(channel, axis=-1),
                        np.std(channel, axis=-1),
                        np.min(channel, axis=-1),
                        np.max(channel, axis=-1),
                        np.mean(np.abs(channel), axis=-1),
                        np.sqrt(np.mean(channel**2, axis=-1)),
                        np.mean(np.abs(np.diff(channel, axis=-1)), axis=-1),
                        np.max(np.abs(channel), axis=-1),
                    ],
                    axis=-1,
                )
            )

    spectral_power = np.abs(np.fft.rfft(waveform, axis=-1)) ** 2
    spectral_total = np.sum(spectral_power, axis=-1, keepdims=True)
    probabilities = spectral_power / np.maximum(spectral_total, np.finfo(float).eps)
    spectral_entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, np.finfo(float).eps)), axis=-1
    ) / np.log(spectral_power.shape[-1])
    global_summaries = np.stack(
        [*residual_ratios, *derivative_peaks, spectral_entropy], axis=-1
    )
    output = np.concatenate(
        [
            values,
            np.concatenate(channels, axis=-1),
            np.concatenate(summaries, axis=-1),
            global_summaries,
        ],
        axis=-1,
    )
    if output.shape[-1] != 948:
        raise RuntimeError("matched Gaussian control has an invalid width")
    return output.astype(np.float32)
