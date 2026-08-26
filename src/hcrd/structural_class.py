"""Label-assisted morphology descriptors for auditing HCRD's target class.

These descriptors are analysis tools, not detector inputs. They use anomaly
labels to ask whether performance differences follow the proposed chord-lobe
mechanism without defining a class from HCRD scores themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class EventChordMorphology:
    """Observable geometry of one labelled anomaly event."""

    start: int
    end: int
    duration_fraction: float
    sign_coherence: float
    shape_concentration: float
    peak_to_background_mad: float
    curvature_contrast: float


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.diff(np.pad(labels.astype(int), (1, 1))))
    return [
        (int(left), int(right - 1))
        for left, right in zip(changes[::2], changes[1::2], strict=True)
    ]


def event_chord_morphologies(
    signal: ArrayLike, labels: ArrayLike
) -> tuple[EventChordMorphology, ...]:
    """Describe labelled events relative to their immediate endpoint chord."""

    values = np.asarray(signal, dtype=float)
    binary = np.asarray(labels, dtype=int)
    if values.ndim != 1 or binary.shape != values.shape or values.size < 3:
        raise ValueError("signal and labels must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.all((binary == 0) | (binary == 1)):
        raise ValueError("signal must be finite and labels binary")
    events: list[EventChordMorphology] = []
    global_mad = float(np.median(np.abs(values - np.median(values))))
    global_floor = max(global_mad, float(np.ptp(values)) * 1e-12, np.finfo(float).tiny)
    for start, end in _runs(binary):
        left = max(start - 1, 0)
        right = min(end + 1, values.size - 1)
        if right <= left:
            continue
        indices = np.arange(start, end + 1)
        chord = np.interp(
            indices,
            np.asarray([left, right], dtype=float),
            values[[left, right]],
        )
        residual = values[indices] - chord
        absolute = np.abs(residual)
        mass = float(np.sum(absolute))
        sign_coherence = float(abs(np.sum(residual)) / mass) if mass > 0.0 else 0.0
        concentration = float(np.max(absolute) / mass) if mass > 0.0 else 0.0

        width = max(20, 5 * (end - start + 1))
        context_left = max(0, start - width)
        context_right = min(values.size, end + width + 1)
        context_indices = np.arange(context_left, context_right)
        normal_context = values[context_indices[binary[context_indices] == 0]]
        if normal_context.size:
            local_median = float(np.median(normal_context))
            local_mad = float(np.median(np.abs(normal_context - local_median)))
        else:
            local_mad = 0.0
        scale = max(1.4826 * local_mad, global_floor)
        peak_to_background = float(np.max(absolute, initial=0.0) / scale)

        event_extended = values[left : right + 1]
        event_curvature = np.abs(np.diff(event_extended, n=2))
        context_values = values[context_left:context_right]
        context_curvature = np.abs(np.diff(context_values, n=2))
        if context_curvature.size:
            background_curvature = float(np.median(context_curvature))
        else:
            background_curvature = 0.0
        curvature_floor = max(
            background_curvature,
            float(np.max(np.abs(np.diff(values, n=2)), initial=0.0)) * 1e-12,
            np.finfo(float).tiny,
        )
        curvature_contrast = float(
            np.max(event_curvature, initial=0.0) / curvature_floor
        )
        events.append(
            EventChordMorphology(
                start=start,
                end=end,
                duration_fraction=(end - start + 1) / values.size,
                sign_coherence=sign_coherence,
                shape_concentration=concentration,
                peak_to_background_mad=peak_to_background,
                curvature_contrast=curvature_contrast,
            )
        )
    return tuple(events)


def aggregate_event_chord_morphology(
    signal: ArrayLike, labels: ArrayLike
) -> dict[str, float]:
    """Return series-level medians of the predefined event descriptors."""

    events = event_chord_morphologies(signal, labels)
    if not events:
        raise ValueError("labels must contain at least one anomaly event")
    return {
        "event_count": float(len(events)),
        "median_duration_fraction": float(
            np.median([item.duration_fraction for item in events])
        ),
        "median_sign_coherence": float(
            np.median([item.sign_coherence for item in events])
        ),
        "median_shape_concentration": float(
            np.median([item.shape_concentration for item in events])
        ),
        "median_peak_to_background_mad": float(
            np.median([item.peak_to_background_mad for item in events])
        ),
        "median_curvature_contrast": float(
            np.median([item.curvature_contrast for item in events])
        ),
    }
