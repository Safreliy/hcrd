"""Multilevel HCRD representations for pulse-event detection in raw PPG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import decompose_sparse
from .energy import sparse_structure_energies


@dataclass(frozen=True)
class PPGRecording:
    """One raw PPG recording with expert event and artifact annotations."""

    subject: int
    activity: str
    trial: int
    time: NDArray[np.float64]
    signal: NDArray[np.float64]
    peaks: NDArray[np.int64]
    artifacts: NDArray[np.int64]

    @property
    def key(self) -> str:
        return f"s{self.subject}_{self.activity}{self.trial}"

    @property
    def sampling_frequency(self) -> float:
        return float(1.0 / np.median(np.diff(self.time)))


@dataclass(frozen=True)
class HCRDCandidateBank:
    """Candidate positions and aligned structural and morphology features."""

    positions: NDArray[np.int64]
    geometry: NDArray[np.float64]
    morphology: NDArray[np.float64]
    geometry_names: tuple[str, ...]
    morphology_names: tuple[str, ...]
    conditioned_signal: NDArray[np.float64]


@dataclass(frozen=True)
class EventMatch:
    """One-to-one event matching result."""

    true_positive: int
    false_positive: int
    false_negative: int
    absolute_errors: NDArray[np.int64]

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 0.0


def _read_integer_csv(path: Path, delimiter: str | None = None) -> NDArray[np.int64]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        columns = 2 if delimiter else 1
        return np.empty((0, columns), dtype=np.int64) if columns == 2 else np.empty(0, dtype=np.int64)
    rows = [line.split(delimiter) if delimiter else [line] for line in text.splitlines()]
    values = np.asarray([[int(item) for item in row] for row in rows], dtype=np.int64)
    return values if delimiter else values[:, 0]


def load_ppgopt_recording(
    root: str | Path,
    subject: int,
    activity: str,
    trial: int,
    *,
    load_annotations: bool = True,
) -> PPGRecording:
    """Load one PPGopt recording without executing the supplied pickle files."""

    from scipy.io import loadmat

    base = Path(root)
    raw_path = (
        base
        / "PMC6971339_supplementary"
        / "mmc1"
        / "PPG_ACC_dataset"
        / f"S{subject}"
        / f"{activity}{trial}_ppg.mat"
    )
    annotation = base / "ppgopt_annot" / "ppgopt_annot" / "csv" / f"S{subject}"
    key = f"s{subject}_{activity}{trial}"
    matrix = np.asarray(loadmat(raw_path)["PPG"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"unexpected PPG matrix shape in {raw_path}: {matrix.shape}")
    if matrix.shape[1] > 2 and np.any(np.nan_to_num(matrix[:, 2:]) != 0.0):
        raise ValueError(f"ambiguous additional nonzero PPG columns in {raw_path}")
    if load_annotations:
        peaks = _read_integer_csv(annotation / f"{key}_pks.csv")
        artifacts = _read_integer_csv(annotation / f"{key}_atf.csv", delimiter=";")
    else:
        peaks = np.empty(0, dtype=np.int64)
        artifacts = np.empty((0, 2), dtype=np.int64)
    return PPGRecording(
        subject=subject,
        activity=activity,
        trial=trial,
        time=matrix[:, 0],
        signal=matrix[:, 1],
        peaks=peaks,
        artifacts=artifacts,
    )


def iter_ppgopt_keys() -> tuple[tuple[int, str, int], ...]:
    """Return the 105 canonical subject/activity/trial keys."""

    return tuple(
        (subject, activity, trial)
        for subject in range(1, 8)
        for activity in ("rest", "squat", "step")
        for trial in range(1, 6)
    )


def artifact_mask(length: int, artifacts: ArrayLike) -> NDArray[np.bool_]:
    """Return an inclusive Boolean mask for annotated invalid intervals."""

    mask = np.zeros(int(length), dtype=bool)
    intervals = np.asarray(artifacts, dtype=np.int64).reshape(-1, 2)
    for left, right in intervals:
        start = max(0, int(left))
        stop = min(mask.size, int(right) + 1)
        if start < stop:
            mask[start:stop] = True
    return mask


def robust_bandpass(
    signal: ArrayLike,
    sampling_frequency: float,
    low_hz: float = 0.5,
    high_hz: float = 15.0,
) -> NDArray[np.float64]:
    """Zero-phase fourth-order Butterworth conditioning and robust scaling."""

    from scipy.signal import butter, sosfiltfilt

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 32:
        raise ValueError("signal must be a one-dimensional array of length >= 32")
    if not np.all(np.isfinite(values)):
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size < 2:
            raise ValueError("signal must contain at least two finite samples")
        values = np.interp(np.arange(values.size), finite, values[finite])
    if not 0.0 < low_hz < high_hz < 0.5 * sampling_frequency:
        raise ValueError("cutoffs must lie strictly between zero and Nyquist")
    sos = butter(
        4,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )
    filtered = sosfiltfilt(sos, values)
    filtered -= np.median(filtered)
    scale = 1.4826 * np.median(np.abs(filtered - np.median(filtered)))
    if scale <= np.finfo(float).eps:
        scale = float(np.std(filtered))
    if scale <= np.finfo(float).eps:
        return np.zeros_like(filtered)
    return np.asarray(filtered / scale, dtype=float)


def _nearest_peak(
    peaks: NDArray[np.int64],
    position: int,
    tolerance: int,
) -> int | None:
    insertion = int(np.searchsorted(peaks, position))
    choices = []
    if insertion < peaks.size:
        choices.append(int(peaks[insertion]))
    if insertion:
        choices.append(int(peaks[insertion - 1]))
    if not choices:
        return None
    nearest = min(choices, key=lambda item: (abs(item - position), item))
    return nearest if abs(nearest - position) <= tolerance else None


def _longest_run(levels: Iterable[int]) -> int:
    ordered = sorted(set(int(item) for item in levels))
    best = current = 0
    previous = None
    for level in ordered:
        current = current + 1 if previous is not None and level == previous + 1 else 1
        best = max(best, current)
        previous = level
    return best


def hcrd_candidate_bank(
    signal: ArrayLike,
    sampling_frequency: float,
    *,
    max_levels: int = 8,
    window_seconds: float = 30.0,
    halo_seconds: float = 2.0,
    minimum_duration_seconds: float = 0.05,
    maximum_duration_seconds: float = 2.0,
    snap_seconds: float = 0.05,
    cluster_seconds: float = 0.08,
) -> HCRDCandidateBank:
    """Build the full multilevel HCRD event representation from one PPG trace."""

    from scipy.signal import find_peaks, peak_prominences, peak_widths

    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    conditioned = robust_bandpass(signal, sampling_frequency)
    local_maxima = np.asarray(find_peaks(conditioned)[0], dtype=np.int64)
    window = max(16, int(round(window_seconds * sampling_frequency)))
    halo = max(0, int(round(halo_seconds * sampling_frequency)))
    snap = max(0, int(round(snap_seconds * sampling_frequency)))
    cluster = max(0, int(round(cluster_seconds * sampling_frequency)))
    observations: list[dict[str, float | int]] = []

    for core_start in range(0, conditioned.size, window):
        core_stop = min(conditioned.size, core_start + window)
        segment_start = max(0, core_start - halo)
        segment_stop = min(conditioned.size, core_stop + halo)
        segment = conditioned[segment_start:segment_stop]
        if segment.size < 4:
            continue
        locations = np.arange(segment.size, dtype=float) / sampling_frequency
        sparse = decompose_sparse(segment, locations, max_levels=max_levels)
        dense = sparse.materialize()
        energy_levels = sparse_structure_energies(sparse)
        for level_index, (level, energies) in enumerate(
            zip(dense.levels, energy_levels, strict=True)
        ):
            structures = level.structures
            for structure_index, (structure, energy) in enumerate(
                zip(structures, energies, strict=True)
            ):
                if structure.sign <= 0:
                    continue
                if not minimum_duration_seconds <= structure.duration <= maximum_duration_seconds:
                    continue
                anchor = segment_start + structure.peak_index
                mapped = _nearest_peak(local_maxima, anchor, snap)
                if mapped is None or not core_start <= mapped < core_stop:
                    continue
                left_negative = structures[structure_index - 1] if structure_index else None
                right_negative = (
                    structures[structure_index + 1]
                    if structure_index + 1 < len(structures)
                    else None
                )
                if left_negative is not None and left_negative.sign >= 0:
                    left_negative = None
                if right_negative is not None and right_negative.sign >= 0:
                    right_negative = None
                duration = max(float(structure.duration), np.finfo(float).eps)
                left = segment_start + structure.left
                right = segment_start + structure.right
                observations.append(
                    {
                        "position": int(mapped),
                        "level": int(level_index),
                        "amplitude": float(energy.amplitude),
                        "duration": float(energy.duration),
                        "area": float(energy.polygon_area),
                        "signed_area": float(energy.signed_polygon_area),
                        "quadratic": float(energy.quadratic_energy),
                        "triangle": float(energy.triangle_area),
                        "shape": float(energy.shape_factor),
                        "peak_offset": abs(mapped - anchor) / sampling_frequency / duration,
                        "midpoint_offset": abs(mapped - 0.5 * (left + right))
                        / sampling_frequency
                        / duration,
                        "left_distance": (mapped - left) / sampling_frequency,
                        "right_distance": (right - mapped) / sampling_frequency,
                        "left_negative_amplitude": (
                            float(left_negative.amplitude) if left_negative is not None else 0.0
                        ),
                        "right_negative_amplitude": (
                            float(right_negative.amplitude) if right_negative is not None else 0.0
                        ),
                    }
                )

    observations.sort(key=lambda item: int(item["position"]))
    groups: list[list[dict[str, float | int]]] = []
    for observation in observations:
        if not groups or int(observation["position"]) - int(groups[-1][-1]["position"]) > cluster:
            groups.append([observation])
        else:
            groups[-1].append(observation)

    level_fields = (
        "support",
        "log1p_amplitude",
        "duration_seconds",
        "log1p_polygon_area",
        "signed_polygon_area",
        "log1p_quadratic_energy",
        "log1p_triangle_area",
        "shape_factor",
        "peak_offset_fraction",
        "midpoint_offset_fraction",
        "left_distance_seconds",
        "right_distance_seconds",
        "log1p_left_negative_amplitude",
        "log1p_right_negative_amplitude",
        "negative_to_positive_ratio",
    )
    geometry_names = tuple(
        f"level_{level}_{field}" for level in range(max_levels) for field in level_fields
    ) + (
        "supporting_level_count",
        "longest_consecutive_level_run",
        "first_supporting_level",
        "last_supporting_level",
        "supporting_level_span",
        "position_dispersion_seconds",
        "left_boundary_dispersion_seconds",
        "right_boundary_dispersion_seconds",
        "duration_growth_ratio",
        "amplitude_decay_ratio",
        "area_decay_ratio",
        "energy_decay_ratio",
        "log1p_total_amplitude",
        "log1p_total_polygon_area",
        "log1p_total_quadratic_energy",
        "log1p_total_triangle_area",
        "mean_shape_factor",
        "max_amplitude_level",
        "max_area_level",
        "max_energy_level",
    )
    morphology_names = (
        "conditioned_height",
        "log1p_prominence",
        "width_seconds",
        "left_slope",
        "right_slope",
        "local_curvature",
        "local_range",
    )

    if not groups:
        return HCRDCandidateBank(
            positions=np.empty(0, dtype=np.int64),
            geometry=np.empty((0, len(geometry_names)), dtype=float),
            morphology=np.empty((0, len(morphology_names)), dtype=float),
            geometry_names=geometry_names,
            morphology_names=morphology_names,
            conditioned_signal=conditioned,
        )

    positions: list[int] = []
    geometry_rows: list[list[float]] = []
    for group in groups:
        exact_positions = sorted({int(item["position"]) for item in group})
        position = max(
            exact_positions,
            key=lambda candidate: (
                len({int(item["level"]) for item in group if int(item["position"]) == candidate}),
                sum(float(item["area"]) for item in group if int(item["position"]) == candidate),
                -candidate,
            ),
        )
        positions.append(position)
        row: list[float] = []
        selected_by_level: dict[int, dict[str, float | int]] = {}
        for item in group:
            level = int(item["level"])
            if level not in selected_by_level or float(item["area"]) > float(
                selected_by_level[level]["area"]
            ):
                selected_by_level[level] = item
        for level in range(max_levels):
            item = selected_by_level.get(level)
            if item is None:
                row.extend([0.0] * len(level_fields))
                continue
            amplitude = float(item["amplitude"])
            left_negative = float(item["left_negative_amplitude"])
            right_negative = float(item["right_negative_amplitude"])
            row.extend(
                [
                    1.0,
                    np.log1p(amplitude),
                    float(item["duration"]),
                    np.log1p(float(item["area"])),
                    float(item["signed_area"]),
                    np.log1p(float(item["quadratic"])),
                    np.log1p(float(item["triangle"])),
                    float(item["shape"]),
                    float(item["peak_offset"]),
                    float(item["midpoint_offset"]),
                    float(item["left_distance"]),
                    float(item["right_distance"]),
                    np.log1p(left_negative),
                    np.log1p(right_negative),
                    (left_negative + right_negative) / max(amplitude, 1e-12),
                ]
            )
        selected = list(selected_by_level.values())
        levels = [int(item["level"]) for item in selected]
        normalizer = max(1, max_levels - 1)
        amplitudes = np.asarray([float(item["amplitude"]) for item in selected])
        areas = np.asarray([float(item["area"]) for item in selected])
        energies = np.asarray([float(item["quadratic"]) for item in selected])
        triangles = np.asarray([float(item["triangle"]) for item in selected])
        durations = np.asarray([float(item["duration"]) for item in selected])
        shapes = np.asarray([float(item["shape"]) for item in selected])
        observed_positions = np.asarray([int(item["position"]) for item in selected])
        left_boundaries = np.asarray(
            [position - float(item["left_distance"]) * sampling_frequency for item in selected]
        )
        right_boundaries = np.asarray(
            [position + float(item["right_distance"]) * sampling_frequency for item in selected]
        )
        order = np.argsort(levels)
        first = int(order[0])
        last = int(order[-1])
        row.extend(
            [
                float(len(set(levels))),
                float(_longest_run(levels)),
                min(levels) / normalizer,
                max(levels) / normalizer,
                (max(levels) - min(levels)) / normalizer,
                float(np.std(observed_positions) / sampling_frequency),
                float(np.std(left_boundaries) / sampling_frequency),
                float(np.std(right_boundaries) / sampling_frequency),
                float(durations[last] / max(durations[first], 1e-12)),
                float(amplitudes[last] / max(amplitudes[first], 1e-12)),
                float(areas[last] / max(areas[first], 1e-12)),
                float(energies[last] / max(energies[first], 1e-12)),
                float(np.log1p(np.sum(amplitudes))),
                float(np.log1p(np.sum(areas))),
                float(np.log1p(np.sum(energies))),
                float(np.log1p(np.sum(triangles))),
                float(np.mean(shapes)),
                levels[int(np.argmax(amplitudes))] / normalizer,
                levels[int(np.argmax(areas))] / normalizer,
                levels[int(np.argmax(energies))] / normalizer,
            ]
        )
        geometry_rows.append(row)

    peak_positions = np.asarray(positions, dtype=np.int64)
    prominences = peak_prominences(conditioned, peak_positions)[0]
    widths = peak_widths(conditioned, peak_positions, rel_height=0.5)[0]
    slope_width = max(1, int(round(0.1 * sampling_frequency)))
    range_width = max(1, int(round(0.25 * sampling_frequency)))
    morphology_rows = []
    for position, prominence, width in zip(
        peak_positions, prominences, widths, strict=True
    ):
        left_slope_index = max(0, int(position) - slope_width)
        right_slope_index = min(conditioned.size - 1, int(position) + slope_width)
        range_left = max(0, int(position) - range_width)
        range_right = min(conditioned.size, int(position) + range_width + 1)
        curvature = (
            conditioned[position - 1] - 2.0 * conditioned[position] + conditioned[position + 1]
            if 0 < position < conditioned.size - 1
            else 0.0
        )
        morphology_rows.append(
            [
                float(conditioned[position]),
                float(np.log1p(max(0.0, prominence))),
                float(width / sampling_frequency),
                float(
                    (conditioned[position] - conditioned[left_slope_index])
                    / max((position - left_slope_index) / sampling_frequency, 1e-12)
                ),
                float(
                    (conditioned[right_slope_index] - conditioned[position])
                    / max((right_slope_index - position) / sampling_frequency, 1e-12)
                ),
                float(curvature),
                float(np.ptp(conditioned[range_left:range_right])),
            ]
        )
    geometry = np.asarray(geometry_rows, dtype=float)
    morphology = np.asarray(morphology_rows, dtype=float)
    if not np.all(np.isfinite(geometry)) or not np.all(np.isfinite(morphology)):
        raise RuntimeError("non-finite PPG candidate features")
    return HCRDCandidateBank(
        positions=peak_positions,
        geometry=geometry,
        morphology=morphology,
        geometry_names=geometry_names,
        morphology_names=morphology_names,
        conditioned_signal=conditioned,
    )


def suppress_events(
    positions: ArrayLike,
    scores: ArrayLike,
    minimum_distance_samples: int,
    *,
    threshold: float = -np.inf,
) -> NDArray[np.int64]:
    """Score-ordered non-maximum suppression with deterministic ties."""

    points = np.asarray(positions, dtype=np.int64)
    values = np.asarray(scores, dtype=float)
    if points.shape != values.shape or points.ndim != 1:
        raise ValueError("positions and scores must be aligned one-dimensional arrays")
    order = sorted(
        (index for index in range(points.size) if values[index] >= threshold),
        key=lambda index: (-values[index], int(points[index])),
    )
    kept: list[int] = []
    for index in order:
        point = int(points[index])
        if all(abs(point - other) >= minimum_distance_samples for other in kept):
            kept.append(point)
    return np.asarray(sorted(kept), dtype=np.int64)


def mask_events(
    positions: ArrayLike,
    artifacts: ArrayLike,
) -> NDArray[np.int64]:
    """Drop event positions lying in inclusive artifact intervals."""

    points = np.asarray(positions, dtype=np.int64)
    intervals = np.asarray(artifacts, dtype=np.int64).reshape(-1, 2)
    keep = np.ones(points.size, dtype=bool)
    for left, right in intervals:
        keep &= (points < left) | (points > right)
    return points[keep]


def match_events(
    reference: ArrayLike,
    detected: ArrayLike,
    tolerance_samples: int,
) -> EventMatch:
    """Maximum-cardinality, minimum-error one-to-one event matching.

    Dynamic programming makes the tie rule explicit and is cheap for the few
    hundred pulses present in an individual PPGopt recording.
    """

    truth = np.sort(np.asarray(reference, dtype=np.int64))
    prediction = np.sort(np.asarray(detected, dtype=np.int64))
    pairs = match_event_pairs(truth, prediction, tolerance_samples)
    errors = (
        np.abs(truth[pairs[:, 0]] - prediction[pairs[:, 1]])
        if pairs.size
        else np.empty(0, dtype=np.int64)
    )
    true_positive = int(pairs.shape[0])
    return EventMatch(
        true_positive=true_positive,
        false_positive=int(prediction.size - true_positive),
        false_negative=int(truth.size - true_positive),
        absolute_errors=np.asarray(errors, dtype=np.int64),
    )


def match_event_cardinality(
    reference: ArrayLike,
    detected: ArrayLike,
    tolerance_samples: int,
) -> int:
    """Linear-time maximum matching cardinality for sorted 1-D events."""

    truth = np.sort(np.asarray(reference, dtype=np.int64))
    prediction = np.sort(np.asarray(detected, dtype=np.int64))
    i = j = matched = 0
    while i < truth.size and j < prediction.size:
        if prediction[j] < truth[i] - tolerance_samples:
            j += 1
        elif truth[i] < prediction[j] - tolerance_samples:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


def match_event_pairs(
    reference: ArrayLike,
    detected: ArrayLike,
    tolerance_samples: int,
) -> NDArray[np.int64]:
    """Return index pairs for maximum-cardinality, minimum-error matching."""

    truth = np.asarray(reference, dtype=np.int64)
    prediction = np.asarray(detected, dtype=np.int64)
    if np.any(np.diff(truth) < 0) or np.any(np.diff(prediction) < 0):
        raise ValueError("reference and detected positions must be sorted")
    if truth.size == 0 or prediction.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    from scipy.optimize import linear_sum_assignment

    distances = np.abs(truth[:, None] - prediction[None, :])
    # One fewer invalid assignment dominates the largest possible sum of all
    # valid timing errors, hence the solver first maximizes cardinality and only
    # then minimizes total absolute error.
    invalid_cost = (min(truth.size, prediction.size) + 1) * (
        int(tolerance_samples) + 1
    )
    costs = np.where(distances <= tolerance_samples, distances, invalid_cost)
    reference_indices, detection_indices = linear_sum_assignment(costs)
    valid = distances[reference_indices, detection_indices] <= tolerance_samples
    pairs = np.column_stack(
        [reference_indices[valid], detection_indices[valid]]
    ).astype(np.int64, copy=False)
    return pairs[np.argsort(pairs[:, 0])] if pairs.size else pairs.reshape(0, 2)
