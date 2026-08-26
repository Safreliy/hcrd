"""Replicate-split inference for HCRD-selected signed chord structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .core import decompose_sparse


@dataclass(frozen=True)
class InferredStructure:
    """One guide-selected structure scored on an independent replicate."""

    level: int
    left: int
    right: int
    active_sample_count: int
    signed_polygon_area: float
    standard_error: float
    z_score: float
    p_value: float
    holm_p_value: float
    significant: bool


@dataclass(frozen=True)
class InferredMatchedStructure:
    """One guide-selected HCRD shape used as an independent matched filter."""

    level: int
    left: int
    right: int
    active_sample_count: int
    template_norm: float
    matched_amplitude: float
    amplitude_standard_error: float
    z_score: float
    p_value: float
    holm_p_value: float
    significant: bool


def chord_area_coefficients(x: ArrayLike) -> NDArray[np.float64]:
    """Coefficients of trapezoidal signed area relative to the endpoint chord."""

    locations = np.asarray(x, dtype=float)
    if (
        locations.ndim != 1
        or locations.size < 2
        or not np.all(np.isfinite(locations))
        or np.any(np.diff(locations) <= 0.0)
    ):
        raise ValueError("x must be a finite strictly increasing vector")
    weights = np.empty(locations.size, dtype=float)
    widths = np.diff(locations)
    weights[0] = widths[0] / 2.0
    weights[-1] = widths[-1] / 2.0
    if locations.size > 2:
        weights[1:-1] = (locations[2:] - locations[:-2]) / 2.0
    span = locations[-1] - locations[0]
    right_fraction = (locations - locations[0]) / span
    left_fraction = 1.0 - right_fraction
    coefficients = weights.copy()
    coefficients[0] -= float(np.dot(weights, left_fraction))
    coefficients[-1] -= float(np.dot(weights, right_fraction))
    return coefficients


def _holm(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = p_values.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def infer_hcrd_structures(
    guide_signal: ArrayLike,
    scoring_replicate: ArrayLike,
    *,
    noise_sigma: float,
    x: ArrayLike | None = None,
    max_levels: int | None = 8,
    familywise_alpha: float = 0.05,
) -> tuple[InferredStructure, ...]:
    """Select structures on a guide and test them on an independent replicate.

    Conditional validity requires the scoring replicate to have independent
    centred Gaussian errors with known standard deviation no larger than
    ``noise_sigma``. Each null states that the replicate's mean is affine on
    the active samples of that guide-selected interval. Holm adjustment controls
    family-wise error under arbitrary dependence between structure statistics.
    """

    guide = np.asarray(guide_signal, dtype=float)
    scoring = np.asarray(scoring_replicate, dtype=float)
    if guide.shape != scoring.shape or guide.ndim != 1 or guide.size < 2:
        raise ValueError("guide_signal and scoring_replicate must be aligned vectors")
    if not np.all(np.isfinite(guide)) or not np.all(np.isfinite(scoring)):
        raise ValueError("signals must be finite")
    if noise_sigma <= 0.0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and strictly positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    locations = (
        np.arange(guide.size, dtype=float)
        if x is None
        else np.asarray(x, dtype=float)
    )
    if locations.shape != guide.shape:
        raise ValueError("x must be aligned with the signals")

    hierarchy = decompose_sparse(guide, locations, max_levels=max_levels)
    previous_knots = np.arange(guide.size, dtype=np.int64)
    raw: list[tuple[int, int, int, int, float, float, float, float]] = []
    for level in hierarchy.levels:
        positions = np.searchsorted(previous_knots, level.knots)
        if not np.array_equal(previous_knots[positions], level.knots):
            raise RuntimeError("guide hierarchy is not nested")
        for start, stop, left, right in zip(
            positions[:-1],
            positions[1:],
            level.knots[:-1],
            level.knots[1:],
            strict=True,
        ):
            active = previous_knots[int(start) : int(stop) + 1]
            coefficients = chord_area_coefficients(locations[active])
            area = float(np.dot(coefficients, scoring[active]))
            standard_error = float(noise_sigma * np.linalg.norm(coefficients))
            # Two endpoints alone have an identically zero chord residual and
            # contain no testable interior structure.
            if standard_error == 0.0:
                continue
            z_score = area / standard_error
            p_value = float(2.0 * norm.sf(abs(z_score)))
            raw.append(
                (
                    int(level.index),
                    int(left),
                    int(right),
                    int(active.size),
                    area,
                    standard_error,
                    z_score,
                    p_value,
                )
            )
        previous_knots = level.knots
    if not raw:
        return ()
    adjusted = _holm(np.asarray([item[-1] for item in raw], dtype=float))
    return tuple(
        InferredStructure(
            level=item[0],
            left=item[1],
            right=item[2],
            active_sample_count=item[3],
            signed_polygon_area=item[4],
            standard_error=item[5],
            z_score=item[6],
            p_value=item[7],
            holm_p_value=float(adjusted[index]),
            significant=bool(adjusted[index] <= familywise_alpha),
        )
        for index, item in enumerate(raw)
    )


def infer_hcrd_matched_structures(
    guide_signal: ArrayLike,
    scoring_replicate: ArrayLike,
    *,
    noise_sigma: float,
    x: ArrayLike | None = None,
    max_levels: int | None = 8,
    familywise_alpha: float = 0.05,
) -> tuple[InferredMatchedStructure, ...]:
    """Test guide-selected HCRD shapes on an independent scoring replicate.

    Every guide detail is projected off the affine span before it becomes a
    matched-filter template. Conditional on an independent guide, the scoring
    statistic is therefore exactly standard normal for any affine mean and iid
    Gaussian noise of known standard deviation. Holm adjustment controls FWER
    over all selected intervals and hierarchy levels.
    """

    guide = np.asarray(guide_signal, dtype=float)
    scoring = np.asarray(scoring_replicate, dtype=float)
    if guide.shape != scoring.shape or guide.ndim != 1 or guide.size < 2:
        raise ValueError("guide_signal and scoring_replicate must be aligned vectors")
    if not np.all(np.isfinite(guide)) or not np.all(np.isfinite(scoring)):
        raise ValueError("signals must be finite")
    if noise_sigma <= 0.0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and strictly positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    locations = (
        np.arange(guide.size, dtype=float)
        if x is None
        else np.asarray(x, dtype=float)
    )
    if locations.shape != guide.shape:
        raise ValueError("x must be aligned with the signals")

    hierarchy = decompose_sparse(guide, locations, max_levels=max_levels)
    previous_knots = np.arange(guide.size, dtype=np.int64)
    raw: list[tuple[int, int, int, int, float, float, float, float, float]] = []
    for level in hierarchy.levels:
        positions = np.searchsorted(previous_knots, level.knots)
        if not np.array_equal(previous_knots[positions], level.knots):
            raise RuntimeError("guide hierarchy is not nested")
        for start, stop, left, right in zip(
            positions[:-1],
            positions[1:],
            level.knots[:-1],
            level.knots[1:],
            strict=True,
        ):
            active = previous_knots[int(start) : int(stop) + 1]
            if active.size < 3:
                continue
            active_x = locations[active]
            guide_chord = np.interp(
                active_x,
                active_x[[0, -1]],
                guide[active[[0, -1]]],
            )
            guide_detail = guide[active] - guide_chord
            centred_x = active_x - np.mean(active_x)
            x_scale = float(np.ptp(active_x))
            design = np.column_stack(
                [np.ones(active.size), centred_x / x_scale]
            )
            affine_fit = design @ np.linalg.lstsq(
                design, guide_detail, rcond=None
            )[0]
            template = guide_detail - affine_fit
            template_norm = float(np.linalg.norm(template))
            numerical_floor = (
                np.finfo(float).eps
                * np.sqrt(active.size)
                * max(1.0, float(np.max(np.abs(guide_detail), initial=0.0)))
            )
            if template_norm <= numerical_floor:
                continue
            projection = float(np.dot(template, scoring[active]))
            z_score = projection / (noise_sigma * template_norm)
            amplitude = projection / template_norm**2
            standard_error = noise_sigma / template_norm
            p_value = float(2.0 * norm.sf(abs(z_score)))
            raw.append(
                (
                    int(level.index),
                    int(left),
                    int(right),
                    int(active.size),
                    template_norm,
                    amplitude,
                    standard_error,
                    z_score,
                    p_value,
                )
            )
        previous_knots = level.knots
    if not raw:
        return ()
    adjusted = _holm(np.asarray([item[-1] for item in raw], dtype=float))
    return tuple(
        InferredMatchedStructure(
            level=item[0],
            left=item[1],
            right=item[2],
            active_sample_count=item[3],
            template_norm=item[4],
            matched_amplitude=item[5],
            amplitude_standard_error=item[6],
            z_score=item[7],
            p_value=item[8],
            holm_p_value=float(adjusted[index]),
            significant=bool(adjusted[index] <= familywise_alpha),
        )
        for index, item in enumerate(raw)
    )
