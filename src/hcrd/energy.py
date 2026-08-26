"""Geometric mass and energy-like summaries of an HCRD hierarchy.

The polygon area is the exact area between a piecewise-linear HCRD input
baseline and its retained chord.  It is an L1 geometric mass, not Parseval
energy: HCRD details are not orthogonal.  Quadratic energy is reported
separately, and the amplitude-duration triangle is an inexpensive surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import SparseDecomposition, decompose_sparse


@dataclass(frozen=True)
class StructureEnergy:
    """Energy-like quantities for one signed chord structure."""

    level: int
    left: int
    right: int
    sign: int
    duration: float
    amplitude: float
    signed_polygon_area: float
    polygon_area: float
    quadratic_energy: float
    triangle_area: float
    shape_factor: float

    @property
    def quadratic_shape_factor(self) -> float:
        """Return the scale-free ``area**2 / (duration * L2 energy)``."""

        denominator = self.duration * self.quadratic_energy
        return self.polygon_area**2 / denominator if denominator > 0.0 else 0.0


@dataclass(frozen=True)
class LevelEnergy:
    """Additive summaries over all structures at one HCRD level."""

    level: int
    knot_count: int
    structure_count: int
    signed_polygon_area: float
    polygon_area: float
    positive_polygon_area: float
    negative_polygon_area: float
    quadratic_energy: float
    triangle_area: float
    peak_amplitude: float
    mean_amplitude: float
    mean_duration: float
    area_concentration: float
    weighted_shape_factor: float


ENERGY_FEATURE_NAMES_PER_LEVEL = (
    "log1p_structure_count",
    "knot_fraction",
    "log1p_polygon_area",
    "log1p_quadratic_energy",
    "log1p_triangle_area",
    "log1p_peak_amplitude",
    "log1p_mean_amplitude",
    "positive_area_fraction",
    "sign_balance",
    "area_concentration",
    "weighted_shape_factor",
    "mean_duration_fraction",
)


def _integrate_absolute_linear(
    values: NDArray[np.float64], x: NDArray[np.float64]
) -> float:
    """Integrate the absolute value of a piecewise-linear function exactly."""

    left = values[:-1]
    right = values[1:]
    widths = np.diff(x)
    same_sign = left * right >= 0.0
    areas = np.empty_like(widths)
    areas[same_sign] = (
        0.5 * widths[same_sign] * (np.abs(left[same_sign]) + np.abs(right[same_sign]))
    )
    opposite = ~same_sign
    denominator = np.abs(left[opposite]) + np.abs(right[opposite])
    areas[opposite] = np.divide(
        0.5 * widths[opposite] * (left[opposite] ** 2 + right[opposite] ** 2),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(np.sum(areas))


def _integrate_square_linear(
    values: NDArray[np.float64], x: NDArray[np.float64]
) -> float:
    """Integrate the square of a piecewise-linear function exactly."""

    left = values[:-1]
    right = values[1:]
    return float(np.sum(np.diff(x) * (left**2 + left * right + right**2) / 3.0))


def sparse_structure_energies(
    decomposition: SparseDecomposition,
) -> tuple[tuple[StructureEnergy, ...], ...]:
    """Compute exact polygonal summaries without dense length-n details.

    At the active knots of a level's input baseline, ordinates equal the
    original samples.  Therefore each residual is integrated only on the
    previous knot set; no dense component materialization is needed.
    """

    x = decomposition.x
    y = decomposition.original
    previous_knots = np.arange(y.size, dtype=np.int64)
    output: list[tuple[StructureEnergy, ...]] = []
    for level in decomposition.levels:
        knots = level.knots
        positions = np.searchsorted(previous_knots, knots)
        if np.any(positions >= previous_knots.size) or not np.array_equal(
            previous_knots[positions], knots
        ):
            raise RuntimeError("HCRD knot hierarchy is not nested")
        structures: list[StructureEnergy] = []
        for start_position, stop_position, left_index, right_index in zip(
            positions[:-1], positions[1:], knots[:-1], knots[1:], strict=True
        ):
            active = previous_knots[int(start_position) : int(stop_position) + 1]
            active_x = x[active]
            chord = np.interp(
                active_x,
                np.asarray([x[left_index], x[right_index]], dtype=float),
                np.asarray([y[left_index], y[right_index]], dtype=float),
            )
            residual = y[active] - chord
            signed_area = float(np.trapezoid(residual, active_x))
            polygon_area = _integrate_absolute_linear(residual, active_x)
            quadratic_energy = _integrate_square_linear(residual, active_x)
            peak_position = int(np.argmax(np.abs(residual)))
            amplitude = float(abs(residual[peak_position]))
            peak_value = float(residual[peak_position])
            sign = int(np.sign(peak_value)) if amplitude > 0.0 else 0
            duration = float(x[right_index] - x[left_index])
            triangle_area = 0.5 * duration * amplitude
            envelope_area = duration * amplitude
            shape_factor = polygon_area / envelope_area if envelope_area > 0.0 else 0.0
            structures.append(
                StructureEnergy(
                    level=level.index,
                    left=int(left_index),
                    right=int(right_index),
                    sign=sign,
                    duration=duration,
                    amplitude=amplitude,
                    signed_polygon_area=signed_area,
                    polygon_area=polygon_area,
                    quadratic_energy=quadratic_energy,
                    triangle_area=triangle_area,
                    shape_factor=shape_factor,
                )
            )
        output.append(tuple(structures))
        previous_knots = knots
    return tuple(output)


def multiscale_area_density(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    max_levels: int | None = None,
) -> NDArray[np.float64]:
    """Return the exact polygon-mass density at every time and HCRD level.

    Row ``level`` is ``abs(detail_level)``.  Consequently, integrating a row
    over ``x`` gives the sum of exact polygon areas of all structures at that
    level.  The matrix therefore turns the scalar per-level masses into a
    multivariate time series without discarding where the mass occurred.

    This representation is intended for temporal tasks such as anomaly and
    change-point detection.  It is computed from the sparse knot hierarchy but
    materializes one dense row per requested level, so its storage cost is
    ``O(n * levels)``.
    """

    hierarchy = decompose_sparse(signal, x, max_levels=max_levels)
    current = hierarchy.original.copy()
    densities = np.empty((hierarchy.depth, current.size), dtype=float)
    for row, level in enumerate(hierarchy.levels):
        knots = level.knots
        baseline = np.interp(
            hierarchy.x,
            hierarchy.x[knots],
            hierarchy.original[knots],
        )
        densities[row] = np.abs(current - baseline)
        current = baseline
    return densities


def level_energies(
    decomposition: SparseDecomposition,
) -> tuple[LevelEnergy, ...]:
    """Aggregate exact structure quantities level by level."""

    structure_levels = sparse_structure_energies(decomposition)
    summaries: list[LevelEnergy] = []
    for sparse_level, structures in zip(
        decomposition.levels, structure_levels, strict=True
    ):
        polygon_areas = np.asarray([item.polygon_area for item in structures])
        amplitudes = np.asarray([item.amplitude for item in structures])
        durations = np.asarray([item.duration for item in structures])
        polygon_area = float(np.sum(polygon_areas))
        positive = float(
            sum(item.polygon_area for item in structures if item.sign > 0)
        )
        negative = float(
            sum(item.polygon_area for item in structures if item.sign < 0)
        )
        concentration = (
            float(np.sum((polygon_areas / polygon_area) ** 2))
            if polygon_area > 0.0
            else 0.0
        )
        shape_weights = np.asarray(
            [item.duration * item.amplitude for item in structures]
        )
        weight_sum = float(np.sum(shape_weights))
        weighted_shape = (
            float(
                np.sum(
                    shape_weights
                    * np.asarray([item.shape_factor for item in structures])
                )
                / weight_sum
            )
            if weight_sum > 0.0
            else 0.0
        )
        summaries.append(
            LevelEnergy(
                level=sparse_level.index,
                knot_count=int(sparse_level.knots.size),
                structure_count=len(structures),
                signed_polygon_area=float(
                    sum(item.signed_polygon_area for item in structures)
                ),
                polygon_area=polygon_area,
                positive_polygon_area=positive,
                negative_polygon_area=negative,
                quadratic_energy=float(
                    sum(item.quadratic_energy for item in structures)
                ),
                triangle_area=float(sum(item.triangle_area for item in structures)),
                peak_amplitude=float(np.max(amplitudes, initial=0.0)),
                mean_amplitude=float(np.mean(amplitudes)) if amplitudes.size else 0.0,
                mean_duration=float(np.mean(durations)) if durations.size else 0.0,
                area_concentration=concentration,
                weighted_shape_factor=weighted_shape,
            )
        )
    return tuple(summaries)


def multiscale_energy_feature_names(max_levels: int) -> tuple[str, ...]:
    """Return stable names corresponding to ``multiscale_energy_features``."""

    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    return tuple(
        f"level_{level}_{name}"
        for level in range(max_levels)
        for name in ENERGY_FEATURE_NAMES_PER_LEVEL
    )


def multiscale_energy_features(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    max_levels: int = 6,
    amplitude_scale: float | None = None,
) -> NDArray[np.float64]:
    """Return fixed-length, interpretable HCRD energy features.

    ``amplitude_scale`` can be a healthy-baseline scale learned without test
    label information.  Leaving it unset preserves absolute amplitude, which
    is important in degradation monitoring.  Time units are inherited from
    ``x``; use a normalized grid when duration invariance is desired.
    """

    if max_levels < 1:
        raise ValueError("max_levels must be positive")
    values = np.asarray(signal, dtype=float)
    if amplitude_scale is not None:
        if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
            raise ValueError("amplitude_scale must be finite and positive")
        values = values / amplitude_scale
    decomposition = decompose_sparse(values, x, max_levels=max_levels)
    summaries = level_energies(decomposition)
    total_duration = float(decomposition.x[-1] - decomposition.x[0])
    total_duration = max(total_duration, np.finfo(float).eps)
    sample_count = decomposition.original.size
    rows: list[float] = []
    for level_index in range(max_levels):
        if level_index >= len(summaries):
            rows.extend([0.0] * len(ENERGY_FEATURE_NAMES_PER_LEVEL))
            continue
        item = summaries[level_index]
        total_area = item.polygon_area
        positive_fraction = (
            item.positive_polygon_area / total_area if total_area > 0.0 else 0.0
        )
        sign_balance = (
            (item.positive_polygon_area - item.negative_polygon_area) / total_area
            if total_area > 0.0
            else 0.0
        )
        rows.extend(
            [
                np.log1p(item.structure_count),
                item.knot_count / sample_count,
                np.log1p(item.polygon_area),
                np.log1p(item.quadratic_energy),
                np.log1p(item.triangle_area),
                np.log1p(item.peak_amplitude),
                np.log1p(item.mean_amplitude),
                positive_fraction,
                sign_balance,
                item.area_concentration,
                item.weighted_shape_factor,
                item.mean_duration / total_duration,
            ]
        )
    features = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(features)):
        raise RuntimeError("nonfinite HCRD energy features")
    return features
