"""Core operators for Hierarchical Convexity-Run Decomposition.

The core uses divided slopes, so nonuniform sample locations are supported.
The mathematical operator corresponds to ``atol=rtol=0``.  A tiny relative
tolerance is the safer numerical default because interpolated affine pieces
otherwise acquire round-off curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoundaryRule = Literal["legacy", "minimum_curvature"]


@dataclass(frozen=True)
class Structure:
    """One detail structure between two consecutive knots."""

    level: int
    left: int
    right: int
    sign: int
    amplitude: float
    signed_area: float
    duration: float
    peak_index: int


@dataclass(frozen=True)
class HCRDLevel:
    """One step ``input_baseline = detail + baseline``."""

    index: int
    input_baseline: NDArray[np.float64]
    baseline: NDArray[np.float64]
    detail: NDArray[np.float64]
    knots: NDArray[np.int64]
    structures: tuple[Structure, ...]
    curvature_tolerance: float


@dataclass(frozen=True)
class Decomposition:
    """A complete finite HCRD decomposition."""

    x: NDArray[np.float64]
    original: NDArray[np.float64]
    levels: tuple[HCRDLevel, ...]
    trend: NDArray[np.float64]
    boundary_rule: BoundaryRule

    def reconstruct(self) -> NDArray[np.float64]:
        reconstructed = self.trend.copy()
        for level in self.levels:
            reconstructed += level.detail
        return reconstructed

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def knot_sets(self) -> tuple[NDArray[np.int64], ...]:
        return tuple(level.knots for level in self.levels)


@dataclass(frozen=True)
class SparseHCRDLevel:
    """One hierarchy level without length-``n`` baseline/detail arrays."""

    index: int
    knots: NDArray[np.int64]
    curvature_tolerance: float


@dataclass(frozen=True)
class SparseDecomposition:
    """Knot-only HCRD hierarchy with linear total centred-rule work.

    Every retained knot keeps its original ordinate, so a level is fully
    determined by its knot indices.  Dense baselines, details, and structures
    are materialized only when explicitly requested.
    """

    x: NDArray[np.float64]
    original: NDArray[np.float64]
    levels: tuple[SparseHCRDLevel, ...]
    boundary_rule: BoundaryRule

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def knot_sets(self) -> tuple[NDArray[np.int64], ...]:
        return tuple(level.knots for level in self.levels)

    @property
    def stored_knot_count(self) -> int:
        return sum(level.knots.size for level in self.levels)

    def materialize(self) -> Decomposition:
        """Expand the sparse hierarchy to the backwards-compatible dense API."""

        current = self.original.copy()
        dense_levels: list[HCRDLevel] = []
        for sparse_level in self.levels:
            knots = sparse_level.knots
            baseline = np.interp(
                self.x, self.x[knots], self.original[knots]
            )
            detail = current - baseline
            dense_levels.append(
                HCRDLevel(
                    index=sparse_level.index,
                    input_baseline=current.copy(),
                    baseline=baseline.copy(),
                    detail=detail.copy(),
                    knots=knots.copy(),
                    structures=_structures(
                        sparse_level.index, detail, knots, self.x
                    ),
                    curvature_tolerance=sparse_level.curvature_tolerance,
                )
            )
            current = baseline
        return Decomposition(
            x=self.x.copy(),
            original=self.original.copy(),
            levels=tuple(dense_levels),
            trend=current.copy(),
            boundary_rule=self.boundary_rule,
        )


def _validate_signal(
    signal: ArrayLike, x: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("signal must be a finite one-dimensional array of length >= 2")
    if not np.all(np.isfinite(y)):
        raise ValueError("signal contains NaN or infinite values")
    if x is None:
        sample_locations = np.arange(y.size, dtype=float)
    else:
        sample_locations = np.asarray(x, dtype=float)
        if sample_locations.shape != y.shape:
            raise ValueError("x and signal must have identical shapes")
        if not np.all(np.isfinite(sample_locations)):
            raise ValueError("x contains NaN or infinite values")
        if np.any(np.diff(sample_locations) <= 0):
            raise ValueError("x must be strictly increasing")
    return y, sample_locations


def divided_slopes(signal: ArrayLike, x: ArrayLike | None = None) -> NDArray[np.float64]:
    y, sample_locations = _validate_signal(signal, x)
    return np.diff(y) / np.diff(sample_locations)


def discrete_curvature(
    signal: ArrayLike, x: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Return changes of adjacent divided slopes.

    Entry ``i`` is the discrete curvature at sample ``i + 1``.
    """

    slopes = divided_slopes(signal, x)
    return np.diff(slopes)


def _effective_tolerance(
    slopes: NDArray[np.float64], atol: float, rtol: float
) -> float:
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be nonnegative")
    scale = max(1.0, float(np.max(np.abs(slopes), initial=0.0)))
    return float(atol + rtol * scale)


def _legacy_knots(
    y: NDArray[np.float64], x: NDArray[np.float64], tolerance: float
) -> NDArray[np.int64]:
    """Faithful translation of the knot walk in ``local_blanket.ipynb``."""

    n = y.size
    if n == 2:
        return np.array([0, 1], dtype=np.int64)

    knots: list[int] = []
    point = 0
    is_convex = True
    k1 = k2 = 0.0
    while point < n - 2:
        knots.append(point)
        point += 1
        k1 = (y[point] - y[knots[-1]]) / (x[point] - x[knots[-1]])
        k2 = (y[point + 1] - y[point]) / (x[point + 1] - x[point])

        while abs(k1 - k2) <= tolerance and point < n - 2:
            point += 1
            k1 = k2
            k2 = (y[point + 1] - y[point]) / (x[point + 1] - x[point])

        is_convex = k1 < k2
        while point < n - 2 and (
            (is_convex and k1 < k2 - tolerance)
            or ((not is_convex) and k1 > k2 + tolerance)
            or abs(k1 - k2) <= tolerance
        ):
            point += 1
            k1 = k2
            k2 = (y[point + 1] - y[point]) / (x[point + 1] - x[point])

    if (
        (((not is_convex) and k1 < k2) or (is_convex and k1 > k2))
        and abs(k1 - k2) > tolerance
        and point not in knots
    ):
        knots.append(point)

    if not knots or knots[-1] != n - 1:
        knots.append(n - 1)
    return np.asarray(knots, dtype=np.int64)


def _minimum_curvature_knots(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    tolerance: float,
    eligible_knots: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Greedy maximal convex/concave intervals with centred transitions.

    Given a current knot, the interval is extended through curvature values of
    one sign and insignificant values.  An isolated eligible zero between
    opposite signs is selected as the discrete inflection.  Otherwise the first
    opposite active point closes the interval, as in the legacy walk.  Selecting
    the opposite point is necessary for strict coarsening; selecting the last
    point of the old run can leave the operator unchanged.
    """

    n = y.size
    if n == 2:
        return np.array([0, 1], dtype=np.int64)
    curvature = discrete_curvature(y, x)
    knots = [0]
    start = 0

    while start < n - 1:
        first = start + 1
        while first <= n - 2 and abs(curvature[first - 1]) <= tolerance:
            first += 1
        if first > n - 2:
            knots.append(n - 1)
            break

        active_sign = 1 if curvature[first - 1] > 0 else -1
        last_active = first
        cursor = first + 1
        transition: int | None = None
        while cursor <= n - 2:
            value = curvature[cursor - 1]
            if abs(value) <= tolerance:
                cursor += 1
                continue
            sign = 1 if value > 0 else -1
            if sign != active_sign:
                zero_count = cursor - last_active - 1
                zero_location = last_active + 1
                if zero_count == 1 and zero_location in eligible_knots:
                    # A single sampled zero is the natural discrete inflection
                    # location (e.g. an exactly sampled sinusoidal zero crossing).
                    transition = last_active + 1
                else:
                    # For a long affine plateau, selecting an interior zero would
                    # invent a new geometric vertex and destroy knot nestedness.
                    # Use the smaller-magnitude side of an unsampled zero
                    # crossing, unless that would fail to pass the first active
                    # curvature and therefore prevent strict coarsening.
                    transition = (
                        last_active
                        if abs(curvature[last_active - 1]) < abs(curvature[cursor - 1])
                        and last_active != first
                        else cursor
                    )
                break
            last_active = cursor
            cursor += 1

        if transition is None:
            knots.append(n - 1)
            break
        # Every nonterminal interval contains at least two original segments.
        # This guarantees strict coarsening and logarithmic-depth termination.
        transition = max(min(start + 2, n - 1), min(transition, n - 1))
        if transition == knots[-1]:
            transition += 1
        knots.append(transition)
        start = transition

    if knots[-1] != n - 1:
        knots.append(n - 1)
    return np.asarray(knots, dtype=np.int64)


def find_convexity_knots(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    atol: float = 0.0,
    rtol: float = 64 * np.finfo(float).eps,
    boundary_rule: BoundaryRule = "minimum_curvature",
    _eligible_knots: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Return interpolation knots for one HCRD step."""

    y, sample_locations = _validate_signal(signal, x)
    slopes = np.diff(y) / np.diff(sample_locations)
    tolerance = _effective_tolerance(slopes, atol, rtol)
    if boundary_rule == "legacy":
        return _legacy_knots(y, sample_locations, tolerance)
    if boundary_rule == "minimum_curvature":
        eligible = (
            np.arange(y.size, dtype=np.int64)
            if _eligible_knots is None
            else np.asarray(_eligible_knots, dtype=np.int64)
        )
        return _minimum_curvature_knots(y, sample_locations, tolerance, eligible)
    raise ValueError(f"unknown boundary rule: {boundary_rule}")


def _structures(
    level: int,
    detail: NDArray[np.float64],
    knots: NDArray[np.int64],
    x: NDArray[np.float64],
) -> tuple[Structure, ...]:
    structures: list[Structure] = []
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        segment = detail[left : right + 1]
        local_peak = int(np.argmax(np.abs(segment)))
        peak = int(left + local_peak)
        amplitude = float(abs(segment[local_peak]))
        sign = int(np.sign(segment[local_peak])) if amplitude > 0 else 0
        signed_area = float(np.trapezoid(segment, x[left : right + 1]))
        structures.append(
            Structure(
                level=level,
                left=int(left),
                right=int(right),
                sign=sign,
                amplitude=amplitude,
                signed_area=signed_area,
                duration=float(x[right] - x[left]),
                peak_index=peak,
            )
        )
    return tuple(structures)


def decompose_sparse(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    atol: float = 0.0,
    rtol: float = 64 * np.finfo(float).eps,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
    minimum_knot_spacing: int = 1,
) -> SparseDecomposition:
    """Compute the knot-only hierarchy without dense per-level arrays.

    For the centred rule, the halving recurrence makes the total number of
    visited eligible intervals linear in the input length.  Materializing all
    length-``n`` details remains an explicit ``O(n * depth)`` export operation.
    """

    y, sample_locations = _validate_signal(signal, x)
    if max_levels is not None and max_levels < 1:
        raise ValueError("max_levels must be positive")
    if minimum_knot_spacing < 1:
        raise ValueError("minimum_knot_spacing must be positive")
    levels: list[SparseHCRDLevel] = []
    hard_limit = max_levels if max_levels is not None else y.size
    eligible_knots = np.arange(y.size, dtype=np.int64)

    for level_index in range(hard_limit):
        # Interpolation retains the ordinate at every selected knot.  Hence all
        # later active values can be read directly from the original signal.
        active_values = y[eligible_knots]
        active_locations = sample_locations[eligible_knots]
        slopes = np.diff(active_values) / np.diff(active_locations)
        tolerance = _effective_tolerance(slopes, atol, rtol)
        local_knots = find_convexity_knots(
            active_values,
            active_locations,
            atol=atol,
            rtol=rtol,
            boundary_rule=boundary_rule,
        )
        knots = eligible_knots[local_knots]
        if minimum_knot_spacing > 1 and knots.size > 2:
            spaced = [int(knots[0])]
            for knot in knots[1:-1]:
                if int(knot) - spaced[-1] >= minimum_knot_spacing:
                    spaced.append(int(knot))
            if knots[-1] - spaced[-1] < minimum_knot_spacing and len(spaced) > 1:
                spaced.pop()
            spaced.append(int(knots[-1]))
            knots = np.asarray(spaced, dtype=np.int64)
        levels.append(
            SparseHCRDLevel(
                index=level_index,
                knots=knots.copy(),
                curvature_tolerance=tolerance,
            )
        )
        eligible_knots = knots
        if knots.size == 2:
            break
    else:
        if max_levels is None:
            raise RuntimeError("HCRD failed to terminate within n levels")

    return SparseDecomposition(
        x=sample_locations.copy(),
        original=y.copy(),
        levels=tuple(levels),
        boundary_rule=boundary_rule,
    )


def decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    atol: float = 0.0,
    rtol: float = 64 * np.finfo(float).eps,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
    minimum_knot_spacing: int = 1,
) -> Decomposition:
    """Compute the complete finite HCRD hierarchy with dense level arrays."""

    return decompose_sparse(
        signal,
        x,
        atol=atol,
        rtol=rtol,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
        minimum_knot_spacing=minimum_knot_spacing,
    ).materialize()


def total_variation(signal: ArrayLike) -> float:
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    return float(np.sum(np.abs(np.diff(y))))
