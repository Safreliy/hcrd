"""Stable signed persistence summaries of discrete curvature.

Hard HCRD knot coordinates are discontinuous.  This module instead computes
ordinary zero-dimensional superlevel persistence on the path of positive and
negative discrete curvatures.  The unique essential component is represented
by its birth height; all finite bars use the usual bottleneck metric.  Peak
indices are metadata only and deliberately do not enter the signal
pseudometric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import discrete_curvature


@dataclass(frozen=True)
class PersistenceBar:
    """A finite superlevel-persistence bar on the curvature path."""

    birth: float
    death: float
    peak_index: int

    @property
    def lifetime(self) -> float:
        return self.birth - self.death


@dataclass(frozen=True)
class CurvaturePersistenceDiagram:
    """Finite bars plus the birth of the single essential path component."""

    bars: tuple[PersistenceBar, ...]
    essential_birth: float
    essential_peak_index: int | None


@dataclass(frozen=True)
class CurvaturePersistenceSignature:
    """Ordered positive/negative curvature diagrams for one signal."""

    positive: CurvaturePersistenceDiagram
    negative: CurvaturePersistenceDiagram
    curvature_constant: float


def curvature_lipschitz_constant(
    signal_length: int, x: ArrayLike | None = None
) -> float:
    """Return the exact induced ``l_inf -> l_inf`` norm of divided curvature."""

    if signal_length < 2:
        raise ValueError("signal_length must be at least two")
    if x is None:
        locations = np.arange(signal_length, dtype=float)
    else:
        locations = np.asarray(x, dtype=float)
        if locations.shape != (signal_length,):
            raise ValueError("x must have length signal_length")
        if not np.all(np.isfinite(locations)) or np.any(np.diff(locations) <= 0):
            raise ValueError("x must be finite and strictly increasing")
    if signal_length == 2:
        return 0.0
    spacings = np.diff(locations)
    row_norms = 2.0 * (1.0 / spacings[:-1] + 1.0 / spacings[1:])
    return float(np.max(row_norms))


def _superlevel_diagram(
    values: NDArray[np.float64], *, index_offset: int
) -> CurvaturePersistenceDiagram:
    """Compute ordinary H0 persistence for a vertex-filtered finite path."""

    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("persistence values must be a finite one-dimensional array")
    if values.size == 0:
        return CurvaturePersistenceDiagram((), 0.0, None)
    if np.any(values < 0):
        raise ValueError("superlevel curvature magnitudes must be nonnegative")

    size = values.size
    parent = np.arange(size, dtype=np.int64)
    births = values.copy()
    peaks = np.arange(size, dtype=np.int64)
    active = np.zeros(size, dtype=bool)
    bars: list[PersistenceBar] = []

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            following = int(parent[index])
            parent[index] = root
            index = following
        return root

    def merge(first: int, second: int, level: float) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        first_key = (float(births[first_root]), -int(peaks[first_root]))
        second_key = (float(births[second_root]), -int(peaks[second_root]))
        if first_key >= second_key:
            survivor, dying = first_root, second_root
        else:
            survivor, dying = second_root, first_root
        birth = float(births[dying])
        if birth > level:
            bars.append(
                PersistenceBar(
                    birth=birth,
                    death=float(level),
                    peak_index=int(peaks[dying]) + index_offset,
                )
            )
        parent[dying] = survivor

    order = np.lexsort((np.arange(size), -values))
    for vertex_value_index in order:
        vertex = int(vertex_value_index)
        level = float(values[vertex])
        active[vertex] = True
        if vertex > 0 and active[vertex - 1]:
            merge(vertex, vertex - 1, level)
        if vertex + 1 < size and active[vertex + 1]:
            merge(vertex, vertex + 1, level)

    root = find(0)
    bars.sort(key=lambda bar: (-bar.lifetime, -bar.birth, bar.peak_index))
    return CurvaturePersistenceDiagram(
        bars=tuple(bars),
        essential_birth=float(births[root]),
        essential_peak_index=int(peaks[root]) + index_offset,
    )


def curvature_persistence(
    signal: ArrayLike, x: ArrayLike | None = None
) -> CurvaturePersistenceSignature:
    """Return stable positive and negative H0 curvature persistence."""

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be finite, one-dimensional, and length >= 2")
    curvature = discrete_curvature(y, x)
    positive = np.maximum(curvature, 0.0)
    negative = np.maximum(-curvature, 0.0)
    return CurvaturePersistenceSignature(
        positive=_superlevel_diagram(positive, index_offset=1),
        negative=_superlevel_diagram(negative, index_offset=1),
        curvature_constant=curvature_lipschitz_constant(y.size, x),
    )


def bottleneck_distance(
    first: CurvaturePersistenceDiagram,
    second: CurvaturePersistenceDiagram,
) -> float:
    """Exact bottleneck distance for the finite bars of two diagrams.

    The essential classes are intentionally handled by
    :func:`curvature_persistence_distance`, because they cannot match finite
    points or the diagonal.
    """

    from scipy.optimize import linear_sum_assignment

    first_bars = first.bars
    second_bars = second.bars
    n_first = len(first_bars)
    n_second = len(second_bars)
    size = n_first + n_second
    if size == 0:
        return 0.0

    costs = np.full((size, size), np.inf, dtype=float)
    for row, left in enumerate(first_bars):
        for column, right in enumerate(second_bars):
            costs[row, column] = max(
                abs(left.birth - right.birth), abs(left.death - right.death)
            )
        costs[row, n_second + row] = left.lifetime / 2.0
    for column, right in enumerate(second_bars):
        costs[n_first + column, column] = right.lifetime / 2.0
    costs[n_first:, n_second:] = 0.0

    thresholds = np.unique(costs[np.isfinite(costs)])
    low = 0
    high = len(thresholds) - 1
    while low < high:
        middle = (low + high) // 2
        allowed = costs <= thresholds[middle]
        rows, columns = linear_sum_assignment((~allowed).astype(np.int8))
        if bool(np.all(allowed[rows, columns])):
            high = middle
        else:
            low = middle + 1
    return float(thresholds[low])


def curvature_persistence_distance(
    first: CurvaturePersistenceSignature,
    second: CurvaturePersistenceSignature,
) -> float:
    """Signed bottleneck distance, including both essential birth heights.

    Pulling this distance back from persistence summaries to signals gives a
    pseudometric because distinct signals can have the same summaries.
    """

    return max(
        bottleneck_distance(first.positive, second.positive),
        abs(first.positive.essential_birth - second.positive.essential_birth),
        bottleneck_distance(first.negative, second.negative),
        abs(first.negative.essential_birth - second.negative.essential_birth),
    )
