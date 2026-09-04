"""Design-identified convex-to-concave transition sets.

The observation law in fixed-design regression identifies the sampled mean
vector, not a particular interpolation between design points.  This module
computes the closure of all transition locations admitted by at least one
real-valued continuation that is convex before the transition and concave
after it.  Monotonicity and continuity at the transition are not imposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DesignIdentifiedTransitionSet:
    """Closed components of the transition set identified by sampled means."""

    components: tuple[tuple[float, float], ...]

    @property
    def empty(self) -> bool:
        """Whether no convex-to-concave continuation exists."""

        return not self.components

    @property
    def left(self) -> float | None:
        """Infimum of the identified set, or ``None`` for an empty set."""

        return None if self.empty else self.components[0][0]

    @property
    def right(self) -> float | None:
        """Supremum of the identified set, or ``None`` for an empty set."""

        return None if self.empty else self.components[-1][1]

    @property
    def hull(self) -> tuple[float, float] | None:
        """Smallest closed interval containing the identified set."""

        if self.empty:
            return None
        assert self.left is not None and self.right is not None
        return self.left, self.right


def _validated_inputs(
    x: ArrayLike,
    mean: ArrayLike,
    domain: tuple[float, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    locations = np.asarray(x, dtype=float)
    values = np.asarray(mean, dtype=float)
    if locations.ndim != 1 or locations.size < 3:
        raise ValueError("x must be one-dimensional with at least three points")
    if values.shape != locations.shape:
        raise ValueError("mean must have the same one-dimensional shape as x")
    if not np.all(np.isfinite(locations)) or np.any(np.diff(locations) <= 0.0):
        raise ValueError("x must be finite and strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("mean must be finite")

    if domain is None:
        domain_left, domain_right = float(locations[0]), float(locations[-1])
    else:
        domain_left, domain_right = map(float, domain)
    if (
        not np.isfinite(domain_left)
        or not np.isfinite(domain_right)
        or domain_left >= domain_right
        or domain_left > locations[0]
        or domain_right < locations[-1]
    ):
        raise ValueError("domain must contain x and have increasing finite endpoints")
    return locations, values, domain_left, domain_right


def _merge_closed_components(
    pieces: list[tuple[float, float]], tolerance: float
) -> tuple[tuple[float, float], ...]:
    if not pieces:
        return ()
    ordered = sorted(pieces)
    merged: list[list[float]] = [[float(ordered[0][0]), float(ordered[0][1])]]
    for left, right in ordered[1:]:
        if left <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], float(right))
        else:
            merged.append([float(left), float(right)])
    return tuple((left, right) for left, right in merged)


def design_identified_transition_set(
    x: ArrayLike,
    mean: ArrayLike,
    *,
    domain: tuple[float, float] | None = None,
    slope_tolerance: float | None = None,
) -> DesignIdentifiedTransitionSet:
    """Compute the fixed-design identified transition set in linear time.

    Let ``s[i]`` be the divided slope between two adjacent design points.  A
    prefix admits a convex continuation exactly when its slopes are
    nondecreasing, and a suffix admits a concave continuation exactly when its
    slopes are nonincreasing.  For a transition ``m`` between ``x[k]`` and
    ``x[k+1]``, the two pieces can meet at a common value ``v`` exactly when

    ``mean[k] + s[k-1] * (m-x[k]) <= v``

    and

    ``v <= mean[k+1] - s[k+1] * (x[k+1]-m)``,

    with the missing boundary inequality omitted.  These linear inequalities
    give one closed feasible subinterval per design gap.  Design points and
    domain endpoints are checked separately, and touching pieces are merged.

    ``slope_tolerance`` is an absolute tolerance for comparisons of divided
    slopes.  The default only absorbs floating-point error at the scale of the
    computed slopes; callers can set it to zero for exact arithmetic encoded
    without roundoff.
    """

    locations, values, domain_left, domain_right = _validated_inputs(x, mean, domain)
    slopes = np.diff(values) / np.diff(locations)
    if slope_tolerance is None:
        slope_scale = max(1.0, float(np.max(np.abs(slopes))))
        slope_tolerance = 2048.0 * np.finfo(float).eps * slope_scale
    elif not np.isfinite(slope_tolerance) or slope_tolerance < 0.0:
        raise ValueError("slope_tolerance must be finite and nonnegative")
    tolerance = float(slope_tolerance)

    n = locations.size
    convex_prefix = np.ones(n, dtype=bool)
    for index in range(2, n):
        convex_prefix[index] = bool(
            convex_prefix[index - 1]
            and slopes[index - 1] >= slopes[index - 2] - tolerance
        )

    concave_suffix = np.ones(n, dtype=bool)
    for index in range(n - 3, -1, -1):
        concave_suffix[index] = bool(
            concave_suffix[index + 1] and slopes[index] >= slopes[index + 1] - tolerance
        )

    pieces: list[tuple[float, float]] = []
    if concave_suffix[0]:
        pieces.append((domain_left, float(locations[0])))
    for index in range(n):
        if convex_prefix[index] and concave_suffix[index]:
            point = float(locations[index])
            pieces.append((point, point))

    for index in range(n - 1):
        if not (convex_prefix[index] and concave_suffix[index + 1]):
            continue
        t_left, t_right = 0.0, 1.0
        if index >= 1 and index + 1 <= n - 2:
            coefficient = slopes[index - 1] - slopes[index + 1]
            bound = slopes[index] - slopes[index + 1]
            if abs(coefficient) <= tolerance:
                if bound < -tolerance:
                    continue
            elif coefficient > 0.0:
                t_right = min(t_right, float(bound / coefficient))
            else:
                t_left = max(t_left, float(bound / coefficient))

        t_left = max(0.0, t_left)
        t_right = min(1.0, t_right)
        if t_left <= t_right + tolerance:
            gap = locations[index + 1] - locations[index]
            left = float(locations[index] + t_left * gap)
            right = float(locations[index] + t_right * gap)
            pieces.append((left, right))

    if convex_prefix[-1]:
        pieces.append((float(locations[-1]), domain_right))

    location_scale = max(
        1.0, abs(domain_left), abs(domain_right), float(np.max(np.abs(locations)))
    )
    location_tolerance = 4096.0 * np.finfo(float).eps * location_scale
    return DesignIdentifiedTransitionSet(
        components=_merge_closed_components(pieces, location_tolerance)
    )
