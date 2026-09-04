"""An honest pointwise-band baseline for transition sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.stats import norm


@dataclass(frozen=True)
class PointwiseProjectionConfidenceSet:
    """Outer set from a conservative discrete split relaxation."""

    left: float
    right: float
    empty: bool
    alpha: float
    critical_value: float
    minimum_feasible_cut: int | None
    maximum_feasible_cut: int | None
    feasible_cut_count: int

    @property
    def interval(self) -> tuple[float, float] | None:
        if self.empty:
            return None
        return (self.left, self.right)

    @property
    def width(self) -> float:
        return float("nan") if self.empty else self.right - self.left


def gaussian_pointwise_shape_projection(
    x: ArrayLike,
    y: ArrayLike,
    *,
    noise_scale: float,
    alpha: float,
    domain: tuple[float, float] | None = None,
) -> PointwiseProjectionConfidenceSet:
    """Project an exact Gaussian pointwise band onto a split relaxation.

    This is a generic, deliberately simple comparison method. It uses a
    Bonferroni band for every sampled mean value. A candidate cut is retained
    when the band contains a discretely convex prefix and, in a separate
    feasibility problem, a discretely concave suffix. The two feasible pieces
    need not share a transition value. Thus this is a conservative relaxation,
    not an exact projection onto the continuous transition class. It remains
    an outer confidence set under the same known-scale Gaussian model. The
    optional ``domain`` must contain the design. It lets the two boundary cuts
    represent transitions outside the observed range; by default the declared
    domain is the observed range ``[x[0], x[-1]]``.
    """

    locations = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float)
    if (
        locations.ndim != 1
        or locations.size < 3
        or values.shape != locations.shape
        or not np.all(np.isfinite(locations))
        or not np.all(np.isfinite(values))
        or np.any(np.diff(locations) <= 0.0)
    ):
        raise ValueError("x and y must be finite matched vectors with increasing x")
    sigma = float(noise_scale)
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError("noise_scale must be nonnegative and finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
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

    critical = float(norm.ppf(1.0 - alpha / (2.0 * locations.size)))
    lower = values - critical * sigma
    upper = values + critical * sigma
    n = int(locations.size)

    maximum_convex_cut = _maximum_convex_prefix(locations, lower, upper)
    minimum_concave_cut = _minimum_concave_suffix(locations, lower, upper)
    empty = minimum_concave_cut > maximum_convex_cut
    if empty:
        return PointwiseProjectionConfidenceSet(
            left=float("inf"),
            right=float("-inf"),
            empty=True,
            alpha=float(alpha),
            critical_value=critical,
            minimum_feasible_cut=None,
            maximum_feasible_cut=None,
            feasible_cut_count=0,
        )

    left = (
        domain_left
        if minimum_concave_cut == 0
        else float(locations[minimum_concave_cut - 1])
    )
    right = (
        domain_right
        if maximum_convex_cut == n
        else float(locations[maximum_convex_cut])
    )
    return PointwiseProjectionConfidenceSet(
        left=left,
        right=right,
        empty=False,
        alpha=float(alpha),
        critical_value=critical,
        minimum_feasible_cut=minimum_concave_cut,
        maximum_feasible_cut=maximum_convex_cut,
        feasible_cut_count=maximum_convex_cut - minimum_concave_cut + 1,
    )


def _maximum_convex_prefix(
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> int:
    if _shape_feasible(x, lower, upper, convex=True):
        return int(x.size)
    low = 0
    high = int(x.size)
    while low < high:
        middle = (low + high + 1) // 2
        if _shape_feasible(x[:middle], lower[:middle], upper[:middle], convex=True):
            low = middle
        else:
            high = middle - 1
    return low


def _minimum_concave_suffix(
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> int:
    if _shape_feasible(x, lower, upper, convex=False):
        return 0
    low = 0
    high = int(x.size)
    while low < high:
        middle = (low + high) // 2
        if _shape_feasible(x[middle:], lower[middle:], upper[middle:], convex=False):
            high = middle
        else:
            low = middle + 1
    return low


def _shape_feasible(
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    convex: bool,
) -> bool:
    size = int(x.size)
    if size < 3:
        return True
    first_gap = np.diff(x[:-1])
    second_gap = np.diff(x[1:])
    row_count = size - 2
    row = np.repeat(np.arange(row_count, dtype=np.int64), 3)
    column = np.column_stack(
        [
            np.arange(row_count, dtype=np.int64),
            np.arange(1, row_count + 1, dtype=np.int64),
            np.arange(2, row_count + 2, dtype=np.int64),
        ]
    ).ravel()
    coefficient = np.column_stack(
        [
            -1.0 / first_gap,
            1.0 / first_gap + 1.0 / second_gap,
            -1.0 / second_gap,
        ]
    )
    if not convex:
        coefficient = -coefficient
    coefficient /= np.max(np.abs(coefficient), axis=1)[:, None]
    constraints = csr_matrix(
        (coefficient.ravel(), (row, column)), shape=(row_count, size)
    )
    result = linprog(
        np.zeros(size, dtype=float),
        A_ub=constraints,
        b_ub=np.zeros(row_count, dtype=float),
        bounds=np.column_stack([lower, upper]),
        method="highs",
    )
    if result.status not in (0, 2):
        raise RuntimeError(f"linear feasibility solver failed: {result.message}")
    return result.status == 0
