"""Dense temporal component matrices derived from the sparse HCRD hierarchy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import decompose_sparse


def multiscale_detail_series(
    signal: ArrayLike,
    *,
    max_levels: int | None = None,
) -> NDArray[np.float64]:
    """Return signed HCRD details with shape ``(levels, time)``."""

    hierarchy = decompose_sparse(signal, max_levels=max_levels)
    current = hierarchy.original.copy()
    details = np.empty((hierarchy.depth, current.size), dtype=float)
    for row, level in enumerate(hierarchy.levels):
        knots = level.knots
        baseline = np.interp(
            hierarchy.x,
            hierarchy.x[knots],
            hierarchy.original[knots],
        )
        details[row] = current - baseline
        current = baseline
    return details

