"""Label-free anomaly scores derived from the HCRD area-density spectrum."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .energy import multiscale_area_density

AreaAggregation = Literal["total", "sum", "max", "l2", "transport"]


def _robust_positive_surprise(
    densities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Scale each hierarchy row without labels or distributional assumptions."""

    centre = np.median(densities, axis=1, keepdims=True)
    upper = np.quantile(densities, 0.9, axis=1, keepdims=True)
    scale = upper - centre
    fallback = np.max(densities, axis=1, keepdims=True) - centre
    scale = np.where(scale > 0.0, scale, fallback)
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.maximum((densities - centre) / scale, 0.0)


def hcrd_area_anomaly_score(
    signal: ArrayLike,
    *,
    max_levels: int | None = 8,
    aggregation: AreaAggregation = "sum",
) -> NDArray[np.float64]:
    """Return one label-free anomaly score per sample.

    ``total`` is the unnormalised sum of exact area densities.  The other
    aggregations robustly normalise every HCRD level before combining them,
    which makes their scores invariant to nonzero vertical rescaling and to
    addition of an affine trend.  ``transport`` additionally rewards abrupt
    redistribution of mass between hierarchy levels.
    """

    densities = multiscale_area_density(signal, max_levels=max_levels)
    return aggregate_area_density(densities, aggregation=aggregation)


def aggregate_area_density(
    densities: ArrayLike,
    *,
    aggregation: AreaAggregation = "sum",
) -> NDArray[np.float64]:
    """Aggregate an existing ``levels x time`` HCRD area-density matrix."""

    values = np.asarray(densities, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("densities must have shape (levels >= 1, time >= 2)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("densities must be finite and nonnegative")
    if aggregation == "total":
        return np.sum(values, axis=0)

    surprise = _robust_positive_surprise(values)
    if aggregation == "sum":
        return np.mean(surprise, axis=0)
    if aggregation == "max":
        return np.max(surprise, axis=0)
    if aggregation == "l2":
        return np.sqrt(np.mean(surprise**2, axis=0))
    if aggregation == "transport":
        total = np.sum(surprise, axis=0)
        fractions = np.divide(
            surprise,
            total[None, :],
            out=np.zeros_like(surprise),
            where=total[None, :] > 0.0,
        )
        flux = np.zeros_like(total)
        flux[1:] = 0.5 * np.sum(np.abs(np.diff(fractions, axis=1)), axis=0)
        return np.mean(surprise, axis=0) * (1.0 + flux)
    raise ValueError(f"unknown area aggregation: {aggregation}")
