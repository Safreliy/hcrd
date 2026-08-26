"""Buffered folds and Gaussian pivots for finite-memory signal streams."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BufferedFold:
    """One scoring block, its excluded dependence buffer, and its guide set."""

    scoring_indices: NDArray[np.int64]
    buffer_indices: NDArray[np.int64]
    guide_indices: NDArray[np.int64]


@dataclass(frozen=True)
class GaussianContrastPivot:
    """Known-covariance Gaussian contrast pivot."""

    estimate: float
    standard_error: float
    z_score: float
    two_sided_p_value: float


def buffered_crossfit_folds(
    length: int, *, block_size: int, dependence_lag: int
) -> tuple[BufferedFold, ...]:
    """Partition a stream into scoring blocks with fold-specific guide buffers.

    For a scoring block ``B``, the guide contains exactly indices whose integer
    distance from every index in ``B`` exceeds ``dependence_lag``. Under an
    ``m``-dependent process with ``m=dependence_lag``, scoring noise is
    independent of the guide sigma-field. Every index is scored once; buffer
    indices are omitted only from that fold's guide and can be scored in their
    own folds.
    """

    for name, value, minimum in (
        ("length", length, 2),
        ("block_size", block_size, 1),
        ("dependence_lag", dependence_lag, 0),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    n = int(length)
    width = int(block_size)
    lag = int(dependence_lag)
    all_indices = np.arange(n, dtype=np.int64)
    folds: list[BufferedFold] = []
    for start in range(0, n, width):
        stop = min(n, start + width)
        scoring = np.arange(start, stop, dtype=np.int64)
        exclusion_start = max(0, start - lag)
        exclusion_stop = min(n, stop + lag)
        exclusion = np.arange(exclusion_start, exclusion_stop, dtype=np.int64)
        buffer_mask = (exclusion < start) | (exclusion >= stop)
        buffer_indices = exclusion[buffer_mask]
        guide_mask = (all_indices < exclusion_start) | (all_indices >= exclusion_stop)
        guide = all_indices[guide_mask]
        folds.append(
            BufferedFold(
                scoring_indices=scoring,
                buffer_indices=buffer_indices,
                guide_indices=guide,
            )
        )
    return tuple(folds)


def gaussian_contrast_pivot(
    values: ArrayLike, contrast: ArrayLike, covariance: ArrayLike
) -> GaussianContrastPivot:
    """Return the exact known-covariance two-sided Gaussian contrast pivot.

    Validity requires the contrast to annihilate the declared scoring-block
    null mean. In buffered cross-fitting the contrast may be guide-selected,
    provided the guide is independent of scoring noise.
    """

    y = np.asarray(values, dtype=float)
    c = np.asarray(contrast, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if y.ndim != 1 or c.shape != y.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(c)):
        raise ValueError("values and contrast must be aligned finite vectors")
    if sigma.shape != (y.size, y.size) or not np.all(np.isfinite(sigma)):
        raise ValueError("covariance must be a finite square matrix aligned with values")
    if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    variance = float(c @ sigma @ c)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("contrast variance must be positive")
    estimate = float(c @ y)
    standard_error = float(np.sqrt(variance))
    z_score = estimate / standard_error
    p_value = 2.0 * NormalDist().cdf(-abs(z_score))
    return GaussianContrastPivot(
        estimate=estimate,
        standard_error=standard_error,
        z_score=float(z_score),
        two_sided_p_value=float(p_value),
    )
