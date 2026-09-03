"""Finite-sample upper confidence bounds for Gaussian regression noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2


@dataclass(frozen=True)
class GaussianProjectionScaleBound:
    """A one-sided noise-scale bound from a fixed projection residual."""

    upper_scale: float
    residual_sum_squares: float
    residual_degrees_of_freedom: int
    nuisance_rank: int
    lower_chi_square_quantile: float
    failure_probability: float


def consecutive_block_design(
    observation_count: int, block_count: int
) -> NDArray[np.float64]:
    """Return fixed equal-index block indicators with full column rank."""

    if isinstance(observation_count, bool) or not isinstance(
        observation_count, (int, np.integer)
    ):
        raise ValueError("observation_count must be an integer")
    if isinstance(block_count, bool) or not isinstance(block_count, (int, np.integer)):
        raise ValueError("block_count must be an integer")
    n = int(observation_count)
    blocks = int(block_count)
    if n < 2 or not 1 <= blocks < n:
        raise ValueError("require 1 <= block_count < observation_count")
    labels = np.floor(np.arange(n, dtype=float) * blocks / n).astype(int)
    design = np.zeros((n, blocks), dtype=float)
    design[np.arange(n), labels] = 1.0
    return design


def gaussian_block_upper_scale(
    y: ArrayLike,
    block_count: int,
    *,
    failure_probability: float,
) -> GaussianProjectionScaleBound:
    """Fast projection-scale bound for consecutive block indicators."""

    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite one-dimensional vector")
    if isinstance(block_count, bool) or not isinstance(block_count, (int, np.integer)):
        raise ValueError("block_count must be an integer")
    blocks = int(block_count)
    if not 1 <= blocks < values.size:
        raise ValueError("require 1 <= block_count < len(y)")
    labels = np.floor(np.arange(values.size, dtype=float) * blocks / values.size).astype(int)
    counts = np.bincount(labels, minlength=blocks)
    sums = np.bincount(labels, weights=values, minlength=blocks)
    fitted = (sums / counts)[labels]
    residual = values - fitted
    rss = float(residual @ residual)
    degrees = int(values.size - blocks)
    eta = float(failure_probability)
    if not 0.0 < eta < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    lower_quantile = float(chi2.ppf(eta, degrees))
    if not np.isfinite(lower_quantile) or lower_quantile <= 0.0:
        raise RuntimeError("lower chi-square quantile is not numerically usable")
    return GaussianProjectionScaleBound(
        upper_scale=float(np.sqrt(max(rss, 0.0) / lower_quantile)),
        residual_sum_squares=rss,
        residual_degrees_of_freedom=degrees,
        nuisance_rank=blocks,
        lower_chi_square_quantile=lower_quantile,
        failure_probability=eta,
    )


def gaussian_projection_upper_scale(
    y: ArrayLike,
    nuisance_design: ArrayLike,
    *,
    failure_probability: float,
) -> GaussianProjectionScaleBound:
    """Return an honest upper bound for iid Gaussian noise scale.

    The column space must be fixed independently of ``y``.  If ``R`` is its
    orthogonal residual projector, then ``||R y||^2 / sigma^2`` is noncentral
    chi-square.  Its lower tail is largest at zero noncentrality, which makes
    the central lower quantile valid for every unknown regression mean.
    """

    values = np.asarray(y, dtype=float)
    design = np.asarray(nuisance_design, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite one-dimensional vector")
    if (
        design.ndim != 2
        or design.shape[0] != values.size
        or design.shape[1] < 1
        or not np.all(np.isfinite(design))
    ):
        raise ValueError("nuisance_design must be a finite matrix matching y")
    eta = float(failure_probability)
    if not 0.0 < eta < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")

    rank = int(np.linalg.matrix_rank(design))
    degrees = int(values.size - rank)
    if degrees < 1:
        raise ValueError("nuisance design must leave positive residual degrees of freedom")
    fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - fitted
    rss = float(residual @ residual)
    lower_quantile = float(chi2.ppf(eta, degrees))
    if not np.isfinite(lower_quantile) or lower_quantile <= 0.0:
        raise RuntimeError("lower chi-square quantile is not numerically usable")
    upper = float(np.sqrt(max(rss, 0.0) / lower_quantile))
    return GaussianProjectionScaleBound(
        upper_scale=upper,
        residual_sum_squares=rss,
        residual_degrees_of_freedom=degrees,
        nuisance_rank=rank,
        lower_chi_square_quantile=lower_quantile,
        failure_probability=eta,
    )
