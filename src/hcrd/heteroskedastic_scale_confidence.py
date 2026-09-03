"""Honest Gaussian noise envelopes under bounded heteroskedasticity.

The construction uses residual energy from a fixed partition into blocks of
size at least two.  No regularity of the regression mean is required: lack of
fit only shifts the residual Gaussian vector and therefore makes a small
residual norm less likely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GaussianHeteroskedasticEnvelope:
    """A simultaneous upper envelope for all observation noise scales."""

    upper_scale: float
    residual_sum_squares: float
    block_count: int
    residual_degrees_of_freedom: int
    max_to_mean_variance_ratio: float
    failure_probability: float
    concentration_denominator: float
    block_labels: NDArray[np.int64]


def balanced_residual_block_labels(observation_count: int) -> NDArray[np.int64]:
    """Partition observations into consecutive blocks of size two or three."""

    if isinstance(observation_count, bool) or not isinstance(
        observation_count, (int, np.integer)
    ):
        raise ValueError("observation_count must be an integer")
    n = int(observation_count)
    if n < 2:
        raise ValueError("at least two observations are required")
    block_count = n // 2
    labels = np.floor(np.arange(n, dtype=float) * block_count / n).astype(np.int64)
    sizes = np.bincount(labels, minlength=block_count)
    if np.any(sizes < 2) or np.any(sizes > 3):
        raise RuntimeError("internal block construction failed")
    return labels


def gaussian_heteroskedastic_upper_envelope(
    y: ArrayLike,
    *,
    max_to_mean_variance_ratio: float,
    failure_probability: float,
) -> GaussianHeteroskedasticEnvelope:
    """Bound every Gaussian noise standard deviation with finite-sample validity.

    Let independent errors have variances ``sigma_i**2`` and suppose

    ``max_i sigma_i**2 <= kappa * mean_i sigma_i**2``.

    The returned value is at least ``max_i sigma_i`` except on an event of
    probability at most ``failure_probability``.  The mean vector is arbitrary.
    Validity requires the supplied ``kappa`` to be a genuine upper bound.
    """

    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("y must be a finite one-dimensional vector")
    kappa = float(max_to_mean_variance_ratio)
    if not np.isfinite(kappa) or kappa < 1.0:
        raise ValueError("max_to_mean_variance_ratio must be finite and at least one")
    eta = float(failure_probability)
    if not 0.0 < eta < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")

    n = values.size
    denominator = 1.0 - 2.0 * np.sqrt(2.0 * kappa * np.log(1.0 / eta) / n)
    if denominator <= 0.0:
        raise ValueError(
            "sample size is too small for this heterogeneity ratio and failure probability"
        )

    labels = balanced_residual_block_labels(n)
    block_count = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=block_count)
    sums = np.bincount(labels, weights=values, minlength=block_count)
    fitted = (sums / counts)[labels]
    residual = values - fitted
    rss = float(residual @ residual)
    variance_envelope = 2.0 * kappa * max(rss, 0.0) / (n * denominator)
    return GaussianHeteroskedasticEnvelope(
        upper_scale=float(np.sqrt(variance_envelope)),
        residual_sum_squares=rss,
        block_count=block_count,
        residual_degrees_of_freedom=int(n - block_count),
        max_to_mean_variance_ratio=kappa,
        failure_probability=eta,
        concentration_denominator=float(denominator),
        block_labels=labels,
    )
