"""Finite-sample shape-contrast bands from independent replicate curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t

from .inference import ShapeContrastFamily


@dataclass(frozen=True)
class ReplicatedShapeContrastBand:
    """A simultaneous Student-t band for replicate-curve contrasts."""

    estimate: NDArray[np.float64]
    radius: NDArray[np.float64]
    critical_value: float
    alpha: float
    replicate_count: int
    degrees_of_freedom: int

    @property
    def lower(self) -> NDArray[np.float64]:
        return self.estimate - self.radius

    @property
    def upper(self) -> NDArray[np.float64]:
        return self.estimate + self.radius

    @property
    def certified_signs(self) -> NDArray[np.int8]:
        numerical_scale = max(
            1.0,
            float(np.max(np.abs(self.estimate), initial=0.0)),
            float(np.max(np.abs(self.radius), initial=0.0)),
        )
        tolerance = 64.0 * np.finfo(float).eps * numerical_scale
        signs = np.zeros(self.estimate.size, dtype=np.int8)
        signs[self.estimate - self.radius > tolerance] = 1
        signs[self.estimate + self.radius < -tolerance] = -1
        return signs


def replicated_t_shape_band(
    family: ShapeContrastFamily,
    curves: ArrayLike,
    *,
    alpha: float,
) -> ReplicatedShapeContrastBand:
    """Return a finite-sample band from independent Gaussian replicates.

    Rows of ``curves`` are independent replicate curves observed on the same
    design. Dependence and unequal variance across design points are allowed.
    The replicate curves must share one mean vector and one covariance matrix.
    Each fixed contrast then has an exact univariate Student statistic.
    Bonferroni correction gives simultaneous coverage without estimating the
    full covariance matrix.
    """

    values = np.asarray(curves, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != family.observation_x.size
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "curves must be a finite matrix with one replicate curve per row"
        )
    replicate_count = int(values.shape[0])
    if replicate_count < 2:
        raise ValueError("at least two replicate curves are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    scores = family.means_many(values)
    estimate = np.mean(scores, axis=0)
    standard_error = np.std(scores, axis=0, ddof=1) / np.sqrt(replicate_count)
    degrees_of_freedom = replicate_count - 1
    critical = float(
        t.ppf(1.0 - alpha / (2.0 * family.contrast_count), degrees_of_freedom)
    )
    return ReplicatedShapeContrastBand(
        estimate=estimate,
        radius=critical * standard_error,
        critical_value=critical,
        alpha=float(alpha),
        replicate_count=replicate_count,
        degrees_of_freedom=degrees_of_freedom,
    )
