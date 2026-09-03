"""Finite-sample confidence sets for an S-shaped inflection location.

The construction inverts simultaneously valid signs of multiscale chord
contrasts.  It requires convexity to the left of an admissible inflection and
concavity to its right, but it does not estimate derivatives or assume that
the regression function is smooth at the inflection.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class ShapeContrastFamily:
    """A fixed family of sign-valid averaged chord residuals."""

    observation_x: NDArray[np.float64]
    operator: NDArray[np.float64]
    support_left: NDArray[np.float64]
    support_right: NDArray[np.float64]
    block_size: NDArray[np.int64]
    separation: NDArray[np.int64]
    start_index: NDArray[np.int64]
    weight_l2: NDArray[np.float64]

    @property
    def contrast_count(self) -> int:
        return int(self.operator.shape[0])

    def means(self, y: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or values.shape != self.observation_x.shape:
            raise ValueError("y must be one-dimensional and match observation_x")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must be finite")
        return np.asarray(self.operator @ values, dtype=float)


@dataclass(frozen=True)
class ShapeContrastBand:
    """Simultaneous confidence intervals for all contrast expectations."""

    estimate: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    radius: NDArray[np.float64]
    critical_value: float
    alpha: float
    noise_scale: float

    @property
    def certified_signs(self) -> NDArray[np.int8]:
        numerical_scale = max(
            1.0,
            float(np.max(np.abs(self.estimate), initial=0.0)),
            float(np.max(np.abs(self.radius), initial=0.0)),
        )
        tolerance = 64.0 * np.finfo(float).eps * numerical_scale
        signs = np.zeros(self.estimate.size, dtype=np.int8)
        signs[self.lower > tolerance] = 1
        signs[self.upper < -tolerance] = -1
        return signs


@dataclass(frozen=True)
class GaussianShapeContrastCalibration:
    """Finite-simulation upper calibration of a Gaussian contrast maximum."""

    critical_value: float
    total_failure_probability: float
    data_failure_probability: float
    calibration_failure_probability: float
    simulations: int
    order_statistic_rank: int
    seed: int
    family_id: str


@dataclass(frozen=True)
class InflectionConfidenceSet:
    """Interval obtained by inverting certified shape contradictions."""

    left: float
    right: float
    empty: bool
    positive_contrast_count: int
    negative_contrast_count: int
    active_left_contrast: int | None
    active_right_contrast: int | None

    @property
    def interval(self) -> tuple[float, float] | None:
        if self.empty:
            return None
        return (self.left, self.right)

    @property
    def width(self) -> float:
        return float("nan") if self.empty else self.right - self.left

    def project(self, candidate: float) -> float | None:
        value = float(candidate)
        if not np.isfinite(value):
            raise ValueError("candidate must be finite")
        if self.empty:
            return None
        return float(np.clip(value, self.left, self.right))


def dyadic_block_sizes(observation_count: int) -> tuple[int, ...]:
    """Return ``1,2,4,...`` while a three-block contrast still fits."""

    if observation_count < 3:
        raise ValueError("at least three observations are required")
    sizes: list[int] = []
    block_size = 1
    while 3 * block_size <= observation_count:
        sizes.append(block_size)
        block_size *= 2
    return tuple(sizes)


def build_shape_contrast_family(
    x: ArrayLike,
    *,
    block_sizes: tuple[int, ...] | None = None,
    separation_multipliers: tuple[int, ...] = (1,),
    stride_multiplier: int = 1,
) -> ShapeContrastFamily:
    """Build averaged chord residuals over predeclared three-block windows.

    For a window starting at ``a``, block size ``q`` and separation ``s``, the
    observations ``a+r``, ``a+s+r`` and ``a+2s+r`` form one ordered triple.
    Separations are fixed multiples of ``q``.  Each chord residual is
    nonnegative for every convex mean function and nonpositive for every
    concave mean function.  Averaging over ``r`` retains those signs and
    reduces noise.
    """

    locations = np.asarray(x, dtype=float)
    if locations.ndim != 1 or locations.size < 3:
        raise ValueError("x must be one-dimensional with at least three points")
    if not np.all(np.isfinite(locations)) or np.any(np.diff(locations) <= 0.0):
        raise ValueError("x must be finite and strictly increasing")
    if not isinstance(stride_multiplier, int) or stride_multiplier < 1:
        raise ValueError("stride_multiplier must be a positive integer")
    if (
        not separation_multipliers
        or any(
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, np.integer))
            or multiplier < 1
            for multiplier in separation_multipliers
        )
        or len(set(int(multiplier) for multiplier in separation_multipliers))
        != len(separation_multipliers)
    ):
        raise ValueError("separation multipliers must be distinct positive integers")

    sizes = dyadic_block_sizes(locations.size) if block_sizes is None else block_sizes
    if not sizes:
        raise ValueError("at least one block size is required")
    if any(
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size < 1
        or 3 * size > locations.size
        for size in sizes
    ):
        raise ValueError("every block size must be positive with 3q <= len(x)")
    if len(set(int(size) for size in sizes)) != len(sizes):
        raise ValueError("block sizes must be distinct")

    rows: list[NDArray[np.float64]] = []
    support_left: list[float] = []
    support_right: list[float] = []
    row_sizes: list[int] = []
    separations: list[int] = []
    starts: list[int] = []
    n = locations.size
    for size_raw in sorted(int(size) for size in sizes):
        size = int(size_raw)
        for multiplier_raw in sorted(int(value) for value in separation_multipliers):
            separation = multiplier_raw * size
            support_size = 2 * separation + size
            if support_size > n:
                continue
            stride = stride_multiplier * size
            last_start = n - support_size
            scale_starts = list(range(0, last_start + 1, stride))
            if scale_starts[-1] != last_start:
                scale_starts.append(last_start)
            for start in scale_starts:
                row = np.zeros(n, dtype=float)
                for offset in range(size):
                    left_index = start + offset
                    middle_index = start + separation + offset
                    right_index = start + 2 * separation + offset
                    left = locations[left_index]
                    middle = locations[middle_index]
                    right = locations[right_index]
                    left_weight = (right - middle) / (right - left)
                    right_weight = (middle - left) / (right - left)
                    row[left_index] += left_weight / size
                    row[middle_index] -= 1.0 / size
                    row[right_index] += right_weight / size
                rows.append(row)
                support_left.append(float(locations[start]))
                support_right.append(float(locations[start + support_size - 1]))
                row_sizes.append(size)
                separations.append(separation)
                starts.append(start)

    operator = np.vstack(rows)
    return ShapeContrastFamily(
        observation_x=locations.copy(),
        operator=operator,
        support_left=np.asarray(support_left, dtype=float),
        support_right=np.asarray(support_right, dtype=float),
        block_size=np.asarray(row_sizes, dtype=np.int64),
        separation=np.asarray(separations, dtype=np.int64),
        start_index=np.asarray(starts, dtype=np.int64),
        weight_l2=np.linalg.norm(operator, axis=1),
    )


def gaussian_bonferroni_shape_band(
    family: ShapeContrastFamily,
    y: ArrayLike,
    *,
    noise_scale: float,
    alpha: float,
) -> ShapeContrastBand:
    """Return a finite-sample simultaneous Gaussian contrast band."""

    sigma = float(noise_scale)
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError("noise_scale must be nonnegative and finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    count = family.contrast_count
    critical = float(norm.ppf(1.0 - alpha / (2.0 * count)))
    estimate = family.means(y)
    radius = critical * sigma * family.weight_l2
    return ShapeContrastBand(
        estimate=estimate,
        lower=estimate - radius,
        upper=estimate + radius,
        radius=radius,
        critical_value=critical,
        alpha=float(alpha),
        noise_scale=sigma,
    )


def calibrate_gaussian_shape_contrast_max(
    family: ShapeContrastFamily,
    *,
    alpha: float,
    calibration_failure_probability: float,
    simulations: int,
    seed: int,
    chunk_size: int = 256,
) -> GaussianShapeContrastCalibration:
    """Calibrate the joint standardized contrast maximum honestly.

    An upper order statistic exceeds the population
    ``1-(alpha-calibration_failure_probability)`` quantile except on the
    allocated calibration-failure event.  Sparse multiplication exploits the
    three-block construction without changing the calibrated distribution.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    eta = float(calibration_failure_probability)
    if not 0.0 < eta < alpha:
        raise ValueError("calibration failure must lie in (0, alpha)")
    if simulations < 20:
        raise ValueError("at least 20 simulations are required")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if np.any(family.weight_l2 <= 0.0):
        raise ValueError("every contrast must have positive norm")

    from scipy.stats import binom

    data_failure = float(alpha - eta)
    target_probability = 1.0 - data_failure
    rank = int(binom.ppf(1.0 - eta, simulations, target_probability)) + 1
    if rank > simulations:
        raise ValueError("too few simulations for calibration confidence")

    standardized = csr_matrix(
        family.operator / family.weight_l2[:, None]
    )
    rng = np.random.default_rng(seed)
    maxima = np.empty(simulations, dtype=float)
    observation_count = family.observation_x.size
    for start in range(0, simulations, chunk_size):
        stop = min(start + chunk_size, simulations)
        gaussian = rng.normal(0.0, 1.0, size=(stop - start, observation_count))
        transformed = standardized @ gaussian.T
        maxima[start:stop] = np.asarray(np.max(np.abs(transformed), axis=0)).ravel()
    critical = float(np.partition(maxima, rank - 1)[rank - 1])

    fingerprint = sha256()
    fingerprint.update(np.asarray(family.observation_x, dtype=np.float64).tobytes())
    fingerprint.update(np.asarray(family.operator, dtype=np.float64).tobytes())
    fingerprint.update(repr((alpha, eta, simulations, seed)).encode("ascii"))
    return GaussianShapeContrastCalibration(
        critical_value=critical,
        total_failure_probability=float(alpha),
        data_failure_probability=data_failure,
        calibration_failure_probability=eta,
        simulations=int(simulations),
        order_statistic_rank=rank,
        seed=int(seed),
        family_id=fingerprint.hexdigest(),
    )


def gaussian_calibrated_shape_band(
    family: ShapeContrastFamily,
    y: ArrayLike,
    *,
    noise_scale: float,
    calibration: GaussianShapeContrastCalibration,
) -> ShapeContrastBand:
    """Apply a fixed joint-Gaussian calibration to one response."""

    sigma = float(noise_scale)
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError("noise_scale must be nonnegative and finite")
    estimate = family.means(y)
    radius = calibration.critical_value * sigma * family.weight_l2
    return ShapeContrastBand(
        estimate=estimate,
        lower=estimate - radius,
        upper=estimate + radius,
        radius=radius,
        critical_value=calibration.critical_value,
        alpha=calibration.total_failure_probability,
        noise_scale=sigma,
    )


def invert_s_shaped_inflection(
    family: ShapeContrastFamily,
    band: ShapeContrastBand,
    *,
    domain: tuple[float, float],
) -> InflectionConfidenceSet:
    """Invert certified convex/concave contrast signs into an interval.

    A positive contrast on ``[a,b]`` excludes every candidate inflection at or
    left of ``a``.  A negative contrast excludes every candidate at or right
    of ``b``.  On simultaneous coverage, all admissible S-shaped inflections
    therefore remain in the returned interval.
    """

    domain_left, domain_right = map(float, domain)
    if (
        not np.isfinite(domain_left)
        or not np.isfinite(domain_right)
        or domain_left >= domain_right
    ):
        raise ValueError("domain must have finite increasing endpoints")
    if family.support_left.min() < domain_left or family.support_right.max() > domain_right:
        raise ValueError("contrast supports must lie inside the domain")
    if band.estimate.shape != (family.contrast_count,):
        raise ValueError("band and family sizes do not match")

    signs = band.certified_signs
    positive = np.flatnonzero(signs == 1)
    negative = np.flatnonzero(signs == -1)

    if positive.size:
        active_left = int(positive[np.argmax(family.support_left[positive])])
        left = float(family.support_left[active_left])
    else:
        active_left = None
        left = domain_left
    if negative.size:
        active_right = int(negative[np.argmin(family.support_right[negative])])
        right = float(family.support_right[active_right])
    else:
        active_right = None
        right = domain_right

    return InflectionConfidenceSet(
        left=left,
        right=right,
        empty=bool(left > right),
        positive_contrast_count=int(positive.size),
        negative_contrast_count=int(negative.size),
        active_left_contrast=active_left,
        active_right_contrast=active_right,
    )
