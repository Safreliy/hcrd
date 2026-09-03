"""Memory-efficient confidence sets for convex-to-concave transitions.

The implementation stores only the design and the start indices of each
contrast scale. It never builds a dense contrast-by-observation matrix.
Uniform designs use prefix sums; irregular designs evaluate weights in
bounded-memory chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom, norm


@dataclass(frozen=True)
class ContrastScale:
    """Compact description of one block-size and separation pair."""

    block_size: int
    separation: int
    start_index: NDArray[np.int64]
    weight_l2: float | NDArray[np.float64]

    @property
    def support_size(self) -> int:
        return 2 * self.separation + self.block_size

    @property
    def contrast_count(self) -> int:
        return int(self.start_index.size)


@dataclass(frozen=True)
class ShapeContrastFamily:
    """A compact family of sign-valid averaged chord contrasts."""

    observation_x: NDArray[np.float64]
    scales: tuple[ContrastScale, ...]
    uniform_design: bool

    @property
    def contrast_count(self) -> int:
        return sum(scale.contrast_count for scale in self.scales)

    @property
    def stored_bytes(self) -> int:
        """Bytes held by the compact numerical arrays."""

        total = int(self.observation_x.nbytes)
        for scale in self.scales:
            total += int(scale.start_index.nbytes)
            if isinstance(scale.weight_l2, np.ndarray):
                total += int(scale.weight_l2.nbytes)
        return total

    @property
    def start_index(self) -> NDArray[np.int64]:
        return np.concatenate([scale.start_index for scale in self.scales])

    @property
    def block_size(self) -> NDArray[np.int64]:
        return np.concatenate(
            [
                np.full(scale.contrast_count, scale.block_size, dtype=np.int64)
                for scale in self.scales
            ]
        )

    @property
    def separation(self) -> NDArray[np.int64]:
        return np.concatenate(
            [
                np.full(scale.contrast_count, scale.separation, dtype=np.int64)
                for scale in self.scales
            ]
        )

    @property
    def support_left(self) -> NDArray[np.float64]:
        return np.concatenate(
            [self.observation_x[scale.start_index] for scale in self.scales]
        )

    @property
    def support_right(self) -> NDArray[np.float64]:
        return np.concatenate(
            [
                self.observation_x[
                    scale.start_index + scale.support_size - 1
                ]
                for scale in self.scales
            ]
        )

    @property
    def weight_l2(self) -> NDArray[np.float64]:
        chunks: list[NDArray[np.float64]] = []
        for scale in self.scales:
            if isinstance(scale.weight_l2, np.ndarray):
                chunks.append(scale.weight_l2)
            else:
                chunks.append(
                    np.full(scale.contrast_count, scale.weight_l2, dtype=float)
                )
        return np.concatenate(chunks)

    def means(self, y: ArrayLike) -> NDArray[np.float64]:
        """Evaluate every contrast with bounded intermediate memory."""

        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or values.shape != self.observation_x.shape:
            raise ValueError("y must be one-dimensional and match observation_x")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must be finite")
        return self._apply_many(values[None, :], squared_weights=False)[0]

    def means_many(self, y: ArrayLike) -> NDArray[np.float64]:
        """Evaluate a batch whose rows are independent response curves."""

        values = np.asarray(y, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != self.observation_x.size
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "y must be a finite matrix with one response curve per row"
            )
        return self._apply_many(values, squared_weights=False)

    def contrast_variances(
        self, point_variances: ArrayLike
    ) -> NDArray[np.float64]:
        """Return exact contrast variances for independent observations."""

        variances = np.asarray(point_variances, dtype=float)
        if variances.ndim != 1 or variances.shape != self.observation_x.shape:
            raise ValueError(
                "point_variances must be one-dimensional and match observation_x"
            )
        if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
            raise ValueError("point_variances must be finite and nonnegative")
        return self._apply_many(variances[None, :], squared_weights=True)[0]

    def _apply_many(
        self,
        values: NDArray[np.float64],
        *,
        squared_weights: bool,
    ) -> NDArray[np.float64]:
        batch, observation_count = values.shape
        if observation_count != self.observation_x.size:
            raise ValueError("values do not match the contrast family")
        output = np.empty((batch, self.contrast_count), dtype=float)
        prefix = None
        if self.uniform_design:
            prefix = np.empty((batch, observation_count + 1), dtype=float)
            prefix[:, 0] = 0.0
            np.cumsum(values, axis=1, out=prefix[:, 1:])

        cursor = 0
        for scale in self.scales:
            stop = cursor + scale.contrast_count
            if self.uniform_design:
                assert prefix is not None
                output[:, cursor:stop] = _uniform_scale_values(
                    prefix, scale, squared_weights=squared_weights
                )
            else:
                output[:, cursor:stop] = _irregular_scale_values(
                    self.observation_x,
                    values,
                    scale,
                    squared_weights=squared_weights,
                )
            cursor = stop
        return output

    def fingerprint(self) -> str:
        """Return a deterministic identifier for this contrast family."""

        digest = sha256()
        digest.update(self.observation_x.astype(np.float64, copy=False).tobytes())
        digest.update(bytes([self.uniform_design]))
        for scale in self.scales:
            digest.update(
                np.asarray(
                    [scale.block_size, scale.separation], dtype=np.int64
                ).tobytes()
            )
            digest.update(scale.start_index.astype(np.int64, copy=False).tobytes())
            digest.update(
                np.asarray(scale.weight_l2, dtype=np.float64).tobytes()
            )
        return digest.hexdigest()


@dataclass(frozen=True)
class ShapeContrastBand:
    """A simultaneous confidence band for the contrast expectations."""

    estimate: NDArray[np.float64]
    radius: NDArray[np.float64]
    critical_value: float
    alpha: float
    noise_scale: float

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


@dataclass(frozen=True)
class GaussianShapeContrastCalibration:
    """A finite-simulation upper calibration of the Gaussian maximum."""

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
    """An outer confidence interval for all compatible transitions."""

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
    """Return ``1, 2, 4, ...`` while a three-block contrast still fits."""

    if isinstance(observation_count, bool) or not isinstance(
        observation_count, (int, np.integer)
    ):
        raise ValueError("observation_count must be an integer")
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
    """Build a compact family of averaged three-block chord contrasts."""

    locations = np.asarray(x, dtype=float)
    if locations.ndim != 1 or locations.size < 3:
        raise ValueError("x must be one-dimensional with at least three points")
    differences = np.diff(locations)
    if not np.all(np.isfinite(locations)) or np.any(differences <= 0.0):
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
        or len(set(int(value) for value in separation_multipliers))
        != len(separation_multipliers)
    ):
        raise ValueError("separation multipliers must be distinct positive integers")

    sizes = dyadic_block_sizes(locations.size) if block_sizes is None else block_sizes
    if (
        not sizes
        or any(
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or size < 1
            or 3 * size > locations.size
            for size in sizes
        )
        or len(set(int(size) for size in sizes)) != len(sizes)
    ):
        raise ValueError("block sizes must be distinct and satisfy 3q <= len(x)")

    uniform = bool(
        np.allclose(
            differences,
            differences[0],
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0 * np.finfo(float).eps * max(1.0, abs(differences[0])),
        )
    )
    scales: list[ContrastScale] = []
    n = locations.size
    for size in sorted(int(value) for value in sizes):
        for multiplier in sorted(int(value) for value in separation_multipliers):
            separation = multiplier * size
            support_size = 2 * separation + size
            if support_size > n:
                continue
            last_start = n - support_size
            starts = np.arange(
                0, last_start + 1, stride_multiplier * size, dtype=np.int64
            )
            if starts[-1] != last_start:
                starts = np.append(starts, np.int64(last_start))
            if uniform:
                norms: float | NDArray[np.float64] = float(
                    np.sqrt(1.5 / size)
                )
            else:
                norms = _irregular_weight_norms(
                    locations, starts, size, separation
                )
            scales.append(
                ContrastScale(
                    block_size=size,
                    separation=separation,
                    start_index=starts,
                    weight_l2=norms,
                )
            )
    if not scales:
        raise ValueError("no requested contrast scale fits the design")
    return ShapeContrastFamily(
        observation_x=locations.copy(),
        scales=tuple(scales),
        uniform_design=uniform,
    )


def gaussian_bonferroni_shape_band(
    family: ShapeContrastFamily,
    y: ArrayLike,
    *,
    noise_scale: float,
    alpha: float,
) -> ShapeContrastBand:
    """Return an exact finite-sample Bonferroni band under Gaussian noise."""

    sigma = float(noise_scale)
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError("noise_scale must be nonnegative and finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    critical = float(norm.ppf(1.0 - alpha / (2.0 * family.contrast_count)))
    estimate = family.means(y)
    radius = critical * sigma * family.weight_l2
    return ShapeContrastBand(
        estimate=estimate,
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
    """Calibrate the standardized Gaussian maximum without a dense operator."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    eta = float(calibration_failure_probability)
    if not 0.0 < eta < alpha:
        raise ValueError("calibration failure must lie in (0, alpha)")
    if simulations < 20:
        raise ValueError("at least 20 simulations are required")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    data_failure = float(alpha - eta)
    target_probability = 1.0 - data_failure
    rank = int(binom.ppf(1.0 - eta, simulations, target_probability)) + 1
    if rank > simulations:
        raise ValueError("too few simulations for calibration confidence")

    rng = np.random.default_rng(seed)
    maxima = np.empty(simulations, dtype=float)
    norms = family.weight_l2
    n = family.observation_x.size
    for start in range(0, simulations, chunk_size):
        stop = min(start + chunk_size, simulations)
        gaussian = rng.normal(0.0, 1.0, size=(stop - start, n))
        transformed = family._apply_many(gaussian, squared_weights=False)
        maxima[start:stop] = np.max(np.abs(transformed / norms), axis=1)
    critical = float(np.partition(maxima, rank - 1)[rank - 1])

    fingerprint = sha256()
    fingerprint.update(family.fingerprint().encode("ascii"))
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
    """Apply a fixed finite-simulation Gaussian calibration."""

    sigma = float(noise_scale)
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError("noise_scale must be nonnegative and finite")
    estimate = family.means(y)
    radius = calibration.critical_value * sigma * family.weight_l2
    return ShapeContrastBand(
        estimate=estimate,
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
    """Invert certified signs into an outer transition confidence set."""

    domain_left, domain_right = map(float, domain)
    if (
        not np.isfinite(domain_left)
        or not np.isfinite(domain_right)
        or domain_left >= domain_right
    ):
        raise ValueError("domain must have finite increasing endpoints")
    if (
        family.observation_x[0] < domain_left
        or family.observation_x[-1] > domain_right
    ):
        raise ValueError("contrast supports must lie inside the domain")
    if band.estimate.shape != (family.contrast_count,) or band.radius.shape != (
        family.contrast_count,
    ):
        raise ValueError("band and family sizes do not match")

    signs = band.certified_signs
    left = domain_left
    right = domain_right
    active_left: int | None = None
    active_right: int | None = None
    positive_count = 0
    negative_count = 0
    cursor = 0
    for scale in family.scales:
        local = signs[cursor : cursor + scale.contrast_count]
        positive = np.flatnonzero(local == 1)
        negative = np.flatnonzero(local == -1)
        positive_count += int(positive.size)
        negative_count += int(negative.size)
        if positive.size:
            supports = family.observation_x[scale.start_index[positive]]
            local_best = int(np.argmax(supports))
            candidate = float(supports[local_best])
            if active_left is None or candidate > left:
                left = candidate
                active_left = cursor + int(positive[local_best])
        if negative.size:
            supports = family.observation_x[
                scale.start_index[negative] + scale.support_size - 1
            ]
            local_best = int(np.argmin(supports))
            candidate = float(supports[local_best])
            if active_right is None or candidate < right:
                right = candidate
                active_right = cursor + int(negative[local_best])
        cursor += scale.contrast_count

    return InflectionConfidenceSet(
        left=left,
        right=right,
        empty=bool(left > right),
        positive_contrast_count=positive_count,
        negative_contrast_count=negative_count,
        active_left_contrast=active_left,
        active_right_contrast=active_right,
    )


def _uniform_scale_values(
    prefix: NDArray[np.float64],
    scale: ContrastScale,
    *,
    squared_weights: bool,
) -> NDArray[np.float64]:
    starts = scale.start_index
    size = scale.block_size
    separation = scale.separation

    def block_sum(offset: int) -> NDArray[np.float64]:
        left = starts + offset
        return prefix[:, left + size] - prefix[:, left]

    left_sum = block_sum(0)
    middle_sum = block_sum(separation)
    right_sum = block_sum(2 * separation)
    if squared_weights:
        return (0.25 * left_sum + middle_sum + 0.25 * right_sum) / size**2
    return (0.5 * left_sum - middle_sum + 0.5 * right_sum) / size


def _irregular_scale_values(
    x: NDArray[np.float64],
    values: NDArray[np.float64],
    scale: ContrastScale,
    *,
    squared_weights: bool,
) -> NDArray[np.float64]:
    batch = values.shape[0]
    size = scale.block_size
    separation = scale.separation
    result = np.empty((batch, scale.contrast_count), dtype=float)
    max_index_elements = 1_000_000
    rows_per_chunk = max(
        1, min(8192, max_index_elements // (size * max(1, batch)))
    )
    offsets = np.arange(size, dtype=np.int64)
    for first in range(0, scale.contrast_count, rows_per_chunk):
        last = min(first + rows_per_chunk, scale.contrast_count)
        left_index = scale.start_index[first:last, None] + offsets
        middle_index = left_index + separation
        right_index = left_index + 2 * separation
        x_left = x[left_index]
        x_middle = x[middle_index]
        x_right = x[right_index]
        left_weight = (x_right - x_middle) / (x_right - x_left) / size
        middle_weight = np.full_like(left_weight, -1.0 / size)
        right_weight = (x_middle - x_left) / (x_right - x_left) / size
        if squared_weights:
            left_weight = left_weight**2
            middle_weight = middle_weight**2
            right_weight = right_weight**2
        result[:, first:last] = np.sum(
            values[:, left_index] * left_weight
            + values[:, middle_index] * middle_weight
            + values[:, right_index] * right_weight,
            axis=2,
        )
    return result


def _irregular_weight_norms(
    x: NDArray[np.float64],
    starts: NDArray[np.int64],
    size: int,
    separation: int,
) -> NDArray[np.float64]:
    result = np.empty(starts.size, dtype=float)
    max_index_elements = 1_000_000
    rows_per_chunk = max(1, min(65536, max_index_elements // size))
    offsets = np.arange(size, dtype=np.int64)
    for first in range(0, starts.size, rows_per_chunk):
        last = min(first + rows_per_chunk, starts.size)
        left_index = starts[first:last, None] + offsets
        middle_index = left_index + separation
        right_index = left_index + 2 * separation
        x_left = x[left_index]
        x_middle = x[middle_index]
        x_right = x[right_index]
        left_weight = (x_right - x_middle) / (x_right - x_left) / size
        right_weight = (x_middle - x_left) / (x_right - x_left) / size
        result[first:last] = np.sqrt(
            np.sum(
                left_weight**2 + (1.0 / size) ** 2 + right_weight**2,
                axis=1,
            )
        )
    return result
