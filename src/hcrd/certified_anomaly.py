"""Noise-certified common-scale HCRD transient score.

Unlike row-wise robust fusion, this companion keeps every hierarchy level in
the input's physical amplitude units. Its conservative finite-sample
certificate uses a supplied Gaussian-noise standard-deviation upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import decompose_sparse, discrete_curvature


@dataclass(frozen=True)
class CertifiedAreaScore:
    """Score and deterministic radii used by its certificate."""

    score: NDArray[np.float64]
    densities: NDArray[np.float64]
    input_radius: float
    curvature_tolerance: float
    curvature_multiplier: float
    confidence_delta: float


@dataclass(frozen=True)
class HierarchyMarginCertificate:
    """Sufficient decision margins for noisy/noiseless hierarchy agreement."""

    certified: bool
    input_radius: float
    curvature_perturbation_radius: float
    curvature_tolerance: float
    threshold_slack_min: float
    transition_comparison_slack_min: float
    levels_checked: int
    knot_sets: tuple[NDArray[np.int64], ...]


@dataclass(frozen=True)
class HierarchyDecisionRadius:
    """Largest strict sufficient input ball exposed by the visited decisions."""

    input_radius: float
    curvature_perturbation_radius: float
    curvature_tolerance: float
    threshold_margin_min: float
    transition_comparison_margin_min: float
    levels_checked: int
    knot_sets: tuple[NDArray[np.int64], ...]


def _minimum_curvature_decision_slacks(
    signal: NDArray[np.float64],
    locations: NDArray[np.float64],
    *,
    tolerance: float,
    perturbation_radius: float,
) -> tuple[NDArray[np.int64], list[float], list[float]]:
    """Replay the centred knot walk and expose its sufficient branch slacks."""

    n = signal.size
    if n == 2:
        return np.asarray([0, 1], dtype=np.int64), [], []
    curvature = discrete_curvature(signal, locations)
    knots = [0]
    threshold_slacks: list[float] = []
    comparison_slacks: list[float] = []

    def record(value: float) -> bool:
        inactive = abs(value) <= tolerance
        if inactive:
            threshold_slacks.append(
                tolerance - abs(value) - perturbation_radius
            )
        else:
            threshold_slacks.append(
                abs(value) - tolerance - perturbation_radius
            )
        return inactive

    start = 0
    while start < n - 1:
        first = start + 1
        while first <= n - 2:
            if not record(float(curvature[first - 1])):
                break
            first += 1
        if first > n - 2:
            knots.append(n - 1)
            break

        active_sign = 1 if curvature[first - 1] > 0 else -1
        last_active = first
        cursor = first + 1
        transition: int | None = None
        while cursor <= n - 2:
            value = float(curvature[cursor - 1])
            if record(value):
                cursor += 1
                continue
            sign = 1 if value > 0 else -1
            if sign != active_sign:
                zero_count = cursor - last_active - 1
                if zero_count == 1:
                    transition = last_active + 1
                else:
                    if last_active != first:
                        comparison_slacks.append(
                            abs(
                                abs(float(curvature[last_active - 1]))
                                - abs(float(curvature[cursor - 1]))
                            )
                            - 2.0 * perturbation_radius
                        )
                    transition = (
                        last_active
                        if abs(curvature[last_active - 1])
                        < abs(curvature[cursor - 1])
                        and last_active != first
                        else cursor
                    )
                break
            last_active = cursor
            cursor += 1

        if transition is None:
            knots.append(n - 1)
            break
        transition = max(min(start + 2, n - 1), min(transition, n - 1))
        if transition == knots[-1]:
            transition += 1
        knots.append(transition)
        start = transition
    if knots[-1] != n - 1:
        knots.append(n - 1)
    return np.asarray(knots, dtype=np.int64), threshold_slacks, comparison_slacks


def certify_hcrd_hierarchy_margin(
    reference_signal: ArrayLike,
    *,
    noise_sigma: float,
    confidence_delta: float = 0.05,
    x: ArrayLike | None = None,
    max_levels: int | None = 8,
    curvature_multiplier: float = 1.0,
) -> HierarchyMarginCertificate:
    """Check sufficient margins for hierarchy agreement under bounded noise.

    ``reference_signal`` is the noiseless mean used in a theorem or simulation,
    not an observed noisy plug-in.  If the result is certified and an observed
    replicate equals this reference plus iid Gaussian noise of standard
    deviation at most ``noise_sigma``, its thresholded HCRD knot sets agree
    with the returned reference knot sets with probability at least
    ``1 - confidence_delta``.
    """

    values = np.asarray(reference_signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("reference_signal must be a finite vector of length >= 2")
    if noise_sigma < 0.0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and nonnegative")
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("confidence_delta must lie strictly between zero and one")
    if curvature_multiplier < 1.0 or not np.isfinite(curvature_multiplier):
        raise ValueError("curvature_multiplier must be finite and at least one")
    locations = (
        np.arange(values.size, dtype=float)
        if x is None
        else np.asarray(x, dtype=float)
    )
    if locations.shape != values.shape or not np.all(np.isfinite(locations)):
        raise ValueError("x and reference_signal must be aligned finite vectors")
    spacing = np.diff(locations)
    if np.any(spacing <= 0.0):
        raise ValueError("x must be strictly increasing")

    epsilon = float(
        noise_sigma
        * np.sqrt(2.0 * np.log(2.0 * values.size / confidence_delta))
    )
    eta = float(4.0 * epsilon / np.min(spacing))
    tolerance = float(curvature_multiplier * eta)
    hierarchy = decompose_sparse(
        values,
        locations,
        atol=tolerance,
        rtol=0.0,
        max_levels=max_levels,
    )
    previous = np.arange(values.size, dtype=np.int64)
    threshold_slacks: list[float] = []
    comparison_slacks: list[float] = []
    for level in hierarchy.levels:
        local_knots, level_threshold, level_comparison = (
            _minimum_curvature_decision_slacks(
                values[previous],
                locations[previous],
                tolerance=tolerance,
                perturbation_radius=eta,
            )
        )
        if not np.array_equal(previous[local_knots], level.knots):
            raise RuntimeError("decision-margin trace disagrees with HCRD")
        threshold_slacks.extend(level_threshold)
        comparison_slacks.extend(level_comparison)
        previous = level.knots
    threshold_min = min(threshold_slacks, default=float("inf"))
    comparison_min = min(comparison_slacks, default=float("inf"))
    numerical_tolerance = 64.0 * np.finfo(float).eps * max(1.0, tolerance)
    certified = bool(
        threshold_min >= -numerical_tolerance
        and comparison_min > numerical_tolerance
    )
    return HierarchyMarginCertificate(
        certified=certified,
        input_radius=epsilon,
        curvature_perturbation_radius=eta,
        curvature_tolerance=tolerance,
        threshold_slack_min=float(threshold_min),
        transition_comparison_slack_min=float(comparison_min),
        levels_checked=hierarchy.depth,
        knot_sets=tuple(level.knots.copy() for level in hierarchy.levels),
    )


def hcrd_hierarchy_decision_radius(
    reference_signal: ArrayLike,
    *,
    curvature_tolerance: float,
    x: ArrayLike | None = None,
    max_levels: int | None = 8,
) -> HierarchyDecisionRadius:
    """Expose a deterministic sufficient radius for the complete knot trace.

    The tolerance is fixed independently of perturbations.  If ``R`` is the
    returned positive input radius, every perturbation with sup norm strictly
    below ``R`` preserves all visited knot sets.  A zero radius honestly marks
    a threshold equality or a tied centred transition comparison.
    """

    values = np.asarray(reference_signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("reference_signal must be a finite vector of length >= 2")
    if not np.isfinite(curvature_tolerance) or curvature_tolerance < 0.0:
        raise ValueError("curvature_tolerance must be finite and nonnegative")
    locations = (
        np.arange(values.size, dtype=float)
        if x is None
        else np.asarray(x, dtype=float)
    )
    if locations.shape != values.shape or not np.all(np.isfinite(locations)):
        raise ValueError("x and reference_signal must be aligned finite vectors")
    spacing = np.diff(locations)
    if np.any(spacing <= 0.0):
        raise ValueError("x must be strictly increasing")

    hierarchy = decompose_sparse(
        values,
        locations,
        atol=float(curvature_tolerance),
        rtol=0.0,
        max_levels=max_levels,
    )
    previous = np.arange(values.size, dtype=np.int64)
    threshold_margins: list[float] = []
    comparison_margins: list[float] = []
    for level in hierarchy.levels:
        local_knots, level_threshold, level_comparison = (
            _minimum_curvature_decision_slacks(
                values[previous],
                locations[previous],
                tolerance=float(curvature_tolerance),
                perturbation_radius=0.0,
            )
        )
        if not np.array_equal(previous[local_knots], level.knots):
            raise RuntimeError("decision-margin trace disagrees with HCRD")
        threshold_margins.extend(level_threshold)
        comparison_margins.extend(level_comparison)
        previous = level.knots

    threshold_min = min(threshold_margins, default=float("inf"))
    comparison_min = min(comparison_margins, default=float("inf"))
    curvature_radius = min(threshold_min, comparison_min / 2.0)
    curvature_radius = max(0.0, float(curvature_radius))
    input_radius = curvature_radius * float(np.min(spacing)) / 4.0
    return HierarchyDecisionRadius(
        input_radius=float(input_radius),
        curvature_perturbation_radius=curvature_radius,
        curvature_tolerance=float(curvature_tolerance),
        threshold_margin_min=float(threshold_min),
        transition_comparison_margin_min=float(comparison_min),
        levels_checked=hierarchy.depth,
        knot_sets=tuple(level.knots.copy() for level in hierarchy.levels),
    )


def stochastic_hierarchy_agreement_lower_bound(
    *,
    sample_count: int,
    noise_sigma: float,
    radius: float,
    margin_failure_probability: float = 0.0,
) -> float:
    """Unconditional agreement bound from a latent margin small-ball bound.

    Suppose a random latent signal ``F`` is independent of iid
    ``N(0, noise_sigma**2)`` scoring noise and
    ``P(hierarchy_decision_radius(F) <= radius)`` is at most
    ``margin_failure_probability``.  A Gaussian maximum bound and the
    deterministic decision-radius theorem give the returned lower bound.
    """

    if isinstance(sample_count, bool) or int(sample_count) != sample_count:
        raise ValueError("sample_count must be an integer at least two")
    count = int(sample_count)
    if count < 2:
        raise ValueError("sample_count must be an integer at least two")
    if not np.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("noise_sigma must be finite and nonnegative")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    if not 0.0 <= margin_failure_probability <= 1.0:
        raise ValueError("margin_failure_probability must lie in [0, 1]")
    if noise_sigma == 0.0:
        noise_failure = 0.0 if radius > 0.0 else 1.0
    else:
        noise_failure = min(
            1.0,
            2.0 * count * np.exp(-(radius**2) / (2.0 * noise_sigma**2)),
        )
    return float(
        max(0.0, 1.0 - margin_failure_probability - noise_failure)
    )


def certified_hcrd_area_score(
    signal: ArrayLike,
    *,
    noise_sigma: float,
    confidence_delta: float = 0.05,
    x: ArrayLike | None = None,
    max_levels: int | None = 8,
    curvature_multiplier: float = 1.0,
) -> CertifiedAreaScore:
    """Return a common-scale HCRD score with an affine-null certificate.

    If the samples equal an affine signal plus independent centred Gaussian
    noise with standard deviation at most ``noise_sigma``, then the returned
    score is identically zero with probability at least
    ``1 - confidence_delta``. The guarantee is deliberately conservative and
    does not apply when ``noise_sigma`` is estimated from contaminated data.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite one-dimensional array of length >= 2")
    if noise_sigma < 0.0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and nonnegative")
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("confidence_delta must lie strictly between zero and one")
    if curvature_multiplier < 1.0 or not np.isfinite(curvature_multiplier):
        raise ValueError("curvature_multiplier must be finite and at least one")
    locations = (
        np.arange(values.size, dtype=float)
        if x is None
        else np.asarray(x, dtype=float)
    )
    if locations.shape != values.shape or not np.all(np.isfinite(locations)):
        raise ValueError("x and signal must be aligned finite vectors")
    spacing = np.diff(locations)
    if np.any(spacing <= 0.0):
        raise ValueError("x must be strictly increasing")

    epsilon = float(
        noise_sigma
        * np.sqrt(2.0 * np.log(2.0 * values.size / confidence_delta))
    )
    curvature_tolerance = float(
        curvature_multiplier * 4.0 * epsilon / np.min(spacing)
    )
    hierarchy = decompose_sparse(
        values,
        locations,
        atol=curvature_tolerance,
        rtol=0.0,
        max_levels=max_levels,
    )
    current = hierarchy.original.copy()
    densities = np.empty((hierarchy.depth, values.size), dtype=float)
    for row, level in enumerate(hierarchy.levels):
        baseline = np.interp(
            hierarchy.x,
            hierarchy.x[level.knots],
            hierarchy.original[level.knots],
        )
        densities[row] = np.abs(current - baseline)
        current = baseline
    score = np.maximum(np.max(densities, axis=0) - 2.0 * epsilon, 0.0)
    return CertifiedAreaScore(
        score=score,
        densities=densities,
        input_radius=epsilon,
        curvature_tolerance=curvature_tolerance,
        curvature_multiplier=float(curvature_multiplier),
        confidence_delta=confidence_delta,
    )
