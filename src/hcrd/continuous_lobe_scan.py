"""Entropy and sieve inference for continuous chord-lobe families.

The fixed-dictionary scan in :mod:`hcrd.lobe_scan` is exact for a supplied
finite family.  This module adds two distinct bridges to an uncountable family:

* an epsilon-net (sieve) scan with an explicit approximation loss; and
* a Dudley--Borell threshold for a compact continuous template family with a
  supplied Lipschitz/entropy certificate.

Neither construction permits templates selected from the scoring noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .lobe_scan import residualized_lobe_dictionary, scan_detection_threshold


@dataclass(frozen=True)
class SieveLobeScanResult:
    """Finite epsilon-net scan interpreted over a continuous lobe family."""

    scores: NDArray[np.float64]
    selected_index: int
    selected_parameter: NDArray[np.float64]
    statistic: float
    threshold: float
    rejected: bool
    templates: NDArray[np.float64]
    parameters: NDArray[np.float64]
    canonical_mesh_radius: float
    continuous_signal_factor: float


@dataclass(frozen=True)
class TriangularLobeLipschitzCertificate:
    """Analytic certificate for an interior triangular-lobe parameter box."""

    sample_count: int
    maximum_grid_spacing: float
    minimum_apex_segment: float
    sampled_apex_height_lower: float
    residual_norm_lower: float
    raw_template_lipschitz: float
    normalized_template_lipschitz: float
    parameter_widths: NDArray[np.float64]
    canonical_diameter_upper: float
    entropy_integral_upper: float


def asymmetric_triangular_lobes(
    x: ArrayLike, parameters: ArrayLike
) -> NDArray[np.float64]:
    """Generate unit-height compact chord lobes.

    Each parameter row is ``(support_center, support_width, apex_fraction)``.
    The apex fraction is measured from the left support endpoint.  The raw
    templates are one-sign and vanish at both support endpoints; affine
    residualization and L2 normalization are deliberately left to the scan.
    """

    grid = np.asarray(x, dtype=float)
    theta = np.asarray(parameters, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
        raise ValueError("x must be a finite one-dimensional grid with n >= 3")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2 or theta.shape[1] != 3 or not np.all(np.isfinite(theta)):
        raise ValueError("parameters must have shape (M, 3) and be finite")
    centers, widths, apex_fractions = theta.T
    if np.any(widths <= 0.0):
        raise ValueError("support widths must be positive")
    if np.any((apex_fractions <= 0.0) | (apex_fractions >= 1.0)):
        raise ValueError("apex fractions must lie strictly between zero and one")

    result = np.zeros((theta.shape[0], grid.size), dtype=float)
    for row, (center, width, apex_fraction) in enumerate(theta):
        left = center - width / 2.0
        right = center + width / 2.0
        apex = left + apex_fraction * width
        inside = (grid >= left) & (grid <= right)
        rising = inside & (grid <= apex)
        falling = inside & (grid > apex)
        result[row, rising] = (grid[rising] - left) / (apex - left)
        result[row, falling] = (right - grid[falling]) / (right - apex)
    return result


def triangular_lobe_lipschitz_certificate(
    x: ArrayLike,
    *,
    center_bounds: tuple[float, float],
    width_bounds: tuple[float, float],
    apex_fraction_bounds: tuple[float, float],
) -> TriangularLobeLipschitzCertificate:
    """Certify a normalized interior location--width--asymmetry family.

    The support of every lobe must lie strictly between the first and last
    sample.  If ``h`` is the shortest possible apex-to-support segment and
    ``Delta`` is the largest sample gap, ``1 - Delta/(2h) > 0`` guarantees a
    sampled value near every apex.  Contrasting that value with the two zero
    endpoint samples gives an explicit affine-residual norm lower bound.

    The raw triangular map is globally Lipschitz because its derivatives with
    respect to the left/apex/right knots are bounded by ``sqrt(2)/h``.  The
    returned normalized constant also accounts for affine projection and unit
    normalization.  All bounds are analytic and independent of observations.
    """

    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
        raise ValueError("x must be a finite one-dimensional grid with n >= 3")
    gaps = np.diff(grid)
    if np.any(gaps <= 0.0):
        raise ValueError("x must be strictly increasing")

    center_low, center_high = map(float, center_bounds)
    width_low, width_high = map(float, width_bounds)
    apex_low, apex_high = map(float, apex_fraction_bounds)
    bounds = np.asarray(
        [center_low, center_high, width_low, width_high, apex_low, apex_high]
    )
    if not np.all(np.isfinite(bounds)):
        raise ValueError("all parameter bounds must be finite")
    if center_high <= center_low:
        raise ValueError("center bounds must have positive width")
    if width_low <= 0.0 or width_high <= width_low:
        raise ValueError("width bounds must satisfy 0 < low < high")
    if not 0.0 < apex_low < apex_high < 1.0:
        raise ValueError("apex-fraction bounds must lie strictly inside (0, 1)")
    if center_low - width_high / 2.0 <= grid[0]:
        raise ValueError("every support must be strictly right of the first sample")
    if center_high + width_high / 2.0 >= grid[-1]:
        raise ValueError("every support must be strictly left of the last sample")

    maximum_gap = float(np.max(gaps))
    minimum_segment = min(apex_low, 1.0 - apex_high) * width_low
    sampled_height = 1.0 - maximum_gap / (2.0 * minimum_segment)
    if sampled_height <= 0.0:
        raise ValueError(
            "grid is too coarse to certify a positive near-apex sampled height"
        )

    # For knots (left, apex, right), an active branch has gradient norm at
    # most sqrt(2)/minimum_segment.  The parameter-to-knot Jacobian has rows
    # (1,-1/2,0), (1,a-1/2,w), and (1,1/2,0); its operator norm is bounded by
    # the displayed Frobenius norm uniformly over the box.
    maximum_apex_offset = max(abs(apex_low - 0.5), abs(apex_high - 0.5))
    knot_jacobian_bound = sqrt(
        3.5 + maximum_apex_offset**2 + width_high**2
    )
    raw_lipschitz = (
        sqrt(2.0 * grid.size) * knot_jacobian_bound / minimum_segment
    )
    residual_lower = sampled_height / sqrt(2.0)
    normalized_lipschitz = 2.0 * raw_lipschitz / residual_lower
    parameter_widths = np.asarray(
        [center_high - center_low, width_high - width_low, apex_high - apex_low],
        dtype=float,
    )
    diameter = min(
        2.0, normalized_lipschitz * float(np.linalg.norm(parameter_widths))
    )
    entropy = parameter_entropy_integral_upper(
        parameter_widths,
        normalized_lipschitz,
        canonical_diameter=diameter,
    )
    return TriangularLobeLipschitzCertificate(
        sample_count=int(grid.size),
        maximum_grid_spacing=maximum_gap,
        minimum_apex_segment=float(minimum_segment),
        sampled_apex_height_lower=float(sampled_height),
        residual_norm_lower=float(residual_lower),
        raw_template_lipschitz=float(raw_lipschitz),
        normalized_template_lipschitz=float(normalized_lipschitz),
        parameter_widths=parameter_widths,
        canonical_diameter_upper=float(diameter),
        entropy_integral_upper=float(entropy),
    )


def scan_lobe_sieve(
    observation: ArrayLike,
    templates: ArrayLike,
    parameters: ArrayLike,
    *,
    noise_sigma: float,
    canonical_mesh_radius: float,
    alpha: float = 0.05,
    x: ArrayLike | None = None,
) -> SieveLobeScanResult:
    """Scan an epsilon-net of a fixed continuous template family.

    ``canonical_mesh_radius`` must certify that every normalized template in
    the continuous family lies within that L2 distance of a supplied net row.
    If the true standardized signal is ``mu * v_theta``, its nearest net row
    has mean score at least ``mu * (1 - epsilon**2 / 2)``.
    """

    values = np.asarray(observation, dtype=float)
    theta = np.asarray(parameters, dtype=float)
    dictionary = residualized_lobe_dictionary(templates, x=x)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2 or theta.shape[0] != dictionary.shape[0]:
        raise ValueError("one parameter row is required for every template")
    if values.shape != (dictionary.shape[1],):
        raise ValueError("observation and templates must share their sample length")
    if not np.isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError("noise_sigma must be finite and positive")
    if not np.isfinite(canonical_mesh_radius) or not 0.0 <= canonical_mesh_radius < sqrt(2.0):
        raise ValueError("canonical_mesh_radius must lie in [0, sqrt(2))")

    scores = dictionary @ values / noise_sigma
    selected = int(np.argmax(scores))
    threshold = scan_detection_threshold(dictionary.shape[0], alpha)
    statistic = float(scores[selected])
    factor = 1.0 - canonical_mesh_radius**2 / 2.0
    return SieveLobeScanResult(
        scores=scores,
        selected_index=selected,
        selected_parameter=theta[selected].copy(),
        statistic=statistic,
        threshold=threshold,
        rejected=statistic > threshold,
        templates=dictionary,
        parameters=theta.copy(),
        canonical_mesh_radius=float(canonical_mesh_radius),
        continuous_signal_factor=float(factor),
    )


def sieve_power_sufficient_norm(
    template_count: int,
    alpha: float,
    beta: float,
    canonical_mesh_radius: float,
) -> float:
    """Power bound uniform over alternatives covered by a certified net."""

    if not 0.0 < beta < 1.0:
        raise ValueError("0 < beta < 1 is required")
    if not 0.0 <= canonical_mesh_radius < sqrt(2.0):
        raise ValueError("canonical_mesh_radius must lie in [0, sqrt(2))")
    factor = 1.0 - canonical_mesh_radius**2 / 2.0
    numerator = scan_detection_threshold(template_count, alpha) + NormalDist().inv_cdf(
        1.0 - beta
    )
    return numerator / factor


def affine_residual_subspace_rank(x: ArrayLike) -> int:
    """Rank of the orthogonal complement of the affine null design.

    Every residualized lobe template lies in this deterministic subspace.  The
    value is computed rather than hard-coded so degenerate designs fail safely.
    """

    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
        raise ValueError("x must be a finite one-dimensional grid with n >= 3")
    design = np.c_[np.ones(grid.size), grid]
    design_rank = int(np.linalg.matrix_rank(design))
    residual_rank = int(grid.size - design_rank)
    if residual_rank <= 0:
        raise ValueError("affine residual subspace must have positive rank")
    return residual_rank


def subspace_scan_detection_threshold(subspace_rank: int, alpha: float) -> float:
    """Dependency-free level bound for any unit-template subspace scan.

    If all templates lie in a fixed ``q``-dimensional subspace, their Gaussian
    scan supremum is at most the norm of a standard Gaussian vector in that
    subspace.  Laurent--Massart gives

    ``||Z_q||^2 <= q + 2 sqrt(q log(1/alpha)) + 2 log(1/alpha)``

    with probability at least ``1-alpha``.  The exact square-root chi-square
    quantile is a smaller valid threshold when a chi-square quantile routine is
    available.
    """

    if isinstance(subspace_rank, bool) or int(subspace_rank) != subspace_rank:
        raise ValueError("subspace_rank must be a positive integer")
    rank = int(subspace_rank)
    if rank <= 0:
        raise ValueError("subspace_rank must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("0 < alpha < 1 is required")
    tail = log(1.0 / alpha)
    return sqrt(rank + 2.0 * sqrt(rank * tail) + 2.0 * tail)


def subspace_scan_power_sufficient_norm(
    subspace_rank: int, alpha: float, beta: float
) -> float:
    """Uniform pointwise power bound for the subspace-calibrated scan."""

    if not 0.0 < beta < 1.0:
        raise ValueError("0 < beta < 1 is required")
    return subspace_scan_detection_threshold(
        subspace_rank, alpha
    ) + NormalDist().inv_cdf(1.0 - beta)


def subspace_scan_localization_sufficient_norm(
    subspace_rank: int, delta: float, identifiability_gap: float
) -> float:
    """Sufficient norm for localization using one subspace noise envelope."""

    if not 0.0 < delta < 1.0:
        raise ValueError("0 < delta < 1 is required")
    if not np.isfinite(identifiability_gap) or not 0.0 < identifiability_gap <= 2.0:
        raise ValueError("identifiability_gap must lie in (0, 2]")
    return (
        2.0 * subspace_scan_detection_threshold(subspace_rank, delta)
        / identifiability_gap
    )


def parameter_entropy_integral_upper(
    parameter_widths: ArrayLike,
    template_lipschitz: float,
    *,
    canonical_diameter: float | None = None,
) -> float:
    """Explicit upper bound for a continuous family's entropy integral.

    Assume a parameter box with side lengths ``parameter_widths`` and a
    normalized template map satisfying

    ``||v(theta) - v(theta')||_2 <= L ||theta - theta'||_2``.

    Coordinate grids give
    ``N(epsilon) <= product_j (1 + L*sqrt(d)*R_j/epsilon)``.  Integrating a
    closed-form majorant yields the returned rigorous bound for
    ``integral sqrt(log N(epsilon)) d epsilon``.  The Lipschitz constant is a
    theorem input, not estimated from the scoring observation.
    """

    widths = np.asarray(parameter_widths, dtype=float)
    if widths.ndim != 1 or not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError("parameter_widths must be a finite nonnegative vector")
    if not np.isfinite(template_lipschitz) or template_lipschitz < 0.0:
        raise ValueError("template_lipschitz must be finite and nonnegative")
    active = widths[widths > 0.0]
    if active.size == 0 or template_lipschitz == 0.0:
        return 0.0
    natural_diameter = min(2.0, template_lipschitz * float(np.linalg.norm(active)))
    diameter = natural_diameter if canonical_diameter is None else float(canonical_diameter)
    if not np.isfinite(diameter) or diameter <= 0.0 or diameter > 2.0:
        raise ValueError("canonical_diameter must lie in (0, 2]")
    dimension = active.size
    coefficients = template_lipschitz * sqrt(float(dimension)) * active
    constant = float(np.sum(np.log1p(coefficients / diameter)))
    return diameter * (sqrt(constant) + 0.5 * sqrt(pi * dimension))


def signed_entropy_integral_upper(
    entropy_integral_upper: float, canonical_diameter: float
) -> float:
    """Bound the entropy integral after adjoining both template signs."""

    if not np.isfinite(entropy_integral_upper) or entropy_integral_upper < 0.0:
        raise ValueError("entropy_integral_upper must be finite and nonnegative")
    if not np.isfinite(canonical_diameter) or not 0.0 <= canonical_diameter <= 2.0:
        raise ValueError("canonical_diameter must lie in [0, 2]")
    return entropy_integral_upper + canonical_diameter * sqrt(log(2.0))


def continuous_scan_detection_threshold(
    entropy_integral_upper: float,
    alpha: float,
    *,
    dudley_constant: float = 24.0,
) -> float:
    """Dudley--Borell level-alpha threshold for an uncountable scan."""

    if not np.isfinite(entropy_integral_upper) or entropy_integral_upper < 0.0:
        raise ValueError("entropy_integral_upper must be finite and nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("0 < alpha < 1 is required")
    if not np.isfinite(dudley_constant) or dudley_constant <= 0.0:
        raise ValueError("dudley_constant must be finite and positive")
    return dudley_constant * entropy_integral_upper + sqrt(2.0 * log(1.0 / alpha))


def continuous_scan_power_sufficient_norm(
    entropy_integral_upper: float,
    alpha: float,
    beta: float,
    *,
    dudley_constant: float = 24.0,
) -> float:
    """Sufficient signal norm for the exact continuous-family supremum."""

    if not 0.0 < beta < 1.0:
        raise ValueError("0 < beta < 1 is required")
    return continuous_scan_detection_threshold(
        entropy_integral_upper, alpha, dudley_constant=dudley_constant
    ) + NormalDist().inv_cdf(1.0 - beta)


def continuous_scan_localization_sufficient_norm(
    signed_entropy_upper: float,
    delta: float,
    identifiability_gap: float,
    *,
    dudley_constant: float = 24.0,
) -> float:
    """Sufficient norm for localization at a declared parameter resolution."""

    if not 0.0 < delta < 1.0:
        raise ValueError("0 < delta < 1 is required")
    if not np.isfinite(identifiability_gap) or not 0.0 < identifiability_gap <= 2.0:
        raise ValueError("identifiability_gap must lie in (0, 2]")
    noise_envelope = dudley_constant * signed_entropy_upper + sqrt(
        2.0 * log(1.0 / delta)
    )
    return 2.0 * noise_envelope / identifiability_gap
