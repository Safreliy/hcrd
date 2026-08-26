"""Globally stable outer splits and certified HCRD guide decisions.

The hard HCRD knot map is discontinuous at zero-curvature boundaries.  This
module therefore keeps two mathematically different claims separate:

* the L1 trend-filter guide and its residual are globally nonexpansive in
  Euclidean norm; and
* HCRD applied to that guide remains a hard, only conditionally stable map.

The explicit guide residual preserves exact reconstruction even when the guide
is used only for geometric knot discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import BoundaryRule, Decomposition, decompose, discrete_curvature


@dataclass(frozen=True)
class ProximalCurvatureSplit:
    """The exact split ``original = residual + guide``.

    ``guide`` is the unique minimizer of

    ``0.5 * ||original - guide||_2**2 + regularization * ||D guide||_1``,

    where ``D`` is the divided-slope curvature operator on ``x``.
    """

    original: NDArray[np.float64]
    x: NDArray[np.float64]
    guide: NDArray[np.float64]
    residual: NDArray[np.float64]
    regularization: float

    def reconstruct(self) -> NDArray[np.float64]:
        return self.residual + self.guide


@dataclass(frozen=True)
class QuadraticCurvatureSplit:
    """Exact split using a quadratic divided-curvature proximal guide.

    The guide minimizes
    ``0.5 * ||original-guide||_2**2 + 0.5 * regularization * ||D guide||_2**2``.
    Unlike the L1 guide, it does not force curvature to be sparse.
    """

    original: NDArray[np.float64]
    x: NDArray[np.float64]
    guide: NDArray[np.float64]
    residual: NDArray[np.float64]
    regularization: float

    def reconstruct(self) -> NDArray[np.float64]:
        return self.residual + self.guide


@dataclass(frozen=True)
class CertifiedProximalGuidedDecomposition:
    """A stable outer split plus a hard HCRD decomposition of its guide.

    The certificate is deliberately not a claim of global continuity for the
    hard knot map.  Entry ``i`` of ``certified_curvature_signs`` concerns the
    guide curvature at sample ``i + 1`` under an input L2 perturbation no larger
    than ``input_perturbation_radius``.  Zero denotes an abstention.
    """

    split: ProximalCurvatureSplit | QuadraticCurvatureSplit
    decomposition: Decomposition
    certified_curvature_signs: NDArray[np.int8]
    curvature_error_bounds: NDArray[np.float64]
    input_perturbation_radius: float

    def reconstruct(self) -> NDArray[np.float64]:
        return self.split.residual + self.decomposition.reconstruct()


def _validated_signal_and_locations(
    signal: ArrayLike, x: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be a finite one-dimensional array of length >= 3")
    locations = (
        np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    )
    if (
        locations.ndim != 1
        or locations.shape != y.shape
        or not np.all(np.isfinite(locations))
        or np.any(np.diff(locations) <= 0)
    ):
        raise ValueError("x must be finite, one-dimensional, and strictly increasing")
    return y, locations


def curvature_perturbation_bounds(
    x: ArrayLike, perturbation_radius: float
) -> NDArray[np.float64]:
    """Bound every curvature error from a uniform sample-value error.

    If ``||e||_inf <= radius``, the divided-slope curvature at interior sample
    ``i`` changes by at most

    ``2 * radius * (1 / (x[i]-x[i-1]) + 1 / (x[i+1]-x[i]))``.

    An input L2 ball of the same radius is also covered because
    ``||e||_inf <= ||e||_2``.
    """

    locations = np.asarray(x, dtype=float)
    if (
        locations.ndim != 1
        or locations.size < 3
        or not np.all(np.isfinite(locations))
        or np.any(np.diff(locations) <= 0)
    ):
        raise ValueError("x must contain at least three strictly increasing points")
    if not np.isfinite(perturbation_radius) or perturbation_radius < 0:
        raise ValueError("perturbation_radius must be finite and nonnegative")
    gaps = np.diff(locations)
    return 2.0 * perturbation_radius * (1.0 / gaps[:-1] + 1.0 / gaps[1:])


def certified_curvature_signs(
    signal: ArrayLike,
    perturbation_radius: float,
    x: ArrayLike | None = None,
) -> NDArray[np.int8]:
    """Return curvature signs proved invariant in the specified error ball.

    A zero is an explicit abstention, not a claim that the curvature is zero.
    The strict inequality makes the boundary conservative.
    """

    y, locations = _validated_signal_and_locations(signal, x)
    bounds = curvature_perturbation_bounds(locations, perturbation_radius)
    curvature = discrete_curvature(y, locations)
    signs = np.zeros(curvature.size, dtype=np.int8)
    signs[curvature > bounds] = 1
    signs[curvature < -bounds] = -1
    return signs


def proximal_curvature_split(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    regularization: float,
    max_iterations: int = 100_000,
    tolerance: float = 1e-8,
) -> ProximalCurvatureSplit:
    """Compute the L1-curvature proximal guide through its box-constrained dual.

    SciPy and OSQP are optional ``comparisons`` dependencies.  The solver is
    used only for the numerical realization; the mathematical map is the unique
    proximal minimizer stated in :class:`ProximalCurvatureSplit`.
    """

    y, locations = _validated_signal_and_locations(signal, x)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    if regularization == 0:
        guide = y.copy()
        return ProximalCurvatureSplit(
            original=y.copy(),
            x=locations.copy(),
            guide=guide,
            residual=np.zeros_like(y),
            regularization=0.0,
        )

    try:
        import osqp
        from scipy.sparse import csc_matrix, eye
    except ImportError as error:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "proximal_curvature_split requires the 'comparisons' extra "
            "(SciPy and OSQP)"
        ) from error

    n = y.size
    gaps = np.diff(locations)
    rows = np.repeat(np.arange(n - 2), 3)
    columns = np.column_stack(
        [np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)]
    ).reshape(-1)
    coefficients = np.column_stack(
        [
            1.0 / gaps[:-1],
            -(1.0 / gaps[:-1] + 1.0 / gaps[1:]),
            1.0 / gaps[1:],
        ]
    ).reshape(-1)
    curvature_operator = csc_matrix(
        (coefficients, (rows, columns)), shape=(n - 2, n)
    )
    quadratic = (curvature_operator @ curvature_operator.T).tocsc()
    linear = -np.asarray(curvature_operator @ y, dtype=float)
    constraints = eye(n - 2, format="csc")
    bound = np.full(n - 2, float(regularization))
    solver = osqp.OSQP()
    solver.setup(
        P=quadratic,
        q=linear,
        A=constraints,
        l=-bound,
        u=bound,
        verbose=False,
        eps_abs=tolerance,
        eps_rel=tolerance,
        max_iter=max_iterations,
        polishing=True,
        adaptive_rho=True,
    )
    result = solver.solve(raise_error=False)
    if result.info.status_val not in {1, 2}:
        raise RuntimeError(f"proximal guide optimization failed: {result.info.status}")
    dual = np.asarray(result.x, dtype=float)
    guide = y - np.asarray(curvature_operator.T @ dual, dtype=float)
    residual = y - guide
    return ProximalCurvatureSplit(
        original=y.copy(),
        x=locations.copy(),
        guide=guide,
        residual=residual,
        regularization=float(regularization),
    )


def quadratic_curvature_split(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    regularization: float,
) -> QuadraticCurvatureSplit:
    """Compute a globally nonexpansive quadratic-curvature guide.

    The linear system realizes the proximal map of a convex quadratic.  Its
    eigenvalues lie in ``(0, 1]``; both the guide and residual maps are firmly
    nonexpansive in Euclidean norm.
    """

    y, locations = _validated_signal_and_locations(signal, x)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if regularization == 0:
        return QuadraticCurvatureSplit(
            original=y.copy(),
            x=locations.copy(),
            guide=y.copy(),
            residual=np.zeros_like(y),
            regularization=0.0,
        )
    try:
        from scipy.sparse import csc_matrix, eye
        from scipy.sparse.linalg import spsolve
    except ImportError as error:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "quadratic_curvature_split requires the 'comparisons' extra (SciPy)"
        ) from error

    n = y.size
    gaps = np.diff(locations)
    rows = np.repeat(np.arange(n - 2), 3)
    columns = np.column_stack(
        [np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)]
    ).reshape(-1)
    coefficients = np.column_stack(
        [
            1.0 / gaps[:-1],
            -(1.0 / gaps[:-1] + 1.0 / gaps[1:]),
            1.0 / gaps[1:],
        ]
    ).reshape(-1)
    curvature_operator = csc_matrix(
        (coefficients, (rows, columns)), shape=(n - 2, n)
    )
    system = eye(n, format="csc") + regularization * (
        curvature_operator.T @ curvature_operator
    )
    guide = np.asarray(spsolve(system, y), dtype=float)
    return QuadraticCurvatureSplit(
        original=y.copy(),
        x=locations.copy(),
        guide=guide,
        residual=y - guide,
        regularization=float(regularization),
    )


def certified_proximal_guided_decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    regularization: float,
    input_perturbation_radius: float = 0.0,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
    numerical_curvature_tolerance: float = 1e-8,
) -> CertifiedProximalGuidedDecomposition:
    """Build an exact proximal-guide + HCRD decomposition.

    The HCRD hierarchy is intentionally reported together with, rather than
    substituted for, the curvature certificate.  Its hard knot decisions are
    only stable away from their decision boundaries.
    """

    if input_perturbation_radius < 0 or not np.isfinite(input_perturbation_radius):
        raise ValueError("input_perturbation_radius must be finite and nonnegative")
    if numerical_curvature_tolerance < 0:
        raise ValueError("numerical_curvature_tolerance must be nonnegative")
    split = proximal_curvature_split(
        signal, x, regularization=regularization
    )
    bounds = curvature_perturbation_bounds(
        split.x, input_perturbation_radius
    )
    signs = certified_curvature_signs(
        split.guide, input_perturbation_radius, split.x
    )
    decomposition = decompose(
        split.guide,
        split.x,
        atol=numerical_curvature_tolerance,
        rtol=64 * np.finfo(float).eps,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
    )
    return CertifiedProximalGuidedDecomposition(
        split=split,
        decomposition=decomposition,
        certified_curvature_signs=signs,
        curvature_error_bounds=bounds,
        input_perturbation_radius=float(input_perturbation_radius),
    )


def certified_quadratic_guided_decompose(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    regularization: float,
    input_perturbation_radius: float = 0.0,
    boundary_rule: BoundaryRule = "minimum_curvature",
    max_levels: int | None = None,
    numerical_curvature_tolerance: float = 1e-12,
) -> CertifiedProximalGuidedDecomposition:
    """Build an exact quadratic-guide + HCRD decomposition and certificate."""

    if input_perturbation_radius < 0 or not np.isfinite(input_perturbation_radius):
        raise ValueError("input_perturbation_radius must be finite and nonnegative")
    if numerical_curvature_tolerance < 0:
        raise ValueError("numerical_curvature_tolerance must be nonnegative")
    split = quadratic_curvature_split(
        signal, x, regularization=regularization
    )
    bounds = curvature_perturbation_bounds(split.x, input_perturbation_radius)
    signs = certified_curvature_signs(
        split.guide, input_perturbation_radius, split.x
    )
    decomposition = decompose(
        split.guide,
        split.x,
        atol=numerical_curvature_tolerance,
        rtol=64 * np.finfo(float).eps,
        boundary_rule=boundary_rule,
        max_levels=max_levels,
    )
    return CertifiedProximalGuidedDecomposition(
        split=split,
        decomposition=decomposition,
        certified_curvature_signs=signs,
        curvature_error_bounds=bounds,
        input_perturbation_radius=float(input_perturbation_radius),
    )
