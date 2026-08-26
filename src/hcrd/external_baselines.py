"""Comparison methods backed by optional third-party scientific packages."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def l1_trend_filter(
    signal: ArrayLike,
    regularization: float,
    *,
    rho: float = 100.0,
    max_iterations: int = 2_000,
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Second-order L1 trend filtering solved by ADMM.

    Minimises ``0.5 ||y-b||_2^2 + regularization ||D2 b||_1``.
    """

    if regularization < 0 or rho <= 0:
        raise ValueError("regularization must be nonnegative and rho positive")
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 3:
        raise ValueError("signal must be one-dimensional with at least three samples")
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import factorized

    n = y.size
    d2 = diags(
        [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    solve = factorized(eye(n, format="csc") + rho * (d2.T @ d2))
    z = np.zeros(n - 2)
    dual = np.zeros(n - 2)
    baseline = y.copy()
    for _ in range(max_iterations):
        baseline = solve(y + rho * d2.T @ (z - dual))
        curvature = d2 @ baseline
        previous_z = z.copy()
        shifted = curvature + dual
        z = np.sign(shifted) * np.maximum(
            np.abs(shifted) - regularization / rho, 0.0
        )
        dual += curvature - z
        primal_residual = np.linalg.norm(curvature - z)
        dual_residual = rho * np.linalg.norm(d2.T @ (z - previous_z))
        if primal_residual <= tolerance * np.sqrt(n) and dual_residual <= tolerance * np.sqrt(n):
            break
    return np.asarray(baseline, dtype=float)


def l1_trend_filter_path(
    signal: ArrayLike,
    regularizations: ArrayLike,
    *,
    max_iterations: int = 100_000,
    tolerance: float = 1e-7,
) -> list[NDArray[np.float64]]:
    """Solve an L1 trend-filtering path through its box-constrained dual.

    OSQP reuses one sparse quadratic-program factorisation while the box bounds
    are updated along the path.  The recovered primal is ``y - D2.T @ dual``.
    """

    import osqp
    from scipy.sparse import diags, eye

    y = np.asarray(signal, dtype=float)
    values = np.asarray(regularizations, dtype=float).reshape(-1)
    if y.ndim != 1 or y.size < 3:
        raise ValueError("signal must be one-dimensional with at least three samples")
    if values.size == 0 or np.any(values < 0):
        raise ValueError("regularizations must be a nonempty nonnegative sequence")
    n = y.size
    d2 = diags(
        [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    quadratic = (d2 @ d2.T).tocsc()
    linear = -np.asarray(d2 @ y, dtype=float)
    constraints = eye(n - 2, format="csc")
    order = np.argsort(values)[::-1]
    first_bound = float(values[order[0]])
    solver = osqp.OSQP()
    solver.setup(
        P=quadratic,
        q=linear,
        A=constraints,
        l=np.full(n - 2, -first_bound),
        u=np.full(n - 2, first_bound),
        verbose=False,
        eps_abs=tolerance,
        eps_rel=tolerance,
        max_iter=max_iterations,
        polishing=True,
        adaptive_rho=True,
    )
    estimates: dict[int, NDArray[np.float64]] = {}
    previous_dual: NDArray[np.float64] | None = None
    for index in order:
        regularization = float(values[index])
        if regularization == 0:
            estimates[int(index)] = y.copy()
            previous_dual = np.zeros(n - 2, dtype=float)
            continue
        solver.update(
            l=np.full(n - 2, -regularization),
            u=np.full(n - 2, regularization),
        )
        if previous_dual is not None:
            solver.warm_start(x=np.clip(previous_dual, -regularization, regularization))
        result = solver.solve(raise_error=False)
        if result.info.status_val not in {1, 2}:
            raise RuntimeError(f"L1 trend dual optimization failed: {result.info.status}")
        previous_dual = np.asarray(result.x, dtype=float)
        estimates[int(index)] = y - np.asarray(d2.T @ previous_dual, dtype=float)
    return [estimates[index] for index in range(values.size)]


def emd_residue(signal: ArrayLike, x: ArrayLike | None = None) -> NDArray[np.float64]:
    """Return the final EMD residue using the EMD-signal/PyEMD package."""

    from PyEMD import EMD

    y = np.asarray(signal, dtype=float)
    locations = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    algorithm = EMD()
    algorithm.emd(y, locations)
    _, residue = algorithm.get_imfs_and_residue()
    return np.asarray(residue, dtype=float)


def ceemdan_slow_tail_path(
    signal: ArrayLike,
    x: ArrayLike | None = None,
    *,
    trials: int = 20,
    noise_seed: int = 0,
    maximum_tail_components: int = 4,
) -> tuple[list[NDArray[np.float64]], float]:
    """Return cumulative slow-tail CEEMDAN baselines and reconstruction error.

    Candidate ``k-1`` is the numerical residue plus the ``k`` slowest returned
    components.  CEEMDAN's own multiprocessing is disabled so callers can
    parallelize independent signals without nested worker pools.
    """

    from PyEMD import CEEMDAN

    y = np.asarray(signal, dtype=float)
    locations = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    if y.ndim != 1 or y.size < 4 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be finite, one-dimensional, and length >= 4")
    if locations.shape != y.shape or not np.all(np.isfinite(locations)):
        raise ValueError("x and signal must have identical finite shapes")
    if np.any(np.diff(locations) <= 0):
        raise ValueError("x must be strictly increasing")
    if trials < 1 or maximum_tail_components < 1:
        raise ValueError("trials and maximum_tail_components must be positive")

    algorithm = CEEMDAN(trials=trials, parallel=False)
    algorithm.noise_seed(noise_seed)
    components = np.asarray(algorithm.ceemdan(y, locations), dtype=float)
    if components.ndim != 2 or components.shape[1] != y.size:
        raise RuntimeError("CEEMDAN returned an invalid component array")
    residue = y - np.sum(components, axis=0)
    reconstruction_error = float(
        np.max(np.abs(y - np.sum(components, axis=0) - residue))
    )
    count = min(maximum_tail_components, components.shape[0])
    candidates = [
        np.asarray(residue + np.sum(components[-tail_count:], axis=0), dtype=float)
        for tail_count in range(1, count + 1)
    ]
    return candidates, reconstruction_error


def iterative_filtering_slow_tail_path(
    signal: ArrayLike,
    *,
    maximum_tail_components: int = 4,
) -> tuple[list[NDArray[np.float64]], float]:
    """Return cumulative slow-tail candidates from the official ``fifpy.IF``.

    The returned IMC matrix is ordered from fast components to the final slow
    component/trend.  Candidate ``k-1`` is the sum of its ``k`` last rows.
    Package defaults are intentionally left untouched.
    """

    import fifpy

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 4 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be finite, one-dimensional, and length >= 4")
    if maximum_tail_components < 1:
        raise ValueError("maximum_tail_components must be positive")

    algorithm = fifpy.IF()
    algorithm.run(y)
    components = np.asarray(algorithm.IMC, dtype=float)
    if components.ndim != 2 or components.shape[1] != y.size:
        raise RuntimeError("iterative filtering returned an invalid component array")
    reconstruction_error = float(np.max(np.abs(y - np.sum(components, axis=0))))
    count = min(maximum_tail_components, components.shape[0])
    candidates = [
        np.asarray(np.sum(components[-tail_count:], axis=0), dtype=float)
        for tail_count in range(1, count + 1)
    ]
    return candidates, reconstruction_error


def vmd_low_frequency(
    signal: ArrayLike,
    *,
    modes: int = 5,
    alpha: float = 2_000.0,
    tolerance: float = 1e-7,
) -> NDArray[np.float64]:
    """Return the lowest-centre-frequency VMD mode."""

    from vmdpy import VMD

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size < 4:
        raise ValueError("signal must be one-dimensional with at least four samples")
    padded = y if y.size % 2 == 0 else np.append(y, y[-1])
    modes_array, _, omega = VMD(
        padded,
        alpha,
        0.0,
        modes,
        1,
        1,
        tolerance,
    )
    final_frequencies = omega[-1]
    lowest = int(np.argmin(np.abs(final_frequencies)))
    return np.asarray(modes_array[lowest, : y.size], dtype=float)
