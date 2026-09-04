from itertools import pairwise

import numpy as np
import pytest
from scipy.optimize import linprog

from shapecontrast import design_identified_transition_set


def _linear_program_feasible(
    x: np.ndarray, mean: np.ndarray, transition: float
) -> bool:
    """Independent finite-dimensional feasibility oracle for one location."""

    nodes = np.unique(np.append(x, transition))
    observed = {float(location): float(value) for location, value in zip(x, mean)}
    bounds = [
        (observed[float(node)], observed[float(node)])
        if float(node) in observed
        else (None, None)
        for node in nodes
    ]
    constraints: list[np.ndarray] = []
    for middle in range(1, nodes.size - 1):
        left_gap = nodes[middle] - nodes[middle - 1]
        right_gap = nodes[middle + 1] - nodes[middle]
        convex_row = np.zeros(nodes.size, dtype=float)
        convex_row[middle - 1 : middle + 2] = (
            -1.0 / left_gap,
            1.0 / left_gap + 1.0 / right_gap,
            -1.0 / right_gap,
        )
        convex_row /= np.max(np.abs(convex_row))
        if nodes[middle + 1] <= transition:
            constraints.append(convex_row)
        if nodes[middle - 1] >= transition:
            constraints.append(-convex_row)

    matrix = np.vstack(constraints) if constraints else None
    result = linprog(
        np.zeros(nodes.size, dtype=float),
        A_ub=matrix,
        b_ub=None if matrix is None else np.zeros(matrix.shape[0]),
        bounds=bounds,
        method="highs",
    )
    assert result.status in (0, 2), result.message
    return result.status == 0


def _target_contains(components: tuple[tuple[float, float], ...], point: float) -> bool:
    return any(left - 1e-8 <= point <= right + 1e-8 for left, right in components)


def _closure_oracle_feasible(
    x: np.ndarray, mean: np.ndarray, transition: float
) -> bool:
    """LP feasibility, including closure at the two observed endpoints."""

    if _linear_program_feasible(x, mean, transition):
        return True
    if transition == x[0]:
        return _linear_program_feasible(x, mean, 0.5 * (x[0] + x[1]))
    if transition == x[-1]:
        return _linear_program_feasible(x, mean, 0.5 * (x[-2] + x[-1]))
    return False


def test_affine_sample_identifies_the_full_declared_domain() -> None:
    x = np.array([0.1, 0.3, 0.55, 0.8])
    mean = 2.0 - 3.0 * x
    target = design_identified_transition_set(x, mean, domain=(0.0, 1.0))

    assert target.components == ((0.0, 1.0),)
    assert target.hull == (0.0, 1.0)


def test_incompatible_slope_oscillation_has_empty_target() -> None:
    x = np.arange(5, dtype=float)
    mean = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    target = design_identified_transition_set(x, mean)

    assert target.empty
    assert target.hull is None


def test_jump_target_is_the_design_gap_containing_the_jump() -> None:
    x = np.arange(1, 1001, dtype=float) / 1001.0
    mean = x + (x >= 0.3).astype(float)
    target = design_identified_transition_set(x, mean, domain=(0.0, 1.0))
    transition_index = int(np.searchsorted(x, 0.3))

    assert target.hull == pytest.approx((x[transition_index - 1], x[transition_index]))


def test_slope_and_dimensionless_tolerances_cannot_create_reversed_piece() -> None:
    x = np.arange(7, dtype=float)
    slopes = np.array([-5.0, -3.0, -2.0, 6.0, -10.0, -9.0])
    mean = np.r_[0.0, np.cumsum(slopes)]

    target = design_identified_transition_set(x, mean, slope_tolerance=0.1)

    assert target.empty


def test_linear_algorithm_agrees_with_independent_lp_oracle() -> None:
    rng = np.random.default_rng(20260904)
    for _ in range(30):
        n = int(rng.integers(3, 8))
        raw_x = np.r_[0.0, np.cumsum(rng.uniform(0.2, 1.0, size=n - 1))]
        x = 0.1 + 0.8 * raw_x / raw_x[-1]
        mean = rng.normal(size=n)
        domain = (0.0, 1.0)
        target = design_identified_transition_set(
            x, mean, domain=domain, slope_tolerance=0.0
        )
        assert len(target.components) <= 1
        candidates = [domain[0], 0.5 * (domain[0] + x[0]), *x]
        for left, right in pairwise(x):
            candidates.extend(
                left + fraction * (right - left) for fraction in (0.25, 0.5, 0.75)
            )
        candidates.extend((0.5 * (x[-1] + domain[1]), domain[1]))

        for candidate in candidates:
            assert _target_contains(
                target.components, float(candidate)
            ) == _closure_oracle_feasible(x, mean, float(candidate))


@pytest.mark.parametrize(
    "signal",
    [
        "cusp",
        "onset",
        "jump",
        "logistic",
    ],
)
def test_published_benchmark_means_have_nontrivial_targets(signal: str) -> None:
    x = np.arange(1, 501, dtype=float) / 501.0
    if signal == "cusp":
        left = 2.0 * (0.3 - np.sqrt(np.maximum(0.09 - x**2, 0.0)))
        right = 2.0 * (0.3 + np.sqrt(np.maximum(0.49 - (1.0 - x) ** 2, 0.0)))
        mean = np.where(x < 0.3, left, right)
    elif signal == "onset":
        mean = np.where(x < 0.3, 0.0, np.sin((x - 0.3) * np.pi / 1.4))
    elif signal == "jump":
        mean = x + (x >= 0.3).astype(float)
    else:
        mean = 4.0 / (1.0 + np.exp(-2.0 * (x - 0.3)))

    target = design_identified_transition_set(x, mean, domain=(0.0, 1.0))

    assert target.left is not None and target.right is not None
    assert target.left < 0.3 < target.right


@pytest.mark.parametrize(
    "x,mean,domain",
    [
        ([0.0, 0.5], [0.0, 1.0], None),
        ([0.0, 0.5, 0.4], [0.0, 1.0, 2.0], None),
        ([0.0, 0.5, 1.0], [0.0, 1.0], None),
        ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], (0.1, 1.0)),
    ],
)
def test_identified_target_rejects_invalid_inputs(x, mean, domain) -> None:
    with pytest.raises(ValueError):
        design_identified_transition_set(x, mean, domain=domain)
