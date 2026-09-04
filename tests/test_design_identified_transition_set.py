import numpy as np
import pytest

from shapecontrast import design_identified_transition_set


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
