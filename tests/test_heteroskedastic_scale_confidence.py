import numpy as np
import pytest

from hcrd.heteroskedastic_scale_confidence import (
    balanced_residual_block_labels,
    gaussian_heteroskedastic_upper_envelope,
)


@pytest.mark.parametrize("n", [2, 3, 10, 11, 221])
def test_balanced_blocks_have_size_two_or_three(n):
    labels = balanced_residual_block_labels(n)
    assert labels.shape == (n,)
    sizes = np.bincount(labels)
    assert np.all((sizes == 2) | (sizes == 3))


def test_arbitrary_mean_lack_of_fit_only_increases_bound():
    baseline = np.tile([1.0, -1.0], 60)
    shifted = baseline + np.repeat(np.linspace(-3.0, 3.0, 60), 2)
    first = gaussian_heteroskedastic_upper_envelope(
        baseline, max_to_mean_variance_ratio=2.0, failure_probability=0.05
    )
    second = gaussian_heteroskedastic_upper_envelope(
        shifted, max_to_mean_variance_ratio=2.0, failure_probability=0.05
    )
    assert second.upper_scale == pytest.approx(first.upper_scale)


def test_bound_fields_are_internally_consistent():
    y = np.linspace(0.0, 1.0, 200) + 0.1 * np.sin(np.arange(200))
    result = gaussian_heteroskedastic_upper_envelope(
        y, max_to_mean_variance_ratio=3.0, failure_probability=0.01
    )
    expected = np.sqrt(
        2.0
        * 3.0
        * result.residual_sum_squares
        / (200 * result.concentration_denominator)
    )
    assert result.upper_scale == pytest.approx(expected)
    assert result.block_count == 100
    assert result.residual_degrees_of_freedom == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_to_mean_variance_ratio": 0.9, "failure_probability": 0.05},
        {"max_to_mean_variance_ratio": 2.0, "failure_probability": 0.0},
        {"max_to_mean_variance_ratio": 2.0, "failure_probability": 1.0},
        {"max_to_mean_variance_ratio": 20.0, "failure_probability": 0.01},
    ],
)
def test_invalid_or_noninformative_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        gaussian_heteroskedastic_upper_envelope(np.arange(20.0), **kwargs)


def test_seeded_gaussian_envelope_has_nominal_coverage():
    rng = np.random.default_rng(20260903)
    n = 240
    variances = np.linspace(0.5, 1.5, n)
    kappa = float(variances.max() / variances.mean())
    standard_deviations = np.sqrt(variances)
    covered = 0
    repetitions = 2500
    for _ in range(repetitions):
        y = rng.normal(0.0, standard_deviations)
        result = gaussian_heteroskedastic_upper_envelope(
            y,
            max_to_mean_variance_ratio=kappa,
            failure_probability=0.05,
        )
        covered += result.upper_scale >= standard_deviations.max()
    assert covered / repetitions >= 0.94
