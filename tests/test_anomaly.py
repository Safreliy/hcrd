import numpy as np
import pytest

from hcrd import aggregate_area_density, hcrd_area_anomaly_score


@pytest.mark.parametrize("aggregation", ["sum", "max", "l2", "transport"])
def test_robust_area_scores_are_affine_and_scale_invariant(aggregation):
    x = np.linspace(-2.0, 3.0, 257)
    signal = np.sin(7.0 * x) + 0.2 * np.sin(31.0 * x)
    original = hcrd_area_anomaly_score(
        signal, max_levels=6, aggregation=aggregation
    )
    transformed = hcrd_area_anomaly_score(
        -3.5 * signal + 0.7 * x - 1.2,
        max_levels=6,
        aggregation=aggregation,
    )
    np.testing.assert_allclose(transformed, original, atol=2e-11, rtol=2e-11)


def test_area_anomaly_score_is_finite_and_rejects_unknown_aggregation():
    signal = np.asarray([0.0, 1.0, 0.0, -2.0, 0.0, 0.5, 0.0])
    score = hcrd_area_anomaly_score(signal, aggregation="transport")
    assert score.shape == signal.shape
    assert np.all(np.isfinite(score))
    assert np.all(score >= 0.0)
    with pytest.raises(ValueError, match="unknown area aggregation"):
        hcrd_area_anomaly_score(signal, aggregation="bad")


def test_precomputed_density_aggregation_validates_its_input():
    with pytest.raises(ValueError, match="shape"):
        aggregate_area_density(np.ones(4))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        aggregate_area_density(np.asarray([[0.0, -1.0]]))
