import numpy as np

from hcrd.metrics import exact_sign_test, knot_f1, nmse, paired_bootstrap_ci, scaled_mse


def test_metrics_basic_cases():
    assert nmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert knot_f1([0, 10, 20], [0, 11, 20], tolerance=1) == 1.0
    assert exact_sign_test(np.full(10, -1.0)) < 0.01
    low, high = paired_bootstrap_ci([-3.0, -2.0, -1.0], samples=1000, seed=1)
    assert high < 0
    assert scaled_mse([1, 2, 3], [1, 2, 3], [0, 1, 0]) == 0.0
