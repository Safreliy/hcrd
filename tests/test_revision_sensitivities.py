import numpy as np
import warnings

from experiments.run_ms_metrics_e2_file_group_sensitivity import (
    _file_folds,
    _nan_summary,
)
from experiments.run_qscore_implementation_sensitivity import _summaries


def test_qscore_summary_variants_use_declared_statistics():
    cube = np.array(
        [
            [[1.0, 0.1], [2.0, 0.2]],
            [[3.0, 0.5], [4.0, np.nan]],
            [[5.0, 0.9], [6.0, 0.8]],
        ],
        dtype=float,
    )
    summaries = _summaries(cube)
    np.testing.assert_allclose(summaries["q2"][0], [3.0, 0.5])
    np.testing.assert_allclose(summaries["q5"][0], [3.0, 0.5, 5.0, 0.9, 0.5])
    np.testing.assert_allclose(summaries["q7"][1, -1], 1.0)
    assert summaries["q2"].shape == (2, 2)
    assert summaries["q5"].shape == (2, 5)
    assert summaries["q7"].shape == (2, 7)


def test_file_delete_groups_are_deterministic_disjoint_partition():
    left = _file_folds(41, 10, 0)
    right = _file_folds(41, 10, 0)
    assert all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))
    combined = np.concatenate(left)
    np.testing.assert_array_equal(np.sort(combined), np.arange(41))
    assert max(map(len, left)) - min(map(len, left)) <= 1


def test_vectorized_nan_summary_matches_numpy_linear_quantiles():
    rng = np.random.default_rng(29)
    block = rng.normal(size=(17, 5, 7)).astype(np.float32)
    block[rng.random(block.shape) < 0.2] = np.nan
    block[:, 0, 0] = np.nan
    median, q90, maximum = _nan_summary(block)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected_median = np.nanmedian(block, axis=0)
        expected_q90 = np.nanquantile(block, 0.9, axis=0)
        expected_maximum = np.nanmax(block, axis=0)
    np.testing.assert_allclose(median, expected_median, equal_nan=True)
    np.testing.assert_allclose(q90, expected_q90, equal_nan=True)
    np.testing.assert_allclose(maximum, expected_maximum, equal_nan=True)
