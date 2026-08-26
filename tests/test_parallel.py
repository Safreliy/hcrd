import numpy as np

from hcrd.features import representation_features_batch
from hcrd.parallel import decompose_batch, decompose_sparse_batch


def _signals() -> list[np.ndarray]:
    rng = np.random.default_rng(812)
    return [rng.normal(size=65) for _ in range(6)]


def test_decompose_batch_preserves_order_and_serial_values():
    signals = _signals()
    serial = decompose_batch(signals, backend="serial", max_levels=3)
    threaded = decompose_batch(signals, backend="thread", workers=3, max_levels=3)
    assert len(serial) == len(threaded) == len(signals)
    for expected, actual in zip(serial, threaded, strict=True):
        np.testing.assert_array_equal(actual.reconstruct(), expected.reconstruct())
        assert actual.depth == expected.depth
        for first, second in zip(expected.levels, actual.levels, strict=True):
            np.testing.assert_array_equal(first.knots, second.knots)


def test_batch_features_are_identical_across_backends():
    signals = _signals()
    serial = representation_features_batch(signals, "hcrd", backend="serial")
    threaded = representation_features_batch(
        signals, "hcrd", backend="thread", workers=3
    )
    np.testing.assert_array_equal(threaded, serial)


def test_batch_features_are_identical_with_process_workers():
    signals = _signals()
    serial = representation_features_batch(signals, "hcrd", backend="serial")
    processed = representation_features_batch(
        signals, "hcrd", backend="process", workers=2, chunksize=2
    )
    np.testing.assert_array_equal(processed, serial)


def test_empty_batch_has_stable_feature_shape():
    result = representation_features_batch([], "hcrd", backend="serial")
    assert result.shape == (0, 50)


def test_batch_rejects_invalid_parallel_options():
    with np.testing.assert_raises(ValueError):
        decompose_batch(_signals(), workers=0)
    with np.testing.assert_raises(ValueError):
        representation_features_batch(_signals(), "hcrd", backend="invalid")


def test_sparse_batch_matches_serial_knot_hierarchies():
    rng = np.random.default_rng(59)
    signals = [rng.normal(size=65 + index) for index in range(6)]
    expected = decompose_sparse_batch(signals, backend="serial")
    actual = decompose_sparse_batch(signals, backend="process", workers=2)
    for first, second in zip(expected, actual, strict=True):
        assert first.depth == second.depth
        for left, right in zip(first.knot_sets, second.knot_sets, strict=True):
            np.testing.assert_array_equal(left, right)
