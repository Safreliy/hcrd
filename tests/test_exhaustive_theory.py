import itertools
import math

import numpy as np

from hcrd.core import decompose, total_variation


def _signal_from_curvature(curvature):
    slopes = [0.0]
    for value in curvature:
        slopes.append(slopes[-1] + value)
    signal = [0.0]
    for slope in slopes:
        signal.append(signal[-1] + slope)
    return np.asarray(signal)


def test_exhaustive_small_curvature_patterns_terminate_with_nested_knots():
    for n in range(3, 10):
        for curvature in itertools.product((-1.0, 0.0, 1.0), repeat=n - 2):
            signal = _signal_from_curvature(curvature)
            result = decompose(signal, atol=0.0, rtol=0.0)
            previous = set(range(n))
            for level in result.levels:
                assert set(level.knots).issubset(previous)
                previous = set(level.knots)
            assert result.depth <= max(1, math.ceil(math.log2(n - 1)))


def test_exhaustive_small_integer_arrays_contract_total_variation():
    for n in range(2, 8):
        for values in itertools.product((-1.0, 0.0, 1.0), repeat=n):
            signal = np.asarray(values)
            first = decompose(signal, atol=0.0, rtol=0.0, max_levels=1).levels[0]
            assert total_variation(first.baseline) <= total_variation(signal) + 1e-12
