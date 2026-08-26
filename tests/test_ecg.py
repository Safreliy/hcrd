import numpy as np

from hcrd.ecg import (
    hcrd_qrs_delineate,
    hcrd_qrs_multilevel_candidates,
    parse_qrs_boundaries,
)


def test_qtdb_qrs_parser_uses_wave_number_one():
    samples = [10, 20, 30, 40, 50, 60, 70]
    symbols = ["(", "p", ")", "(", "N", ")", "t"]
    numbers = [0, 0, 0, 1, 0, 1, 0]
    result = parse_qrs_boundaries(samples, symbols, numbers)
    assert len(result) == 1
    assert (result[0].onset, result[0].fiducial, result[0].offset) == (40, 50, 60)


def test_hcrd_delineator_returns_ordered_boundaries_on_sharp_transient():
    x = np.arange(201)
    signal = (
        -0.3 * np.exp(-0.5 * ((x - 92) / 4) ** 2)
        + 1.8 * np.exp(-0.5 * ((x - 100) / 3) ** 2)
        - 0.5 * np.exp(-0.5 * ((x - 109) / 5) ** 2)
    )
    result = hcrd_qrs_delineate(
        signal,
        100,
        250.0,
        guide="quadratic",
        regularization=20.0,
        amplitude_ratio=0.15,
    )
    assert result.succeeded
    assert result.onset < 100 < result.offset


def test_multilevel_qrs_candidates_preserve_first_level_rule():
    x = np.arange(201)
    signal = (
        -0.3 * np.exp(-0.5 * ((x - 92) / 4) ** 2)
        + 1.8 * np.exp(-0.5 * ((x - 100) / 3) ** 2)
        - 0.5 * np.exp(-0.5 * ((x - 109) / 5) ** 2)
    )
    single = hcrd_qrs_delineate(
        signal,
        100,
        250.0,
        guide="quadratic",
        regularization=20.0,
        amplitude_ratio=0.15,
    )
    candidates = hcrd_qrs_multilevel_candidates(
        signal,
        100,
        250.0,
        guide="quadratic",
        regularization=20.0,
        amplitude_ratios=(0.15,),
        max_levels=4,
    )
    assert len(candidates) >= 2
    first = candidates[0]
    assert first.level == 1
    assert (first.onset, first.offset) == (single.onset, single.offset)
    assert first.normalized_anchor_amplitude > 0
