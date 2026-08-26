from __future__ import annotations

import numpy as np

from hcrd.lcms import eic_feature_bank, eic_pair_feature_bank, eic_partition


def test_eic_feature_bank_is_finite_and_uses_multiple_levels() -> None:
    x = np.linspace(0.0, 14.0, 71) ** 1.03
    y = 50.0 + 4.0 * x + 800.0 / (1.0 + ((x - 7.0) / 0.7) ** 2)
    bank = eic_feature_bank(y, x)
    assert bank.raw64.shape == (75,)
    assert bank.domain.shape == (111,)
    assert bank.hcrd_1.size > bank.raw64.size
    assert bank.hcrd_8.size > bank.hcrd_1.size
    assert bank.hcrd_geometry.shape == (297,)
    assert bank.area_only.shape == (48,)
    assert all(
        np.all(np.isfinite(values))
        for values in (
            bank.raw64,
            bank.domain,
            bank.hcrd_1,
            bank.hcrd_8,
            bank.hcrd_geometry,
            bank.area_only,
        )
    )


def test_eic_duplicate_times_are_aggregated() -> None:
    x = np.r_[np.arange(8.0), 7.0]
    y = np.r_[np.arange(8.0), 9.0]
    bank = eic_feature_bank(y, x)
    assert bank.raw64.size == 75


def test_eic_pair_concatenates_both_windows() -> None:
    short_x = np.linspace(-7.0, 7.0, 31)
    long_x = np.linspace(-14.0, 14.0, 61)
    short_y = np.exp(-0.5 * (short_x / 0.8) ** 2)
    long_y = np.exp(-0.5 * (long_x / 0.8) ** 2)
    pair = eic_pair_feature_bank(short_y, short_x, long_y, long_x)
    assert pair.raw64.shape == (150,)
    assert pair.hcrd_geometry.shape == (594,)


def test_eic_partition_rejects_unknown_axis() -> None:
    try:
        eic_partition("row", "1")
    except ValueError as error:
        assert "axis" in str(error)
    else:
        raise AssertionError("unknown split axis was accepted")
