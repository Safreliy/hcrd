from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

from run_ppg_dalia_local_motion import local_motion_features  # noqa: E402


def test_local_motion_features_have_fixed_shape_and_zero_for_constant_signal():
    acceleration = np.full(320, 1000.0)
    positions = np.asarray([0, 64, 128, 639])
    features, names = local_motion_features(acceleration, 32.0, positions, 64.0)
    assert features.shape == (4, 11)
    assert len(names) == len(set(names)) == 11
    np.testing.assert_array_equal(features, np.zeros_like(features))


def test_local_motion_features_interpolate_nonfinite_values():
    acceleration = np.sin(np.arange(320) / 7.0)
    acceleration[[0, 17, 319]] = np.nan
    features, _ = local_motion_features(
        acceleration, 32.0, np.arange(0, 640, 13), 64.0
    )
    assert features.shape == (50, 11)
    assert np.all(np.isfinite(features))
