from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

from run_xjtu_energy_features import (  # noqa: E402
    PROFILE_BINS,
    _block_rms_profile,
    _log_power_profile,
    _profile_features,
)


def test_xjtu_profiles_have_frozen_length_and_finite_features():
    grid = np.arange(32_768)
    signal = 0.3 * np.sin(2.0 * np.pi * grid / 73.0)
    signal[::997] += 2.0
    rms = _block_rms_profile(signal)
    power = _log_power_profile(signal)
    assert rms.shape == power.shape == (PROFILE_BINS,)
    features = _profile_features(rms, "test_")
    assert len(features) == 72
    assert np.all(np.isfinite(list(features.values())))
