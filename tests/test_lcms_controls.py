from __future__ import annotations

import numpy as np

from hcrd.lcms_controls import gaussian_derivative_control


def test_gaussian_derivative_control_matches_hcrd8_width() -> None:
    x = np.linspace(-3.0, 3.0, 64)
    waveform = np.exp(-0.5 * x**2)
    raw = np.r_[waveform, np.linspace(0.0, 1.0, 11)]
    control = gaussian_derivative_control(np.stack([raw, 0.5 * raw]))
    assert control.shape == (2, 948)
    assert np.all(np.isfinite(control))


def test_gaussian_derivative_control_preserves_missing_rows() -> None:
    raw = np.full((3, 75), np.nan)
    control = gaussian_derivative_control(raw)
    assert control.shape == (3, 948)
    assert np.all(np.isnan(control))
