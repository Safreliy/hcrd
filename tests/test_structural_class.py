"""Tests for label-assisted target-class morphology descriptors."""

from __future__ import annotations

import numpy as np

from hcrd import aggregate_event_chord_morphology, event_chord_morphologies


def test_one_sign_triangle_has_unit_sign_coherence() -> None:
    signal = np.zeros(11)
    signal[3:8] = [0.0, 1.0, 2.0, 1.0, 0.0]
    labels = np.zeros(11, dtype=int)
    labels[3:8] = 1
    event = event_chord_morphologies(signal, labels)[0]
    assert np.isclose(event.sign_coherence, 1.0)
    assert event.peak_to_background_mad > 1.0
    assert event.curvature_contrast > 0.0


def test_sign_changing_event_has_low_coherence() -> None:
    signal = np.zeros(11)
    signal[3:8] = [0.0, 1.0, 0.0, -1.0, 0.0]
    labels = np.zeros(11, dtype=int)
    labels[3:8] = 1
    event = event_chord_morphologies(signal, labels)[0]
    assert np.isclose(event.sign_coherence, 0.0)


def test_series_aggregation_reports_event_count() -> None:
    signal = np.asarray([0, 1, 0, 0, -2, 0, 0], dtype=float)
    labels = np.asarray([0, 1, 0, 0, 1, 0, 0], dtype=int)
    summary = aggregate_event_chord_morphology(signal, labels)
    assert summary["event_count"] == 2.0
