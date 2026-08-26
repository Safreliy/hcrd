"""Small deterministic checks for the C2 WSD protocol helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_wsd_sparse_transient_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location("wsd_c2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_and_event_helpers() -> None:
    module = _module()
    labels = np.asarray([0, 1, 1, 0, 1, 0, 1, 1, 1])
    assert module.longest_run(labels) == 3
    assert module.event_count(labels) == 3


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    module = _module()
    raw = [0.04, 0.001, 0.02]
    adjusted = module.holm_adjust(raw)
    order = np.argsort(raw)
    ordered = np.asarray(adjusted)[order]
    assert np.all(np.diff(ordered) >= 0.0)
    assert np.all(ordered >= np.asarray(raw)[order])
