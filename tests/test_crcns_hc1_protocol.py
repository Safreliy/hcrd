"""Locked split checks for the real paired neural-spike study."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "download_crcns_hc1_spike.py"
    )
    spec = importlib.util.spec_from_file_location("crcns_hc1_download", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_session_split_counts_and_membership() -> None:
    module = _module()
    development = {
        root for root in module.SESSION_ROOTS if module.split_for_root(root) == "development"
    }
    confirmation = set(module.SESSION_ROOTS) - development
    assert development == {
        "d13921", "d14531", "d15121", "d18811", "d5331", "d6111", "d7111", "d7212"
    }
    assert confirmation == {
        "d12821", "d13521", "d13711", "d14921", "d18712", "d5611", "d7211"
    }


def test_padded_chunk_scoring_has_exact_core_coverage() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_crcns_hc1_development_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("crcns_hc1_pilot", path)
    assert spec is not None and spec.loader is not None
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    signal = np.arange(23, dtype=float)
    observed = pilot.chunked_score(
        signal,
        lambda values: values + 1.0,
        core_samples=7,
        padding_samples=2,
    )
    np.testing.assert_array_equal(observed, signal + 1.0)
