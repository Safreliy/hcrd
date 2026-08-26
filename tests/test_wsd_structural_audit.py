"""Deterministic checks for the prospective WSD structural audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "analyze_wsd_structural_audit.py"
    )
    spec = importlib.util.spec_from_file_location("wsd_structural_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_permutation_spearman_detects_perfect_positive_order() -> None:
    module = _module()
    values = np.arange(8, dtype=float)
    rho, p_value = module.permutation_spearman_p(
        values,
        values,
        alternative="greater",
        permutations=9_999,
        seed=7,
        batch_size=500,
    )
    assert rho == 1.0
    assert p_value < 0.001


def test_holm_adjustment_dominates_raw_p_values() -> None:
    module = _module()
    raw = [0.001, 0.02, 0.04, 0.5]
    adjusted = module.holm_adjust(raw)
    assert np.all(np.asarray(adjusted) >= np.asarray(raw))
