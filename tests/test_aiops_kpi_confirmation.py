"""Deterministic helpers for the AIOps C3 confirmation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_aiops_kpi_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location("aiops_c3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_sign_flip_known_cases() -> None:
    module = _module()
    assert module.exact_sign_flip_p(np.asarray([1.0])) == 1.0
    assert module.exact_sign_flip_p(np.ones(3)) == 0.25


def test_longest_run() -> None:
    module = _module()
    assert module.longest_run(np.asarray([0, 1, 1, 0, 1, 1, 1, 0])) == 3
