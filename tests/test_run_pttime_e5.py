from __future__ import annotations

import numpy as np

from experiments.run_ms_metrics_e2 import FINAL_NAMES
from experiments.run_pttime_e5 import (
    _bad_metrics,
    _duration_seconds,
    _pttime_positive_raw_files,
    _raw_files,
    _stratified_bootstrap,
)


def test_duration_seconds_accepts_mzxml_iso_duration() -> None:
    assert _duration_seconds("PT1M2.5S") == 62.5
    assert _duration_seconds("PT0.25H") == 900.0
    assert _duration_seconds(17.0) == 17.0


def test_raw_files_finds_mzml_and_mzxml_recursively(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mzXML").touch()
    (tmp_path / "a.mzML").touch()
    (tmp_path / "ignore.txt").touch()
    assert [path.name for path in _raw_files(tmp_path)] == ["a.mzML", "b.mzXML"]


def test_pttime_positive_raw_files_excludes_negative_mode(tmp_path) -> None:
    (tmp_path / "study_POS_MSMS-v2_1.mzML").touch()
    (tmp_path / "study_NEG_MSMS-v2_1.mzML").touch()
    assert [path.name for path in _pttime_positive_raw_files(tmp_path)] == [
        "study_POS_MSMS-v2_1.mzML"
    ]


def test_bad_metrics_reports_review_queue_yield() -> None:
    labels = np.asarray([1, 0, 1, 0, 0], dtype=np.int8)
    scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1])
    result = _bad_metrics(labels, scores)
    assert result["bad_in_top_17"] == 2
    assert result["bad_in_top_5_percent"] == 1
    assert result["top_5_percent_count"] == 1


def test_stratified_bootstrap_preserves_both_classes() -> None:
    labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8)
    weak = np.asarray([0.55, 0.45, 0.50, 0.40, 0.35, 0.30])
    strong = np.asarray([0.95, 0.90, 0.40, 0.30, 0.20, 0.10])
    scores = {name: weak.copy() for name in FINAL_NAMES}
    scores["hcrd_8_q"] = strong
    result = _stratified_bootstrap(labels, scores, replicates=200)
    assert result["hcrd_8_q"]["ap_bad_difference"] > 0.0
    assert result["hcrd_8_q_vs_hcrd_1_q"]["ap_bad_difference"] > 0.0
