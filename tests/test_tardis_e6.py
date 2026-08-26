from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_tardis_e6 import (
    _prepare_positive_labels,
    _retention_time_seconds,
    _top_decile_enrichment,
)


class _UnitValue(float):
    unit_info = "minute"


def test_mzxml_retention_time_is_converted_to_seconds() -> None:
    spectrum = {"retentionTime": _UnitValue(1.25)}
    assert _retention_time_seconds(spectrum) == 75.0


def test_positive_label_preparation_freezes_ppm_and_time_box(monkeypatch) -> None:
    source = pd.DataFrame(
        {
            "Component": [7, 9],
            "m.z": [100.0, 250.0],
            "tr": [12.0, 70.0],
            "Rating": ["Bad", "Good"],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda _: source)
    result = _prepare_positive_labels(Path("unused.xlsx"))
    assert result["feature"].tolist() == ["7", "9"]
    assert np.isclose(result.loc[0, "min_mz"], 99.999)
    assert np.isclose(result.loc[0, "max_mz"], 100.001)
    assert result.loc[0, "min_rt"] == 0.0
    assert result.loc[0, "max_rt"] == 42.0
    assert result.loc[1, "min_rt"] == 40.0
    assert result.loc[1, "max_rt"] == 100.0


def test_unrated_placeholder_without_waveform_key_is_ignored(monkeypatch) -> None:
    source = pd.DataFrame(
        {
            "Component": [7, 187],
            "m.z": [100.0, np.nan],
            "tr": [12.0, np.nan],
            "Rating": ["Bad", np.nan],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda _: source)
    result = _prepare_positive_labels(Path("unused.xlsx"))
    assert result["feature"].tolist() == ["7"]


def test_top_decile_enrichment_uses_highest_bad_scores() -> None:
    labels = np.asarray([0] * 18 + [1, 1], dtype=np.int8)
    scores = np.arange(20, dtype=float)
    assert _top_decile_enrichment(labels, scores) == 10.0
