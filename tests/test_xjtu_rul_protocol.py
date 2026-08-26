from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

from run_xjtu_rul import add_compact_energy_features, create_causal_windows  # noqa: E402


def test_causal_windows_do_not_cross_bearing_boundaries():
    rows = []
    for bearing in ("Bearing1_1", "Bearing1_2"):
        for index in range(8):
            rows.append(
                {
                    "condition": "35Hz12kN",
                    "bearing_id": bearing,
                    "file_idx": index,
                    "feature": index + (100 if bearing.endswith("2") else 0),
                }
            )
    data = create_causal_windows(
        pd.DataFrame(rows), ["feature"], window_size=3, calibration="first20"
    )
    assert data.values.shape == (12, 3, 1)
    assert np.all(data.bearings[:6] == "Bearing1_1")
    assert np.all(data.bearings[6:] == "Bearing1_2")
    assert np.allclose(data.targets[[0, 5, 6, 11]], [5 / 7, 0, 5 / 7, 0])


def test_compact_energy_aggregates_axes_without_label_information():
    row = {}
    for level in range(6):
        for measure in (
            "log1p_polygon_area",
            "log1p_quadratic_energy",
            "area_concentration",
            "weighted_shape_factor",
        ):
            row[f"h_env_level_{level}_{measure}"] = 1.0 + level
            row[f"v_env_level_{level}_{measure}"] = 3.0 + level
    transformed, names = add_compact_energy_features(pd.DataFrame([row]))
    assert len(names) == 48
    assert transformed.loc[0, names[0]] == 2.0
    assert transformed.loc[0, names[1]] == 3.0
