"""Fixed LightGBM robustness check for the XJTU-SY X2 representations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

from run_xjtu_rul import (  # noqa: E402
    KEY_COLUMNS,
    METADATA_COLUMNS,
    add_compact_energy_features,
    create_causal_windows,
    polygon_mass_feature_names,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from lightgbm import LGBMRegressor

    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-features", type=Path, required=True)
    parser.add_argument("--hcrd-features", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "xjtu_x2_tree"
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=(
            "standard",
            "hcrd_compact",
            "hybrid_compact",
            "hcrd_mass6",
            "hybrid_mass6",
        ),
        default=("standard", "hcrd_compact", "hybrid_compact"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(9,))
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument(
        "--calibration", choices=("fraction20", "first20"), default="fraction20"
    )
    args = parser.parse_args()

    standard = pd.read_csv(args.standard_features)
    energy = pd.read_csv(args.hcrd_features)
    merged = standard.merge(
        energy.drop(columns=["total_files", "rul_normalized"]),
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    merged, compact = add_compact_energy_features(merged)
    standard_columns = [
        column for column in standard.columns if column not in METADATA_COLUMNS
    ]
    representation_columns = {
        "standard": standard_columns,
        "hcrd_compact": compact,
        "hybrid_compact": standard_columns + compact,
        "hcrd_mass6": polygon_mass_feature_names(),
        "hybrid_mass6": standard_columns + polygon_mass_feature_names(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    for representation in args.representations:
        windows = create_causal_windows(
            merged,
            representation_columns[representation],
            window_size=args.window_size,
            calibration=args.calibration,
        )
        flat = windows.values.reshape(windows.values.shape[0], -1)
        for seed in args.seeds:
            for condition in sorted(set(windows.conditions)):
                for test_bearing in sorted(
                    set(windows.bearings[windows.conditions == condition])
                ):
                    test = windows.bearings == test_bearing
                    train = (windows.conditions == condition) & ~test
                    model = LGBMRegressor(
                        objective="regression_l2",
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=5,
                        min_child_samples=30,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.7,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=8,
                        verbosity=-1,
                    )
                    model.fit(flat[train], windows.targets[train])
                    estimate = np.clip(model.predict(flat[test]), 0.0, 1.0)
                    truth = windows.targets[test]
                    error = estimate - truth
                    late = truth <= 0.3
                    row = {
                        "representation": representation,
                        "seed": seed,
                        "condition": condition,
                        "test_bearing": test_bearing,
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "mae": float(np.mean(np.abs(error))),
                        "late_rmse": float(np.sqrt(np.mean(error[late] ** 2))),
                        "test_windows": int(np.sum(test)),
                    }
                    folds.append(row)
                    for global_index, value in zip(
                        np.flatnonzero(test), estimate, strict=True
                    ):
                        predictions.append(
                            {
                                "representation": representation,
                                "seed": seed,
                                "condition": condition,
                                "bearing_id": test_bearing,
                                "file_idx": int(windows.file_indices[global_index]),
                                "truth": float(windows.targets[global_index]),
                                "prediction": float(value),
                            }
                        )
                    print(json.dumps(row), flush=True)
    _write_csv(args.output / "fold_scores.csv", folds)
    _write_csv(args.output / "predictions.csv", predictions)
    aggregate = []
    for representation in args.representations:
        selected = [row for row in folds if row["representation"] == representation]
        aggregate.append(
            {
                "representation": representation,
                "macro_rmse": float(np.mean([row["rmse"] for row in selected])),
                "macro_mae": float(np.mean([row["mae"] for row in selected])),
                "macro_late_rmse": float(
                    np.mean([row["late_rmse"] for row in selected])
                ),
            }
        )
    summary = {
        "protocol": "X2 tree robustness check",
        "model": "LightGBM fixed causal-window configuration",
        "calibration": args.calibration,
        "window_size": args.window_size,
        "seeds": args.seeds,
        "seconds": time.perf_counter() - started,
        "aggregate": aggregate,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
