"""XJTU-SY X1 LOBO comparison: standard versus HCRD-energy features."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

METADATA_COLUMNS = {
    "condition",
    "bearing_id",
    "filename",
    "file_idx",
    "total_files",
    "rul",
    "rul_twostage",
    "rul_normalized",
}
KEY_COLUMNS = ["condition", "bearing_id", "filename", "file_idx"]
COMPACT_ENERGY_MEASURES = (
    "log1p_polygon_area",
    "log1p_quadratic_energy",
    "area_concentration",
    "weighted_shape_factor",
)


@dataclass(frozen=True)
class WindowData:
    values: np.ndarray
    targets: np.ndarray
    bearings: np.ndarray
    conditions: np.ndarray
    file_indices: np.ndarray


def _normalise_per_bearing(
    frame: pd.DataFrame, feature_columns: list[str], calibration: str
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for _, group in frame.groupby("bearing_id", sort=True):
        group = group.sort_values("file_idx").copy()
        group[feature_columns] = group[feature_columns].astype(np.float64)
        count = len(group)
        baseline_count = (
            max(2, count // 5) if calibration == "fraction20" else min(20, count)
        )
        values = group[feature_columns].to_numpy(dtype=np.float64)
        baseline = values[:baseline_count]
        centre = np.mean(baseline, axis=0)
        scale = np.std(baseline, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalised = np.clip((values - centre) / scale, -20.0, 20.0)
        group.loc[:, feature_columns] = normalised
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def create_causal_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    window_size: int = 10,
    calibration: str = "fraction20",
) -> WindowData:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if calibration not in {"fraction20", "first20"}:
        raise ValueError("unknown calibration")
    frame = _normalise_per_bearing(frame, feature_columns, calibration)
    windows: list[np.ndarray] = []
    targets: list[float] = []
    bearings: list[str] = []
    conditions: list[str] = []
    file_indices: list[int] = []
    for bearing_id, group in frame.groupby("bearing_id", sort=True):
        group = group.sort_values("file_idx")
        values = group[feature_columns].to_numpy(dtype=np.float32)
        count = len(group)
        target = 1.0 - np.arange(count, dtype=np.float32) / max(1, count - 1)
        for stop in range(window_size - 1, count):
            windows.append(values[stop - window_size + 1 : stop + 1])
            targets.append(float(target[stop]))
            bearings.append(str(bearing_id))
            conditions.append(str(group.iloc[stop]["condition"]))
            file_indices.append(int(group.iloc[stop]["file_idx"]))
    return WindowData(
        values=np.stack(windows),
        targets=np.asarray(targets, dtype=np.float32),
        bearings=np.asarray(bearings),
        conditions=np.asarray(conditions),
        file_indices=np.asarray(file_indices, dtype=np.int64),
    )


def add_compact_energy_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate both sensor axes into a low-dimensional envelope summary.

    This X2 development representation deliberately excludes spectral HCRD
    coordinates and redundant amplitude/triangle summaries after X1 showed
    that an unfiltered 288-dimensional concatenation was unstable.
    """

    frame = frame.copy()
    names: list[str] = []
    for level in range(6):
        for measure in COMPACT_ENERGY_MEASURES:
            horizontal = f"h_env_level_{level}_{measure}"
            vertical = f"v_env_level_{level}_{measure}"
            if horizontal not in frame or vertical not in frame:
                raise KeyError(f"missing compact HCRD inputs for level {level}")
            mean_name = f"compact_env_level_{level}_{measure}_axis_mean"
            max_name = f"compact_env_level_{level}_{measure}_axis_max"
            frame[mean_name] = 0.5 * (frame[horizontal] + frame[vertical])
            frame[max_name] = np.maximum(frame[horizontal], frame[vertical])
            names.extend([mean_name, max_name])
    return frame, names


def polygon_mass_feature_names() -> list[str]:
    """Six direct envelope-area coordinates for the middle hierarchy levels."""

    return [
        f"{channel}_env_level_{level}_log1p_polygon_area"
        for channel in ("h", "v")
        for level in (2, 3, 4)
    ]


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_model(input_features: int):
    import torch
    from torch import nn

    class FeatureSequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(input_features, 32), nn.ReLU()
            )
            self.lstm = nn.LSTM(
                32, 16, batch_first=True, bidirectional=True
            )
            self.head = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
            )

        def forward(self, values):
            projected = self.projection(values)
            encoded, _ = self.lstm(projected)
            return self.head(encoded[:, -1]).squeeze(-1)

    return FeatureSequenceModel()


def _fit_with_early_stopping(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
) -> tuple[int, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(seed)
    model = _make_model(train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.HuberLoss(delta=0.08)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    validation_values = torch.from_numpy(validation_x)
    validation_targets = torch.from_numpy(validation_y)
    best_loss = np.inf
    best_epoch = 1
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_values, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_values), batch_targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(validation_values), validation_targets)
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= 7:
                break
    return best_epoch, best_loss


def _fit_and_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(seed)
    model = _make_model(train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.HuberLoss(delta=0.08)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for batch_values, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_values), batch_targets)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(test_x)).numpy()
    return np.clip(predicted, 0.0, 1.0)


def _inner_validation_bearing(test_bearing: str, same_condition: list[str]) -> str:
    test_position = same_condition.index(test_bearing)
    return same_condition[(test_position + 1) % len(same_condition)]


def evaluate_representation(
    windows: WindowData,
    representation: str,
    *,
    seed: int,
    max_epochs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for condition in sorted(set(windows.conditions)):
        condition_bearings = sorted(
            set(windows.bearings[windows.conditions == condition])
        )
        for test_bearing in condition_bearings:
            validation_bearing = _inner_validation_bearing(
                test_bearing, condition_bearings
            )
            test = windows.bearings == test_bearing
            validation = windows.bearings == validation_bearing
            train = (
                (windows.conditions == condition)
                & ~test
                & ~validation
            )
            all_training = (windows.conditions == condition) & ~test
            best_epoch, validation_loss = _fit_with_early_stopping(
                windows.values[train],
                windows.targets[train],
                windows.values[validation],
                windows.targets[validation],
                seed=seed,
                max_epochs=max_epochs,
            )
            predicted = _fit_and_predict(
                windows.values[all_training],
                windows.targets[all_training],
                windows.values[test],
                seed=seed + 10_000,
                epochs=best_epoch,
            )
            truth = windows.targets[test]
            error = predicted - truth
            rmse = float(np.sqrt(np.mean(error**2)))
            mae = float(np.mean(np.abs(error)))
            late = truth <= 0.3
            late_rmse = float(np.sqrt(np.mean(error[late] ** 2)))
            denominator = float(np.sum((truth - np.mean(truth)) ** 2))
            r_squared = 1.0 - float(np.sum(error**2)) / denominator
            fold_rows.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "condition": condition,
                    "test_bearing": test_bearing,
                    "inner_validation_bearing": validation_bearing,
                    "selected_epochs": best_epoch,
                    "inner_validation_huber": validation_loss,
                    "rmse": rmse,
                    "mae": mae,
                    "late_rmse": late_rmse,
                    "r_squared": r_squared,
                    "test_windows": int(np.sum(test)),
                }
            )
            test_indices = np.flatnonzero(test)
            for global_index, estimate in zip(test_indices, predicted, strict=True):
                prediction_rows.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "condition": condition,
                        "bearing_id": test_bearing,
                        "file_idx": int(windows.file_indices[global_index]),
                        "truth": float(windows.targets[global_index]),
                        "prediction": float(estimate),
                    }
                )
            print(
                json.dumps(
                    {
                        "representation": representation,
                        "seed": seed,
                        "bearing": test_bearing,
                        "rmse": rmse,
                        "epochs": best_epoch,
                    }
                ),
                flush=True,
            )
    return fold_rows, prediction_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-features", type=Path, required=True)
    parser.add_argument("--hcrd-features", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "results" / "xjtu_x1"
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=(
            "standard",
            "hcrd",
            "hybrid",
            "hcrd_compact",
            "hybrid_compact",
            "hcrd_mass6",
            "hybrid_mass6",
        ),
        default=("standard", "hcrd", "hybrid"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(9,))
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--calibration", choices=("fraction20", "first20"), default="fraction20"
    )
    args = parser.parse_args()
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    standard = pd.read_csv(args.standard_features)
    energy = pd.read_csv(args.hcrd_features)
    merged = standard.merge(
        energy.drop(columns=["total_files", "rul_normalized"]),
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(standard) or len(merged) != len(energy):
        raise RuntimeError("standard/HCRD feature tables do not align one-to-one")
    merged, compact_energy_columns = add_compact_energy_features(merged)
    polygon_mass_columns = polygon_mass_feature_names()
    standard_columns = [
        column for column in standard.columns if column not in METADATA_COLUMNS
    ]
    energy_columns = [
        column
        for column in energy.columns
        if column not in METADATA_COLUMNS
    ]
    selected_columns = {
        "standard": standard_columns,
        "hcrd": energy_columns,
        "hybrid": standard_columns + energy_columns,
        "hcrd_compact": compact_energy_columns,
        "hybrid_compact": standard_columns + compact_energy_columns,
        "hcrd_mass6": polygon_mass_columns,
        "hybrid_mass6": standard_columns + polygon_mass_columns,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for representation in args.representations:
        windows = create_causal_windows(
            merged,
            selected_columns[representation],
            window_size=args.window_size,
            calibration=args.calibration,
        )
        for seed in args.seeds:
            folds, predictions = evaluate_representation(
                windows,
                representation,
                seed=seed,
                max_epochs=args.max_epochs,
            )
            fold_rows.extend(folds)
            prediction_rows.extend(predictions)
    _write_csv(args.output / "fold_scores.csv", fold_rows)
    _write_csv(args.output / "predictions.csv", prediction_rows)
    aggregate: list[dict[str, object]] = []
    for representation in args.representations:
        selected = [
            row for row in fold_rows if row["representation"] == representation
        ]
        aggregate.append(
            {
                "representation": representation,
                "macro_rmse": float(np.mean([row["rmse"] for row in selected])),
                "macro_mae": float(np.mean([row["mae"] for row in selected])),
                "macro_late_rmse": float(
                    np.mean([row["late_rmse"] for row in selected])
                ),
                "macro_r_squared": float(
                    np.mean([row["r_squared"] for row in selected])
                ),
                "fold_seed_count": len(selected),
            }
        )
    metadata = {
        "protocol": "X1",
        "calibration": args.calibration,
        "window_size": args.window_size,
        "max_epochs": args.max_epochs,
        "seeds": args.seeds,
        "standard_feature_count": len(standard_columns),
        "hcrd_feature_count": len(energy_columns),
        "compact_hcrd_feature_count": len(compact_energy_columns),
        "polygon_mass_feature_count": len(polygon_mass_columns),
        "rows": len(merged),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "seconds": time.perf_counter() - started,
        "aggregate": aggregate,
    }
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
