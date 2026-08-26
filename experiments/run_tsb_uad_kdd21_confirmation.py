"""Freeze and execute the D1 cross-source KDD21/UCR confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT.parent / "third_party"
sys.path.insert(0, str(ROOT / "src"))

from hcrd import hcrd_area_anomaly_score  # noqa: E402

METHODS = [
    "IFOREST",
    "LOF",
    "MP",
    "NORMA",
    "IFOREST1",
    "HBOS",
    "OCSVM",
    "PCA",
    "AE",
    "CNN",
    "LSTM",
    "POLY",
]
NON_NEURAL = [
    "IFOREST",
    "LOF",
    "MP",
    "NORMA",
    "IFOREST1",
    "HBOS",
    "OCSVM",
    "PCA",
    "POLY",
]
ARCHIVE_SHA256 = "ff4aa83a5a111835d410d962152e8dbebcda1039b778bae45b6b9c3f46dd49a1"
TSB_UAD_COMMIT = "313f0fdeba14292b9db4e1aa94c74a983a25de31"
ACCURACY_TABLE_SHA256 = "c86cb2cec271a5346e116b00c012376024c5af44897ee1f119d9f8834cfe3534"
TUNING_LIST_SHA256 = "7bf24d2ef834bb39ddd1c8c2b02c177339921dce694354e258ed2b77d8d5cd1c"
EVALUATION_LIST_SHA256 = "6f4e1d4ddbde195f9687f2c5a951b51faff54dda4b658d85f0c07e3e2879615a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_series(path: Path, *, header: int | None) -> np.ndarray:
    values = pd.read_csv(path, header=header).dropna().to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected two columns in {path}")
    return values


def first_anomaly(values: np.ndarray) -> int:
    indices = np.flatnonzero(values[:, 1])
    return int(indices[0]) if indices.size else -1


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    tsb_data = THIRD_PARTY / "TSB-AD-U-data" / "TSB-AD-U"
    kdd_data = THIRD_PARTY / "TSB-UAD-Public-data" / "KDD21"
    file_lists = THIRD_PARTY / "TSB-AD" / "Datasets" / "File_List"
    accuracy_path = (
        THIRD_PARTY
        / "TSB-UAD"
        / "result"
        / "accuracy_table"
        / "mergedTable_AUC_PR.csv"
    )
    accuracy = pd.read_csv(accuracy_path)
    accuracy = accuracy[accuracy["dataset"] == "KDD21"]
    accuracy = accuracy.assign(
        content_stem=accuracy["filename"].map(lambda value: Path(str(value)).stem)
    )
    used = set(pd.read_csv(file_lists / "TSB-AD-U-Tuning.csv").iloc[:, 0])
    used.update(pd.read_csv(file_lists / "TSB-AD-U-Eva.csv").iloc[:, 0])

    candidates: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
    for path in sorted(kdd_data.glob("*.out")):
        values = read_series(path, header=None)
        candidates.setdefault((len(values), first_anomaly(values)), []).append(
            (path.name, values)
        )

    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for path in sorted(tsb_data.glob("*_UCR_*.csv")):
        if path.name in used:
            continue
        values = read_series(path, header=0)
        matches: list[tuple[str, float]] = []
        for name, reference in candidates.get(
            (len(values), first_anomaly(values)), []
        ):
            if not np.array_equal(values[:, 1], reference[:, 1]):
                continue
            maximum_error = float(np.max(np.abs(values[:, 0] - reference[:, 0])))
            if maximum_error <= 1e-10:
                matches.append((name, maximum_error))
        if len(matches) != 1:
            excluded.append(
                {
                    "tsb_ad_file": path.name,
                    "reason": "nonunique_or_missing_content_match",
                    "matching_tsb_uad_files": "|".join(name for name, _ in matches),
                }
            )
            continue
        kdd_name, maximum_error = matches[0]
        official = accuracy[accuracy["content_stem"] == Path(kdd_name).stem]
        if len(official) != 1:
            raise RuntimeError(f"missing unique official result for {kdd_name}")
        item = official.iloc[0]
        rows.append(
            {
                "tsb_ad_file": path.name,
                "tsb_uad_file": kdd_name,
                "point_anom": int(item["point_anom"]),
                "type_an": str(item["type_an"]),
                "length": int(len(values)),
                "maximum_signal_match_error": maximum_error,
                **{method: float(item[method]) for method in METHODS},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def bootstrap_interval(
    difference: np.ndarray, *, draws: int = 20_000, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 1_000):
        stop = min(start + 1_000, draws)
        indices = rng.integers(0, difference.size, size=(stop - start, difference.size))
        means[start:stop] = difference[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    point = metrics[metrics["point_anom"] == 1]
    rows: list[dict[str, object]] = []
    for index, method in enumerate(METHODS):
        difference = point["hcrd_auc_pr"].to_numpy() - point[method].to_numpy()
        low, high = bootstrap_interval(difference, seed=20260826 + index)
        rows.append(
            {
                "baseline": method,
                "series": int(len(point)),
                "hcrd_mean_auc_pr": float(point["hcrd_auc_pr"].mean()),
                "baseline_mean_auc_pr": float(point[method].mean()),
                "mean_difference": float(difference.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "wins": int(np.sum(difference > 1e-12)),
                "ties": int(np.sum(np.abs(difference) <= 1e-12)),
                "losses": int(np.sum(difference < -1e-12)),
            }
        )
    return pd.DataFrame(rows)


def input_integrity() -> dict[str, str]:
    file_lists = THIRD_PARTY / "TSB-AD" / "Datasets" / "File_List"
    accuracy_path = (
        THIRD_PARTY
        / "TSB-UAD"
        / "result"
        / "accuracy_table"
        / "mergedTable_AUC_PR.csv"
    )
    return {
        "archive_sha256": sha256(THIRD_PARTY / "TSB-UAD-Public.zip"),
        "accuracy_table_sha256": sha256(accuracy_path),
        "tuning_list_sha256": sha256(file_lists / "TSB-AD-U-Tuning.csv"),
        "evaluation_list_sha256": sha256(file_lists / "TSB-AD-U-Eva.csv"),
    }


def freeze(output: Path) -> None:
    integrity = input_integrity()
    expected = {
        "archive_sha256": ARCHIVE_SHA256,
        "accuracy_table_sha256": ACCURACY_TABLE_SHA256,
        "tuning_list_sha256": TUNING_LIST_SHA256,
        "evaluation_list_sha256": EVALUATION_LIST_SHA256,
    }
    if integrity != expected:
        raise RuntimeError(f"input integrity mismatch: {integrity}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=THIRD_PARTY / "TSB-UAD",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != TSB_UAD_COMMIT:
        raise RuntimeError(f"unexpected TSB-UAD commit: {commit}")

    manifest, excluded = build_manifest()
    point = manifest[manifest["point_anom"] == 1]
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "population_manifest.csv", index=False)
    excluded.to_csv(output / "excluded_matches.csv", index=False)
    if len(point) < 10:
        screen = {
            "status": "aborted_before_hcrd_population_too_small",
            "matched_series": int(len(manifest)),
            "point_series": int(len(point)),
            "required_point_series": 10,
            "excluded_matches": int(len(excluded)),
            **integrity,
            "tsb_uad_commit": commit,
            "population_manifest_sha256": sha256(
                output / "population_manifest.csv"
            ),
        }
        (output / "population_screen.json").write_text(
            json.dumps(screen, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(screen, indent=2))
        return
    baseline_means = point[METHODS].mean().sort_values(ascending=False)
    primary = baseline_means[NON_NEURAL].idxmax()
    manifest.to_csv(output / "confirmation_manifest.csv", index=False)

    paths = {
        "experiments/run_tsb_uad_kdd21_confirmation.py": Path(__file__).resolve(),
        "src/hcrd/anomaly.py": ROOT / "src" / "hcrd" / "anomaly.py",
        "src/hcrd/energy.py": ROOT / "src" / "hcrd" / "energy.py",
        "docs/tsb_uad_kdd21_d1_protocol.md": ROOT
        / "docs"
        / "tsb_uad_kdd21_d1_protocol.md",
    }
    payload = {
        "status": "frozen_before_kdd21_confirmation_execution",
        "selected_candidate": "hcrd_L8_max",
        "primary_stratum": "official KDD21 point_anom == 1",
        "primary_metric": "mean per-series AUC-PR",
        "primary_comparator": primary,
        "matched_series": int(len(manifest)),
        "point_series": int(len(point)),
        "excluded_matches": int(len(excluded)),
        "primary_comparator_mean_before_hcrd": float(baseline_means[primary]),
        **integrity,
        "tsb_uad_commit": commit,
        "confirmation_manifest_sha256": sha256(output / "confirmation_manifest.csv"),
        "implementation_sha256": {key: sha256(path) for key, path in paths.items()},
    }
    (output / "frozen_configuration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def evaluate(output: Path) -> None:
    frozen_path = output / "frozen_configuration.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_kdd21_confirmation_execution":
        raise RuntimeError("unexpected freeze status")
    paths = {
        "experiments/run_tsb_uad_kdd21_confirmation.py": Path(__file__).resolve(),
        "src/hcrd/anomaly.py": ROOT / "src" / "hcrd" / "anomaly.py",
        "src/hcrd/energy.py": ROOT / "src" / "hcrd" / "energy.py",
        "docs/tsb_uad_kdd21_d1_protocol.md": ROOT
        / "docs"
        / "tsb_uad_kdd21_d1_protocol.md",
    }
    for key, path in paths.items():
        if frozen["implementation_sha256"].get(key) != sha256(path):
            raise RuntimeError(f"implementation changed after freeze: {key}")
    manifest_path = output / "confirmation_manifest.csv"
    if frozen["confirmation_manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError("confirmation manifest changed after freeze")

    manifest = pd.read_csv(manifest_path)
    data_root = THIRD_PARTY / "TSB-AD-U-data" / "TSB-AD-U"
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for _, item in manifest.iterrows():
        values = read_series(data_root / item["tsb_ad_file"], header=0)
        detector_started = time.perf_counter()
        score = hcrd_area_anomaly_score(
            values[:, 0], max_levels=8, aggregation="max"
        )
        rows.append(
            {
                **item.to_dict(),
                "hcrd_auc_pr": float(
                    average_precision_score(values[:, 1].astype(int), score)
                ),
                "detector_seconds": time.perf_counter() - detector_started,
            }
        )

    metrics = pd.DataFrame(rows)
    comparisons = comparison_table(metrics)
    primary = comparisons[
        comparisons["baseline"] == frozen["primary_comparator"]
    ].iloc[0]
    summary = {
        "status": "cross_source_kdd21_confirmation_executed_after_freeze",
        "matched_series": int(len(metrics)),
        "point_series": int(metrics["point_anom"].sum()),
        "point_hcrd_mean_auc_pr": float(primary["hcrd_mean_auc_pr"]),
        "point_primary_comparator": frozen["primary_comparator"],
        "point_primary_comparator_mean_auc_pr": float(
            primary["baseline_mean_auc_pr"]
        ),
        "point_primary_difference": float(primary["mean_difference"]),
        "point_primary_bootstrap_95_low": float(primary["bootstrap_95_low"]),
        "point_primary_bootstrap_95_high": float(primary["bootstrap_95_high"]),
        "point_primary_wins_ties_losses": [
            int(primary["wins"]),
            int(primary["ties"]),
            int(primary["losses"]),
        ],
        "primary_success": bool(primary["bootstrap_95_low"] > 0.0),
        "detector_seconds": float(metrics["detector_seconds"].sum()),
        "wall_seconds": time.perf_counter() - started,
        "frozen_configuration_sha256": sha256(frozen_path),
    }
    metrics.to_csv(output / "confirmation_metrics.csv.gz", index=False)
    comparisons.to_csv(output / "point_comparisons.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "tsb_uad_kdd21_d1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_only:
        freeze(args.output_dir)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
