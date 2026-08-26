#!/usr/bin/env python3
"""Equal-dimensional E2 Gaussian-control sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

try:
    from experiments.run_lcms_eic_e1 import _holm, _weighted_ap, _weighted_ap_preparation
except ModuleNotFoundError:
    from run_lcms_eic_e1 import _holm, _weighted_ap, _weighted_ap_preparation
from hcrd.lcms_controls import gaussian_derivative_control


SEED = 20260826
REPRESENTATIONS = ("qscore", "domain_q", "hcrd_8_q", "gaussian_2847_q")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_control(
    source_dir: Path, output_path: Path, *, block_size: int = 64
) -> None:
    cube = np.load(source_dir / "per_file_domain.npy", mmap_mode="r")
    qscore = np.asarray(np.load(source_dir / "qscore.npy", mmap_mode="r"))
    file_count, feature_count, width = cube.shape
    if width != 111 or qscore.shape != (feature_count, 2):
        raise ValueError("E2 cache schema mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(feature_count, 2847),
    )
    for start in range(0, feature_count, block_size):
        stop = min(feature_count, start + block_size)
        block = np.asarray(cube[:, start:stop, :75])
        control = gaussian_derivative_control(block)
        available = np.mean(np.isfinite(block[:, :, 0]), axis=0)[:, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(control, axis=0)
            q90 = np.nanquantile(control, 0.9, axis=0)
            maximum = np.nanmax(control, axis=0)
        values = np.concatenate(
            [median, q90, maximum, available, qscore[start:stop]], axis=1
        )
        aggregate[start:stop] = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
        print(f"{source_dir.name}: {stop}/{feature_count}", flush=True)
    aggregate.flush()


def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
    )


def _load(
    source_dir: Path, control_path: Path
) -> tuple[NDArray[np.int8], dict[str, NDArray[np.float64]]]:
    labels = np.asarray(np.load(source_dir / "labels.npy"), dtype=np.int8)
    keep = labels >= 0
    arrays = {
        "qscore": np.asarray(np.load(source_dir / "qscore.npy", mmap_mode="r")[keep]),
        "domain_q": np.asarray(
            np.load(source_dir / "domain_q.npy", mmap_mode="r")[keep]
        ),
        "hcrd_8_q": np.asarray(
            np.load(source_dir / "hcrd_8_q.npy", mmap_mode="r")[keep]
        ),
        "gaussian_2847_q": np.asarray(np.load(control_path, mmap_mode="r")[keep]),
    }
    return labels[keep], arrays


def _paired_difference(
    labels: NDArray[np.int8],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    left_preparation = _weighted_ap_preparation(left)
    right_preparation = _weighted_ap_preparation(right)
    rng = np.random.default_rng(seed)
    probability = np.full(labels.size, 1.0 / labels.size)
    differences = np.empty(replicates)
    for replicate in range(replicates):
        weights = rng.multinomial(labels.size, probability).astype(float)
        differences[replicate] = _weighted_ap(labels, weights, *left_preparation) - _weighted_ap(
            labels, weights, *right_preparation
        )
    p_value = min(
        1.0,
        2.0
        * min(
            (np.sum(differences <= 0.0) + 1.0) / (replicates + 1.0),
            (np.sum(differences >= 0.0) + 1.0) / (replicates + 1.0),
        ),
    )
    return {
        "ap_difference": float(
            average_precision_score(labels, left)
            - average_precision_score(labels, right)
        ),
        "bootstrap_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--falkor-dir", type=Path, default=PROJECT / "results/ms_metrics_e2/falkor"
    )
    parser.add_argument(
        "--mesoscope-dir",
        type=Path,
        default=PROJECT / "results/ms_metrics_e2/mesoscope",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results/ms_metrics_e2_matched_capacity",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = {"falkor": args.falkor_dir, "mesoscope": args.mesoscope_dir}
    control_paths = {
        name: args.output_dir / name / "gaussian_2847_q.npy" for name in directories
    }
    for name, directory in directories.items():
        if args.rebuild or not control_paths[name].exists():
            _build_control(directory, control_paths[name])

    datasets = {
        name: _load(directory, control_paths[name])
        for name, directory in directories.items()
    }
    results: dict[str, object] = {}
    audit_p: dict[str, float] = {}
    prediction_rows: list[dict[str, object]] = []
    for direction_index, (source, target) in enumerate(
        (("falkor", "mesoscope"), ("mesoscope", "falkor"))
    ):
        source_y, source_x = datasets[source]
        target_y, target_x = datasets[target]
        scores: dict[str, NDArray[np.float64]] = {}
        metrics: dict[str, dict[str, float]] = {}
        for name in REPRESENTATIONS:
            model = _model()
            model.fit(source_x[name], source_y)
            scores[name] = model.predict_proba(target_x[name])[:, 1]
            metrics[name] = {
                "average_precision": float(
                    average_precision_score(target_y, scores[name])
                ),
                "roc_auc": float(roc_auc_score(target_y, scores[name])),
                "width": int(source_x[name].shape[1]),
            }
            joblib.dump(
                model, args.output_dir / f"model_{source}_to_{target}_{name}.joblib"
            )
        audit = _paired_difference(
            target_y,
            scores["hcrd_8_q"],
            scores["gaussian_2847_q"],
            args.bootstrap,
            SEED + direction_index,
        )
        domain = _paired_difference(
            target_y,
            scores["hcrd_8_q"],
            scores["domain_q"],
            args.bootstrap,
            SEED + 100 + direction_index,
        )
        key = f"{source}_to_{target}"
        audit_p[key] = float(audit["two_sided_bootstrap_p"])
        results[key] = {
            "metrics": metrics,
            "hcrd_8_q_minus_matched_gaussian": audit,
            "hcrd_8_q_minus_domain_q": domain,
        }
        for row_index, label in enumerate(target_y):
            prediction_rows.append(
                {
                    "direction": key,
                    "target_row": row_index,
                    "label": int(label),
                    **{f"score_{name}": float(value[row_index]) for name, value in scores.items()},
                }
            )

    adjusted = _holm(audit_p)
    for key in results:
        results[key]["hcrd_8_q_minus_matched_gaussian"][
            "holm_across_directions_p"
        ] = adjusted[key]
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    summary = {
        "protocol": "supplementary-e2-matched-capacity-v1",
        "protocol_path": "docs/ms_metrics_e2_matched_capacity.md",
        "post_confirmation_sensitivity": True,
        "conditional_estimand": "target-feature performance conditional on each fitted source model",
        "excluded_uncertainty": [
            "source-model refitting",
            "compound/adduct clustering",
            "file-level acquisition clustering",
        ],
        "per_file_width": {"hcrd_8": 948, "gaussian_control": 948},
        "aggregated_width": {"hcrd_8_q": 2847, "gaussian_2847_q": 2847},
        "control_feature_hashes": {
            name: _sha256(path) for name, path in control_paths.items()
        },
        "bootstrap_replicates": args.bootstrap,
        "seed": SEED,
        "directions": results,
    }
    (args.output_dir / "matched_capacity_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
