"""Audit whether U1 discovery supports unlocking the confirmation half."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "results" / "ucr_u1" / "discovery" / "per_dataset"
STRUCTURE = (
    PROJECT
    / "results"
    / "ucr_u1_d3_structure"
    / "discovery"
    / "per_dataset"
)
TRAINING_CV = (
    PROJECT / "results" / "ucr_u1_gate" / "discovery" / "training_cv.csv"
)
OUTPUT = PROJECT / "results" / "ucr_u1" / "discovery_gate_analysis.json"
REPORT = PROJECT / "reports" / "ucr_u1_discovery.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> list[dict[str, object]]:
    with TRAINING_CV.open(newline="", encoding="utf-8") as handle:
        training = {row["dataset"]: row for row in csv.DictReader(handle)}
    rows = []
    for path in sorted(BASE.glob("*.json")):
        base = json.loads(path.read_text(encoding="utf-8"))
        name = str(base["dataset"])
        structure_path = STRUCTURE / f"{name}.json"
        if not structure_path.exists() or name not in training:
            raise RuntimeError(f"incomplete U1 discovery evidence for {name}")
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        hcrd = structure["metrics"]["hcrd_structure_extratrees_cv"]
        raw = float(base["metrics"]["raw_minirocket"]["accuracy"])
        wavelet = float(base["metrics"]["wavelet_minirocket"]["accuracy"])
        hcrd_accuracy = float(hcrd["accuracy"])
        raw_cv = training[name]["raw_cv_accuracy"]
        wavelet_cv = training[name]["wavelet_cv_accuracy"]
        hcrd_cv = hcrd["best_cv_accuracy"]
        finite_cv = (
            raw_cv not in {"", None}
            and wavelet_cv not in {"", None}
            and hcrd_cv is not None
        )
        control = max(raw, wavelet)
        row = {
            "dataset": name,
            "raw_accuracy": raw,
            "wavelet_accuracy": wavelet,
            "hcrd_structure_accuracy": hcrd_accuracy,
            "hcrd_minus_best_control": hcrd_accuracy - control,
            "finite_training_cv": bool(finite_cv),
        }
        if finite_cv:
            row["training_cv_margin"] = float(hcrd_cv) - max(
                float(raw_cv), float(wavelet_cv)
            )
        rows.append(row)
    if len(rows) != 57:
        raise RuntimeError(f"expected 57 discovery datasets, found {len(rows)}")
    return rows


def _best_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    finite = [row for row in rows if bool(row["finite_training_cv"])]
    margins = np.asarray([float(row["training_cv_margin"]) for row in finite])
    gains = np.asarray([float(row["hcrd_minus_best_control"]) for row in finite])
    candidates = []
    for threshold in np.unique(
        np.r_[margins - 1e-9, margins + 1e-9]
    ):
        selected = margins >= threshold
        if int(np.count_nonzero(selected)) < 10:
            continue
        selected_gains = gains[selected]
        candidates.append(
            {
                "threshold": float(threshold),
                "selected_datasets": int(selected_gains.size),
                "mean_selected_accuracy_difference": float(
                    np.mean(selected_gains)
                ),
                "wins": int(np.count_nonzero(selected_gains > 0.0)),
                "ties": int(np.count_nonzero(selected_gains == 0.0)),
                "losses": int(np.count_nonzero(selected_gains < 0.0)),
                "datasets": [
                    str(row["dataset"])
                    for row, keep in zip(finite, selected, strict=True)
                    if keep
                ],
            }
        )
    return max(
        candidates,
        key=lambda item: (
            float(item["mean_selected_accuracy_difference"]),
            -int(item["selected_datasets"]),
            float(item["threshold"]),
        ),
    )


def main() -> None:
    rows = _load()
    gains = np.asarray(
        [float(row["hcrd_minus_best_control"]) for row in rows]
    )
    finite = [row for row in rows if bool(row["finite_training_cv"])]
    correlation = float(
        np.corrcoef(
            [float(row["training_cv_margin"]) for row in finite],
            [float(row["hcrd_minus_best_control"]) for row in finite],
        )[0, 1]
    )
    gate = _best_gate(rows)
    result = {
        "status": "discovery-only; confirmation remains locked",
        "datasets": len(rows),
        "finite_training_cv_datasets": len(finite),
        "hcrd_structure_vs_best_raw_or_wavelet": {
            "wins": int(np.count_nonzero(gains > 0.0)),
            "ties": int(np.count_nonzero(gains == 0.0)),
            "losses": int(np.count_nonzero(gains < 0.0)),
            "mean_accuracy_difference": float(np.mean(gains)),
            "median_accuracy_difference": float(np.median(gains)),
        },
        "training_cv_margin_correlation_with_test_gain": correlation,
        "best_gate_requiring_at_least_10_datasets": gate,
        "confirmation_unlocked": bool(
            float(gate["mean_selected_accuracy_difference"]) > 0.0
        ),
        "decision": (
            "No positive discovery subgroup exists under the fixed D4 gate; "
            "do not inspect the confirmation outcomes."
        ),
        "source_sha256": {
            "training_cv": sha256(TRAINING_CV),
            "base_summary": sha256(BASE.parent / "summary.json"),
            "structure_summary": sha256(STRUCTURE.parent / "summary.json"),
        },
        "per_dataset": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    comparison = result["hcrd_structure_vs_best_raw_or_wavelet"]
    lines = [
        "# UCR U1 discovery decision",
        "",
        "Author: Saveliy Baturin, Independent Researcher",
        "",
        "Status: discovery-only. The 48 confirmation datasets remain uninspected.",
        "",
        "The complete HCRD structure collection with training-only ExtraTrees "
        "selection was compared with the better of raw and db4-wavelet "
        "MiniRocket on 57 discovery datasets.",
        "",
        f"- wins/ties/losses: {comparison['wins']}/{comparison['ties']}/"
        f"{comparison['losses']};",
        f"- mean accuracy difference: "
        f"{comparison['mean_accuracy_difference']:.4f};",
        f"- correlation of the train-CV margin with test gain: "
        f"{correlation:.4f};",
        f"- best admissible gate selected {gate['selected_datasets']} datasets "
        f"but still had mean HCRD-control difference "
        f"{gate['mean_selected_accuracy_difference']:.4f} "
        f"({gate['wins']}/{gate['ties']}/{gate['losses']} "
        "wins/ties/losses).",
        "",
        "No positive subgroup rule was available to freeze. Confirmation "
        "therefore remains locked. This prevents selecting a task class by "
        "looking at the held-out benchmark half.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_dataset"}, indent=2))


if __name__ == "__main__":
    main()
