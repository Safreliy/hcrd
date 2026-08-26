"""Freeze prospective structural effect modifiers before WSD HCRD execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "wsd_sparse_transient_c2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    hcrd_dir = OUTPUT / "metrics" / "HCRD"
    if hcrd_dir.exists() and any(hcrd_dir.glob("*.json")):
        raise RuntimeError("refusing to freeze structural audit after WSD HCRD")
    implementation = ROOT / "src" / "hcrd" / "structural_class.py"
    payload = {
        "status": "structural_effect_modifiers_frozen_before_WSD_HCRD",
        "role": "secondary mechanistic audit; not a primary endpoint or class-selection rule",
        "outcome": "per-series HCRD VUS-PR minus the frozen primary comparator",
        "fixed_hypotheses": {
            "median_sign_coherence": "positive association",
            "median_peak_to_background_mad": "positive association",
            "median_curvature_contrast": "positive association",
            "median_duration_fraction": "negative association",
            "median_shape_concentration": "two-sided exploratory",
        },
        "statistics": "Spearman rho with permutation p-values and Holm correction over the four directional hypotheses",
        "no_threshold_search": True,
        "implementation_sha256": sha256(implementation),
        "population_freeze_sha256": sha256(OUTPUT / "frozen_population.json"),
    }
    path = OUTPUT / "structural_audit_frozen.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
