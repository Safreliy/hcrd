"""Run the prospectively frozen WSD structural effect-modifier audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
DATA_ROOT = REPOSITORY_ROOT / "third_party" / "TSB-AD-U-data" / "TSB-AD-U"
OUTPUT = ROOT / "results" / "wsd_sparse_transient_c2"

sys.path.insert(0, str(ROOT / "src"))

from hcrd.structural_class import aggregate_event_chord_morphology  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values."""

    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def permutation_spearman_p(
    feature: np.ndarray,
    outcome: np.ndarray,
    *,
    alternative: str,
    permutations: int = 100_000,
    seed: int = 20260825,
    batch_size: int = 2_000,
) -> tuple[float, float]:
    """Spearman rho and deterministic Monte-Carlo permutation p-value."""

    x = np.asarray(feature, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3:
        raise ValueError("feature and outcome must be aligned one-dimensional arrays")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("unsupported alternative")
    observed = float(spearmanr(x, y).statistic)
    if not np.isfinite(observed):
        raise ValueError("Spearman correlation is undefined")
    x_rank = rankdata(x).astype(float)
    y_rank = rankdata(y).astype(float)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        raise ValueError("rank variance must be positive")
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, permutations, batch_size):
        count = min(batch_size, permutations - start)
        permuted = np.vstack([rng.permutation(y_centered) for _ in range(count)])
        rho = (permuted @ x_centered) / denominator
        if alternative == "greater":
            exceedances += int(np.sum(rho >= observed - 1e-15))
        elif alternative == "less":
            exceedances += int(np.sum(rho <= observed + 1e-15))
        else:
            exceedances += int(np.sum(np.abs(rho) >= abs(observed) - 1e-15))
    return observed, float((exceedances + 1) / (permutations + 1))


def _read_ok_metrics(method: str) -> pd.DataFrame:
    rows = []
    for path in sorted((OUTPUT / "metrics" / method).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    freeze = json.loads(
        (OUTPUT / "structural_audit_frozen.json").read_text(encoding="utf-8")
    )
    implementation = ROOT / "src" / "hcrd" / "structural_class.py"
    if sha256(implementation) != freeze["implementation_sha256"]:
        raise RuntimeError("the frozen structural descriptor implementation changed")
    comparator_payload = json.loads(
        (OUTPUT / "comparator_frozen.json").read_text(encoding="utf-8")
    )
    comparator = str(comparator_payload["primary_comparator"])
    manifest = pd.read_csv(OUTPUT / "population_manifest.csv")
    primary = manifest[manifest["primary_sparse_transient"]].copy()

    hcrd = _read_ok_metrics("HCRD")[["file", "vus_pr"]].rename(
        columns={"vus_pr": "HCRD_vus_pr"}
    )
    baseline = _read_ok_metrics(comparator)[["file", "vus_pr"]].rename(
        columns={"vus_pr": f"{comparator}_vus_pr"}
    )
    primary = primary.merge(hcrd, on="file", validate="1:1").merge(
        baseline, on="file", validate="1:1"
    )
    if len(primary) != 71:
        raise RuntimeError(f"expected 71 complete primary series, found {len(primary)}")

    morphology_rows: list[dict[str, object]] = []
    for filename in primary["file"].astype(str):
        frame = pd.read_csv(DATA_ROOT / filename).dropna()
        descriptors = aggregate_event_chord_morphology(
            frame["Data"].to_numpy(dtype=float),
            frame["Label"].to_numpy(dtype=int),
        )
        morphology_rows.append({"file": filename, **descriptors})
    morphology = pd.DataFrame(morphology_rows)
    morphology.to_csv(OUTPUT / "structural_morphology.csv", index=False)
    audit = primary.merge(morphology, on="file", validate="1:1")
    audit["hcrd_minus_comparator_vus_pr"] = (
        audit["HCRD_vus_pr"] - audit[f"{comparator}_vus_pr"]
    )
    audit.to_csv(OUTPUT / "structural_audit_series.csv", index=False)

    hypotheses = [
        ("median_sign_coherence", "greater", True),
        ("median_peak_to_background_mad", "greater", True),
        ("median_curvature_contrast", "greater", True),
        ("median_duration_fraction", "less", True),
        ("median_shape_concentration", "two-sided", False),
    ]
    outcome = audit["hcrd_minus_comparator_vus_pr"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for index, (feature, alternative, confirmatory) in enumerate(hypotheses):
        rho, p_value = permutation_spearman_p(
            audit[feature].to_numpy(dtype=float),
            outcome,
            alternative=alternative,
            seed=20260825 + index,
        )
        rows.append(
            {
                "feature": feature,
                "alternative": alternative,
                "confirmatory_directional": confirmatory,
                "spearman_rho": rho,
                "permutation_p": p_value,
                "permutations": 100_000,
            }
        )
    associations = pd.DataFrame(rows)
    mask = associations["confirmatory_directional"]
    associations.loc[mask, "holm_permutation_p"] = holm_adjust(
        associations.loc[mask, "permutation_p"].astype(float).tolist()
    )
    associations.to_csv(OUTPUT / "structural_associations.csv", index=False)
    payload = {
        "status": "complete",
        "role": freeze["role"],
        "primary_series": int(len(audit)),
        "frozen_comparator": comparator,
        "outcome_mean": float(outcome.mean()),
        "associations": associations.where(pd.notna(associations), None).to_dict(
            orient="records"
        ),
        "interpretation_guardrail": (
            "These label-assisted descriptors audit a proposed mechanism; they are "
            "not detector inputs and do not define the primary class post hoc."
        ),
    }
    (OUTPUT / "structural_audit_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
