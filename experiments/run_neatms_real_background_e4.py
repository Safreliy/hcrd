#!/usr/bin/env python3
"""Frozen E4 multiscale lobe-recovery study on real NeatMS TIC backgrounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy
from numpy.typing import NDArray
from pyteomics import mzml
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, grey_opening
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import binomtest

from hcrd import decompose


PROTOCOL = "hcrd-e4-neatms-real-background-v1"
SEED = 20260825
ARCHIVE_MD5 = "47a63b1bcba15d9b5ce6c6e4b6d5537e"
CLASSICAL_METHODS = (
    "gaussian_oracle",
    "savgol_oracle",
    "morph_open_oracle",
    "asls_oracle",
    "wavelet_oracle",
)
METHODS = ("hcrd_l1", "hcrd_l8", *CLASSICAL_METHODS)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(values: NDArray[np.float64]) -> NDArray[np.float64]:
    q05, q95 = np.quantile(values, [0.05, 0.95])
    scale = float(q95 - q05)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("zero TIC block scale")
    return (values - float(np.median(values))) / scale


def _tic(path: Path) -> NDArray[np.float64]:
    values: list[float] = []
    with mzml.MzML(str(path), use_index=False) as reader:
        for spectrum in reader:
            if spectrum.get("ms level") != 1:
                continue
            tic = spectrum.get("total ion current")
            if tic is None:
                tic = float(np.sum(spectrum["intensity array"]))
            values.append(float(tic))
    result = np.asarray(values, dtype=float)
    if result.size < 1032 or not np.all(np.isfinite(result)):
        raise ValueError(f"{path.name} has an invalid MS1 TIC")
    return result


def prepare_backgrounds(data_dir: Path) -> tuple[NDArray[np.float64], pd.DataFrame]:
    backgrounds: list[NDArray[np.float64]] = []
    rows: list[dict[str, object]] = []
    for sample in range(1, 21):
        path = data_dir / f"sample{sample}.mzML"
        values = _tic(path)
        start = (values.size - 1032) // 2
        central = values[start : start + 1032]
        candidates = []
        for block in range(8):
            normalized = _normalise(central[129 * block : 129 * (block + 1)])
            chord = np.linspace(normalized[0], normalized[-1], normalized.size)
            deviation = float(np.median(np.abs(normalized - chord)))
            candidates.append((deviation, block, normalized))
        for rank, (deviation, block, normalized) in enumerate(
            sorted(candidates, key=lambda item: (item[0], item[1]))[:3], start=1
        ):
            backgrounds.append(normalized)
            rows.append(
                {
                    "background_id": len(backgrounds) - 1,
                    "file": path.name,
                    "ms1_scan_count": values.size,
                    "central_start": start,
                    "block_index": block,
                    "affine_deviation_mad": deviation,
                    "affine_rank_within_run": rank,
                    "source_sha256": _hash_file(path),
                }
            )
    result = np.stack(backgrounds)
    if result.shape != (60, 129):
        raise RuntimeError(f"expected 60 backgrounds, found {result.shape}")
    return result, pd.DataFrame(rows)


def _apex_fraction(file: str, block: int, component: int) -> float:
    digest = hashlib.sha256(
        f"{PROTOCOL}|{file}|{block}|{component}".encode("utf-8")
    ).digest()
    return (0.35, 0.50, 0.65)[digest[0] % 3]


def _triangle(
    length: int, left: int, right: int, apex_fraction: float, amplitude: float
) -> NDArray[np.float64]:
    apex = left + apex_fraction * (right - left)
    x = np.arange(length, dtype=float)
    result = np.zeros(length)
    rising = (x >= left) & (x <= apex)
    falling = (x > apex) & (x <= right)
    result[rising] = amplitude * (x[rising] - left) / (apex - left)
    result[falling] = amplitude * (right - x[falling]) / (right - apex)
    return result


def _injection(file: str, block: int, suite: str) -> tuple[NDArray[np.float64], str]:
    if suite == "single":
        fraction = _apex_fraction(file, block, 0)
        return _triangle(129, 32, 96, fraction, 1.0), f"32:96:{fraction}:1"
    if suite != "nested":
        raise ValueError("unknown injection suite")
    definitions = ((16, 112, 1.0), (40, 88, 0.6), (56, 72, 0.3))
    components = []
    descriptions = []
    for component, (left, right, amplitude) in enumerate(definitions):
        fraction = _apex_fraction(file, block, component)
        components.append(_triangle(129, left, right, fraction, amplitude))
        descriptions.append(f"{left}:{right}:{fraction}:{amplitude}")
    return np.sum(components, axis=0), ";".join(descriptions)


def _asls(values: NDArray[np.float64], lam: float, p: float, iterations: int = 20) -> NDArray[np.float64]:
    n = values.size
    difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = lam * (difference.T @ difference)
    weights = np.ones(n)
    baseline = values.copy()
    for _ in range(iterations):
        system = sparse.diags(weights, format="csc") + penalty
        baseline = spsolve(system, weights * values)
        weights = np.where(values > baseline, p, 1.0 - p)
    return np.asarray(baseline)


def _wavelet_lowpass(values: NDArray[np.float64], level: int) -> NDArray[np.float64]:
    coefficients = pywt.wavedec(values, "sym4", mode="symmetric", level=level)
    filtered = [coefficients[0], *[np.zeros_like(item) for item in coefficients[1:]]]
    return pywt.waverec(filtered, "sym4", mode="symmetric")[: values.size]


def _oracle(
    candidates: list[tuple[str, NDArray[np.float64]]],
    truth: NDArray[np.float64],
) -> tuple[str, NDArray[np.float64]]:
    return min(
        candidates,
        key=lambda item: (float(np.mean((item[1] - truth) ** 2)), item[0]),
    )


def _estimates(
    observed: NDArray[np.float64], truth: NDArray[np.float64]
) -> dict[str, tuple[str, NDArray[np.float64]]]:
    hierarchy = decompose(observed, max_levels=8)
    estimates: dict[str, tuple[str, NDArray[np.float64]]] = {
        "hcrd_l1": ("level=1", hierarchy.levels[0].baseline),
        "hcrd_l8": (f"depth={hierarchy.depth}", hierarchy.trend),
    }
    estimates["gaussian_oracle"] = _oracle(
        [(f"sigma={sigma}", gaussian_filter1d(observed, sigma, mode="nearest")) for sigma in (1, 2, 4, 8, 16, 32)],
        truth,
    )
    estimates["savgol_oracle"] = _oracle(
        [
            (f"window={window};poly={polynomial}", savgol_filter(observed, window, polynomial, mode="interp"))
            for window in (9, 17, 33, 65, 97)
            for polynomial in (2, 3)
            if polynomial < window
        ],
        truth,
    )
    estimates["morph_open_oracle"] = _oracle(
        [(f"size={size}", grey_opening(observed, size=size, mode="nearest")) for size in (9, 17, 33, 65, 97)],
        truth,
    )
    estimates["asls_oracle"] = _oracle(
        [
            (f"lambda={lam:g};p={p:g}", _asls(observed, lam, p))
            for lam in (1e2, 1e3, 1e4, 1e5, 1e6)
            for p in (0.001, 0.01, 0.05)
        ],
        truth,
    )
    maximum_level = min(5, pywt.dwt_max_level(observed.size, pywt.Wavelet("sym4").dec_len))
    estimates["wavelet_oracle"] = _oracle(
        [(f"level={level}", _wavelet_lowpass(observed, level)) for level in range(1, maximum_level + 1)],
        truth,
    )
    return estimates


def _bootstrap_comparison(
    reference: NDArray[np.float64],
    comparator: NDArray[np.float64],
    rng: np.random.Generator,
    replicates: int,
    family_size: int,
) -> dict[str, object]:
    difference = reference - comparator
    samples = rng.integers(0, difference.size, size=(replicates, difference.size))
    boot = np.mean(difference[samples], axis=1)
    tail = 0.05 / (2.0 * family_size)
    nonzero = difference[~np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)]
    wins = int(np.sum(nonzero < 0.0))
    losses = int(np.sum(nonzero > 0.0))
    sign_p = float(binomtest(min(wins, losses), wins + losses, 0.5).pvalue) if nonzero.size else 1.0
    return {
        "mean_difference_hcrd_l8_minus_comparator": float(np.mean(difference)),
        "simultaneous_confidence_level": 1.0 - 0.05 / family_size,
        "bonferroni_bootstrap_interval": np.quantile(boot, [tail, 1.0 - tail]).tolist(),
        "wins": wins,
        "ties": int(difference.size - nonzero.size),
        "losses": losses,
        "exact_sign_p": sign_p,
        "bonferroni_sign_p": min(1.0, family_size * sign_p),
    }


def run(data_dir: Path, archive: Path, output_dir: Path, bootstrap: int) -> None:
    if _hash_file(archive, "md5") != ARCHIVE_MD5:
        raise RuntimeError("NeatMS Dataset 1 archive MD5 mismatch")
    backgrounds, manifest = prepare_backgrounds(data_dir)
    rows: list[dict[str, object]] = []
    for background_id, background in enumerate(backgrounds):
        source = manifest.iloc[background_id]
        for suite in ("single", "nested"):
            detail, description = _injection(
                str(source["file"]), int(source["block_index"]), suite
            )
            observed = background + detail
            for method, (parameter, estimate) in _estimates(observed, background).items():
                rows.append(
                    {
                        "background_id": background_id,
                        "file": source["file"],
                        "block_index": int(source["block_index"]),
                        "suite": suite,
                        "injection": description,
                        "method": method,
                        "parameter": parameter,
                        "baseline_mse": float(np.mean((estimate - background) ** 2)),
                        "detail_mse": float(np.mean(((observed - estimate) - detail) ** 2)),
                    }
                )
        if (background_id + 1) % 10 == 0:
            print(f"background {background_id + 1}/60", flush=True)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "case_metrics.csv", index=False)
    manifest.to_csv(output_dir / "background_manifest.csv", index=False)
    np.save(output_dir / "backgrounds.npy", backgrounds)

    aggregate = (
        frame.groupby(["suite", "method"], as_index=False)
        .agg(
            mean_baseline_mse=("baseline_mse", "mean"),
            median_baseline_mse=("baseline_mse", "median"),
            mean_detail_mse=("detail_mse", "mean"),
        )
        .sort_values(["suite", "mean_baseline_mse", "method"])
    )
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    primary = frame.loc[frame["suite"].eq("nested")]
    losses = {
        method: primary.loc[primary["method"].eq(method)]
        .sort_values("background_id")["baseline_mse"]
        .to_numpy()
        for method in METHODS
    }
    rng = np.random.default_rng(SEED)
    comparisons = {
        method: _bootstrap_comparison(losses["hcrd_l8"], losses[method], rng, bootstrap, 6)
        for method in METHODS
        if method != "hcrd_l8"
    }
    strongest_classical = min(
        CLASSICAL_METHODS, key=lambda method: (float(np.mean(losses[method])), method)
    )
    hcrd_mean = float(np.mean(losses["hcrd_l8"]))
    lowest = all(hcrd_mean < float(np.mean(losses[method])) for method in METHODS if method != "hcrd_l8")
    all_upper_negative = all(
        comparison["bonferroni_bootstrap_interval"][1] < 0.0
        for comparison in comparisons.values()
    )
    multilevel = comparisons["hcrd_l1"]["bonferroni_bootstrap_interval"][1] < 0.0
    win_count = comparisons[strongest_classical]["wins"]
    success_components = {
        "lowest_mean_nested_mse": lowest,
        "all_simultaneous_upper_bounds_negative": all_upper_negative,
        "multilevel_beats_level1": multilevel,
        "wins_at_least_45_of_60_vs_strongest_classical": win_count >= 45,
    }
    result = {
        "protocol": PROTOCOL,
        "prospective_success": bool(all(success_components.values())),
        "success_components": success_components,
        "strongest_classical_oracle": strongest_classical,
        "primary_comparisons": comparisons,
        "aggregate": aggregate.to_dict(orient="records"),
        "background_count": backgrounds.shape[0],
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": SEED,
        "source": {
            "doi": "10.5281/zenodo.3973172",
            "archive_md5": ARCHIVE_MD5,
            "archive_sha256": _hash_file(archive),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "pywavelets": pywt.__version__,
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=50000)
    args = parser.parse_args()
    run(
        args.data_dir.resolve(),
        args.archive.resolve(),
        args.output_dir.resolve(),
        args.bootstrap,
    )


if __name__ == "__main__":
    main()
