#!/usr/bin/env python3
"""Post-E4 development audit for mixed-polarity nested chord lobes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import gaussian_filter1d, grey_closing, grey_opening, median_filter
from scipy.signal import savgol_filter

from hcrd import decompose
from hcrd.external_baselines import l1_trend_filter_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_neatms_real_background_e4 as e4  # noqa: E402


SEED = 20260825


def _mixed_detail(file: str, block: int) -> tuple[np.ndarray, str]:
    global_sign = 1.0 if hashlib.sha256(f"mixed|{file}|{block}".encode()).digest()[0] % 2 else -1.0
    definitions = ((16, 112, 1.0), (40, 88, -0.6), (56, 72, 0.3))
    components = []
    descriptions = []
    for index, (left, right, amplitude) in enumerate(definitions):
        fraction = e4._apex_fraction(file, block, index)
        signed_amplitude = global_sign * amplitude
        components.append(e4._triangle(129, left, right, fraction, signed_amplitude))
        descriptions.append(f"{left}:{right}:{fraction}:{signed_amplitude}")
    return np.sum(components, axis=0), ";".join(descriptions)


def _oracle(candidates, truth):
    return min(candidates, key=lambda item: (np.mean((item[1] - truth) ** 2), item[0]))


def _estimates(observed: np.ndarray, truth: np.ndarray):
    hierarchy = decompose(observed, max_levels=8)
    output = {
        "hcrd_l1": ("level=1", hierarchy.levels[0].baseline),
        "hcrd_l8": (f"depth={hierarchy.depth}", hierarchy.trend),
    }
    output["gaussian_oracle"] = _oracle(
        [(f"sigma={s}", gaussian_filter1d(observed, s, mode="nearest")) for s in (1, 2, 4, 8, 16, 32)], truth
    )
    output["savgol_oracle"] = _oracle(
        [(f"window={w};poly={p}", savgol_filter(observed, w, p, mode="interp")) for w in (9, 17, 33, 65, 97) for p in (2, 3)], truth
    )
    output["median_oracle"] = _oracle(
        [(f"size={s}", median_filter(observed, size=s, mode="nearest")) for s in (9, 17, 33, 65, 97)], truth
    )
    output["symmetric_morph_oracle"] = _oracle(
        [
            (
                f"size={s}",
                0.5
                * (
                    grey_opening(observed, size=s, mode="nearest")
                    + grey_closing(observed, size=s, mode="nearest")
                ),
            )
            for s in (9, 17, 33, 65, 97)
        ],
        truth,
    )
    output["signed_asls_oracle"] = _oracle(
        [
            (f"lambda={lam:g};p={p:g}", e4._asls(observed, lam, p))
            for lam in (1e2, 1e3, 1e4, 1e5, 1e6)
            for p in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
        ],
        truth,
    )
    maximum_level = min(5, pywt.dwt_max_level(observed.size, pywt.Wavelet("sym4").dec_len))
    output["wavelet_oracle"] = _oracle(
        [(f"level={level}", e4._wavelet_lowpass(observed, level)) for level in range(1, maximum_level + 1)], truth
    )
    lambdas = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    trend_path = l1_trend_filter_path(observed, lambdas)
    output["l1_trend_oracle"] = _oracle(
        [(f"lambda={lam:g}", estimate) for lam, estimate in zip(lambdas, trend_path, strict=True)], truth
    )
    return output


def analyze(background_dir: Path, output_dir: Path) -> None:
    backgrounds = np.load(background_dir / "backgrounds.npy")
    manifest = pd.read_csv(background_dir / "background_manifest.csv")
    rows = []
    for index, background in enumerate(backgrounds):
        source = manifest.iloc[index]
        detail, description = _mixed_detail(str(source.file), int(source.block_index))
        observed = background + detail
        for method, (parameter, estimate) in _estimates(observed, background).items():
            rows.append(
                {
                    "background_id": index,
                    "file": source.file,
                    "block_index": int(source.block_index),
                    "injection": description,
                    "method": method,
                    "parameter": parameter,
                    "baseline_mse": float(np.mean((estimate - background) ** 2)),
                }
            )
        if (index + 1) % 10 == 0:
            print(f"mixed background {index + 1}/60", flush=True)
    frame = pd.DataFrame(rows)
    aggregate = (
        frame.groupby("method", as_index=False)
        .agg(mean_mse=("baseline_mse", "mean"), median_mse=("baseline_mse", "median"))
        .sort_values(["mean_mse", "method"])
    )
    hcrd = frame.loc[frame.method.eq("hcrd_l8")].sort_values("background_id").baseline_mse.to_numpy()
    comparisons = {}
    rng = np.random.default_rng(SEED)
    for method in aggregate.method:
        if method == "hcrd_l8":
            continue
        other = frame.loc[frame.method.eq(method)].sort_values("background_id").baseline_mse.to_numpy()
        difference = hcrd - other
        boot = np.mean(difference[rng.integers(0, 60, size=(50000, 60))], axis=1)
        comparisons[method] = {
            "mean_difference": float(np.mean(difference)),
            "bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wins": int(np.sum(difference < 0.0)),
            "losses": int(np.sum(difference > 0.0)),
        }
    result = {
        "status": "post-E4 exploratory class development; not confirmation",
        "aggregate": aggregate.to_dict(orient="records"),
        "hcrd_l8_comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "case_metrics.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.background_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
