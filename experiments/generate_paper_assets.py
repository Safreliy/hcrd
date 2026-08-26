"""Generate manuscript figures and LaTeX tables from recorded CSV/JSON results."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hcrd.core import decompose  # noqa: E402

FIGURES = PROJECT / "paper" / "figures"
GENERATED = PROJECT / "paper" / "generated"


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def method_figure() -> None:
    # A deterministic EIC-like signal: affine drift, a main peak, a shoulder,
    # and a small negative baseline feature.  It is synthetic so the opening
    # explanation does not depend on redistributed third-party data.
    x = np.linspace(0.0, 1.0, 129)
    signal = (
        0.12
        + 0.18 * x
        + 1.40 * np.exp(-0.5 * ((x - 0.43) / 0.075) ** 2)
        + 0.52 * np.exp(-0.5 * ((x - 0.61) / 0.045) ** 2)
        - 0.10 * np.exp(-0.5 * ((x - 0.78) / 0.060) ** 2)
    )
    # Two levels expose the recursive mechanism.  Reconstruction is exact at
    # every truncation, even though the full transform may continue.
    result = decompose(signal, x, atol=0.0, rtol=0.0, max_levels=2)
    level_1, level_2 = result.levels
    figure, axes = plt.subplots(1, 4, figsize=(11.4, 2.9), sharex=True, sharey=True)

    axis = axes[0]
    for structure in level_1.structures:
        left, right = structure.left, structure.right
        colour = "#009E73" if structure.sign > 0 else "#CC79A7"
        axis.plot(
            x[left : right + 1],
            signal[left : right + 1],
            color=colour,
            lw=2.35,
            solid_capstyle="round",
        )
    axis.scatter(x[::4], signal[::4], s=6, color="0.25", alpha=0.35, zorder=3)
    axis.scatter(
        x[level_1.knots],
        signal[level_1.knots],
        s=28,
        facecolor="white",
        edgecolor="#0072B2",
        linewidth=1.15,
        zorder=4,
    )
    axis.set_title("1  split at curvature changes", loc="left", fontweight="bold", fontsize=9.2)
    axis.set_xlabel("retention time")
    axis.set_ylabel("intensity")

    axis = axes[1]
    positive = signal >= level_1.baseline
    axis.fill_between(x, level_1.baseline, signal, where=positive, color="#009E73", alpha=0.24, interpolate=True)
    axis.fill_between(x, level_1.baseline, signal, where=~positive, color="#CC79A7", alpha=0.24, interpolate=True)
    axis.plot(x, signal, color="0.18", lw=1.55)
    axis.plot(x, level_1.baseline, color="#D55E00", lw=2.15)
    axis.scatter(
        x[level_1.knots],
        signal[level_1.knots],
        s=24,
        facecolor="white",
        edgecolor="#D55E00",
        linewidth=1.1,
        zorder=4,
    )
    axis.set_title("2  level 1: endpoint chords", loc="left", fontweight="bold", fontsize=9.2)
    axis.set_xlabel(r"$y=b^1+d^1$ (shading is $d^1$)")

    axis = axes[2]
    positive = level_1.baseline >= level_2.baseline
    axis.fill_between(x, level_2.baseline, level_1.baseline, where=positive, color="#56B4E9", alpha=0.28, interpolate=True)
    axis.fill_between(x, level_2.baseline, level_1.baseline, where=~positive, color="#CC79A7", alpha=0.24, interpolate=True)
    axis.plot(x, level_1.baseline, color="0.35", lw=1.55)
    axis.plot(x, level_2.baseline, color="#6A3D9A", lw=2.15)
    axis.scatter(
        x[level_2.knots],
        level_1.baseline[level_2.knots],
        s=24,
        facecolor="white",
        edgecolor="#6A3D9A",
        linewidth=1.1,
        zorder=4,
    )
    axis.set_title("3  level 2: recurse on $b^1$", loc="left", fontweight="bold", fontsize=9.2)
    axis.set_xlabel(r"$b^1=b^2+d^2$")

    axis = axes[3]
    reconstruction = result.trend + sum(entry.detail for entry in result.levels)
    axis.plot(x, signal, color="0.18", lw=2.1, label="input")
    axis.plot(x, reconstruction, color="#0072B2", lw=1.35, ls="--", label="sum")
    axis.set_title("4  exact synthesis", loc="left", fontweight="bold", fontsize=9.2)
    axis.set_xlabel(r"$y=b^2+d^2+d^1$")
    axis.legend(frameon=False, fontsize=7.0, loc="upper right", handlelength=1.7)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.045, right=0.995, bottom=0.22, top=0.86, wspace=0.20)
    _save(figure, "method_overview")


def benchmark_figure() -> None:
    rows = list(
        csv.DictReader((PROJECT / "results" / "synthetic_v03" / "aggregate.csv").open(encoding="utf-8"))
    )
    selected = {
        "hcrd_centered": "HCRD",
        "hcrd_legacy": "legacy",
        "affine_ls": "affine LS",
        "gaussian_oracle": "Gaussian oracle",
        "moving_average_oracle": "MA oracle",
        "fourier_oracle": "Fourier oracle",
    }
    exact = {
        row["method"]: float(row["mean_baseline_mse"])
        for row in rows
        if row["suite"] == "exact" and float(row["noise_sigma"]) == 0.0 and row["method"] in selected
    }
    methods = list(selected)
    values = [max(exact[method], 1e-16) for method in methods]
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    colours = ["#0072B2"] + ["0.55"] * (len(methods) - 1)
    axis.bar([selected[method] for method in methods], values, color=colours)
    axis.set_yscale("log")
    axis.set_ylabel("mean baseline MSE (log scale)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save(figure, "exact_class_benchmark")


def stability_figure() -> None:
    rows = json.loads((PROJECT / "results" / "stability" / "summary.json").read_text(encoding="utf-8"))
    labels = {
        "hcrd_raw": "raw",
        "hcrd_robust": "thresholded",
        "hcrd_guided": "fixed guide",
        "hcrd_adaptive_guided": "adaptive guide",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for method, label in labels.items():
        group = sorted((row for row in rows if row["method"] == method), key=lambda row: row["noise_sigma"])
        axis.plot(
            [row["noise_sigma"] for row in group],
            [row["mean_target_knot_f1"] for row in group],
            marker="o",
            label=label,
        )
    axis.set_xlabel("noise standard deviation")
    axis.set_ylabel("target-knot F1 (one-sample tolerance)")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save(figure, "stability_accuracy")


def falsification_figure() -> None:
    rows = json.loads((PROJECT / "results" / "falsification" / "summary.json").read_text(encoding="utf-8"))
    group = [row for row in rows if row["case"] == "weak_oscillation_strong_curvature"]
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.bar(
        [row["method"].replace("hcrd_", "") for row in group],
        [row["mean_baseline_mse"] for row in group],
        color=["0.55", "#E69F00", "#0072B2", "#56B4E9"],
    )
    axis.set_yscale("log")
    axis.set_ylabel("baseline MSE (log scale)")
    axis.tick_params(axis="x", rotation=20)
    axis.set_title("Falsification: weak oscillation under strong curvature")
    figure.tight_layout()
    _save(figure, "curvature_visibility_failure")


def results_table() -> None:
    rows = list(
        csv.DictReader((PROJECT / "results" / "synthetic_v03" / "aggregate.csv").open(encoding="utf-8"))
    )
    methods = [
        ("hcrd_centered", "Centred HCRD"),
        ("hcrd_robust", "Thresholded HCRD"),
        ("hcrd_guided_adaptive", "Adaptive guided HCRD"),
        ("gaussian_oracle", "Gaussian oracle"),
        ("moving_average_oracle", "Moving-average oracle"),
    ]
    values: dict[tuple[str, str, float], float] = {
        (row["method"], row["suite"], float(row["noise_sigma"])): float(row["mean_baseline_mse"])
        for row in rows
    }
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Mean baseline MSE over 100 preregistered trials. Oracle labels indicate per-signal ground-truth tuning.}",
        "\\label{tab:synthetic}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Exact, $\\sigma=0$ & Exact, $\\sigma=0.1$ & Variable, $\\sigma=0.1$ \\\\",
        "\\midrule",
    ]
    for method, label in methods:
        lines.append(
            f"{label} & {values[(method, 'exact', 0.0)]:.3g} & "
            f"{values[(method, 'exact', 0.1)]:.3g} & "
            f"{values[(method, 'variable', 0.1)]:.3g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    method_figure()
    benchmark_figure()
    stability_figure()
    falsification_figure()
    results_table()
    print(json.dumps({"figures": str(FIGURES), "generated": str(GENERATED)}, indent=2))


if __name__ == "__main__":
    main()
