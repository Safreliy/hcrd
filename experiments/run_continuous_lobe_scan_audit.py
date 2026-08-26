#!/usr/bin/env python3
"""Monte Carlo audit for the continuous triangular-lobe theorem and sieve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import binomtest, chi2, norm

from hcrd import (
    asymmetric_triangular_lobes,
    affine_residual_subspace_rank,
    continuous_scan_detection_threshold,
    continuous_scan_power_sufficient_norm,
    residualized_lobe_dictionary,
    sieve_power_sufficient_norm,
    subspace_scan_detection_threshold,
    subspace_scan_power_sufficient_norm,
    triangular_lobe_lipschitz_certificate,
)


SEED = 20260825


def _parameter_grid() -> np.ndarray:
    center = np.linspace(0.30, 0.70, 17)
    width = np.linspace(0.15, 0.30, 7)
    apex = np.linspace(0.30, 0.70, 5)
    return np.asarray(np.meshgrid(center, width, apex, indexing="ij")).reshape(3, -1).T


def _max_scores(
    dictionary: np.ndarray,
    rng: np.random.Generator,
    draws: int,
    *,
    mean: np.ndarray | None = None,
    batch_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    maxima = np.empty(draws)
    selected = np.empty(draws, dtype=np.int32)
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        noise = rng.normal(size=(dictionary.shape[1], stop - start))
        if mean is not None:
            noise += mean[:, None]
        scores = dictionary @ noise
        selected[start:stop] = np.argmax(scores, axis=0)
        maxima[start:stop] = np.max(scores, axis=0)
    return maxima, selected


def _binomial_interval(successes: int, trials: int) -> list[float]:
    interval = binomtest(successes, trials).proportion_ci(confidence_level=0.95)
    return [float(interval.low), float(interval.high)]


def run(output_dir: Path, null_draws: int, power_draws: int) -> None:
    x = np.linspace(0.0, 1.0, 129)
    parameters = _parameter_grid()
    raw = asymmetric_triangular_lobes(x, parameters)
    dictionary = residualized_lobe_dictionary(raw, x=x)
    certificate = triangular_lobe_lipschitz_certificate(
        x,
        center_bounds=(0.30, 0.70),
        width_bounds=(0.15, 0.30),
        apex_fraction_bounds=(0.30, 0.70),
    )
    alpha = 0.05
    beta = 0.20
    finite_threshold = float(np.sqrt(2.0 * np.log(parameters.shape[0] / alpha)))
    continuous_threshold = continuous_scan_detection_threshold(
        certificate.entropy_integral_upper, alpha
    )
    continuous_power_norm = continuous_scan_power_sufficient_norm(
        certificate.entropy_integral_upper, alpha, beta
    )
    residual_rank = affine_residual_subspace_rank(x)
    exact_subspace_threshold = float(np.sqrt(chi2.ppf(1.0 - alpha, residual_rank)))
    analytic_subspace_threshold = subspace_scan_detection_threshold(
        residual_rank, alpha
    )
    exact_subspace_power_norm = exact_subspace_threshold + float(norm.ppf(1.0 - beta))
    analytic_subspace_power_norm = subspace_scan_power_sufficient_norm(
        residual_rank, alpha, beta
    )

    rng = np.random.default_rng(SEED)
    null_max, _ = _max_scores(dictionary, rng, null_draws)
    finite_exceedances = int(np.sum(null_max > finite_threshold))
    continuous_exceedances = int(np.sum(null_max > continuous_threshold))
    exact_subspace_exceedances_on_sieve = int(
        np.sum(null_max > exact_subspace_threshold)
    )
    analytic_subspace_exceedances_on_sieve = int(
        np.sum(null_max > analytic_subspace_threshold)
    )
    gaussian_subspace_norm_squared = rng.chisquare(residual_rank, size=null_draws)
    exact_chi_exceedances = int(
        np.sum(gaussian_subspace_norm_squared > exact_subspace_threshold**2)
    )
    analytic_chi_exceedances = int(
        np.sum(gaussian_subspace_norm_squared > analytic_subspace_threshold**2)
    )

    true_parameter = np.asarray([0.50, 0.225, 0.50])
    true_index = int(np.argmin(np.linalg.norm(parameters - true_parameter, axis=1)))
    if not np.allclose(parameters[true_index], true_parameter):
        raise RuntimeError("declared true parameter is absent from the sieve")
    true_template = dictionary[true_index]
    finite_sufficient = sieve_power_sufficient_norm(
        parameters.shape[0], alpha, beta, canonical_mesh_radius=0.0
    )
    mu_grid = sorted(
        set(
            [
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                float(finite_sufficient),
                float(exact_subspace_power_norm),
                float(analytic_subspace_power_norm),
                float(continuous_power_norm),
            ]
        )
    )
    power_rows = []
    for mu in mu_grid:
        maxima, selected = _max_scores(
            dictionary, rng, power_draws, mean=mu * true_template
        )
        detected = maxima > finite_threshold
        detections = int(np.sum(detected))
        exact = int(np.sum(selected == true_index))
        selected_templates = dictionary[selected]
        canonical_error = np.linalg.norm(selected_templates - true_template, axis=1)
        power_rows.append(
            {
                "signal_norm": mu,
                "finite_sieve_empirical_power": detections / power_draws,
                "finite_sieve_power_95_ci": _binomial_interval(detections, power_draws),
                "included_true_template_power_lower": float(
                    norm.cdf(mu - finite_threshold)
                ),
                "exact_grid_localization_rate": exact / power_draws,
                "canonical_localization_error_median": float(np.median(canonical_error)),
                "canonical_localization_error_q95": float(np.quantile(canonical_error, 0.95)),
            }
        )
        print(f"power mu={mu:.6g}", flush=True)

    # This dense random audit is explicitly empirical, not a proof certificate.
    validation_parameters = np.c_[
        rng.uniform(0.30, 0.70, 10000),
        rng.uniform(0.15, 0.30, 10000),
        rng.uniform(0.30, 0.70, 10000),
    ]
    validation_dictionary = residualized_lobe_dictionary(
        asymmetric_triangular_lobes(x, validation_parameters), x=x
    )
    nearest_correlation = np.max(validation_dictionary @ dictionary.T, axis=1)
    empirical_radius = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * nearest_correlation))
    center_step = 0.40 / 16.0
    width_step = 0.15 / 6.0
    apex_step = 0.40 / 4.0
    parameter_mesh_radius = 0.5 * np.linalg.norm(
        [center_step, width_step, apex_step]
    )
    analytic_mesh_upper = (
        certificate.normalized_template_lipschitz * parameter_mesh_radius
    )

    result = {
        "protocol": "hcrd-continuous-lobe-audit-v2",
        "seed": SEED,
        "grid": {
            "sample_count": x.size,
            "template_count": parameters.shape[0],
            "center_count": 17,
            "width_count": 7,
            "apex_count": 5,
        },
        "analytic_certificate": {
            "maximum_grid_spacing": certificate.maximum_grid_spacing,
            "minimum_apex_segment": certificate.minimum_apex_segment,
            "sampled_apex_height_lower": certificate.sampled_apex_height_lower,
            "residual_norm_lower": certificate.residual_norm_lower,
            "raw_template_lipschitz": certificate.raw_template_lipschitz,
            "normalized_template_lipschitz": certificate.normalized_template_lipschitz,
            "canonical_diameter_upper": certificate.canonical_diameter_upper,
            "entropy_integral_upper": certificate.entropy_integral_upper,
            "continuous_level_threshold": continuous_threshold,
            "continuous_power_sufficient_norm_beta_0.2": continuous_power_norm,
        },
        "rank_aware_continuous_certificate": {
            "residual_subspace_rank": residual_rank,
            "exact_chi_level_threshold": exact_subspace_threshold,
            "laurent_massart_level_threshold": analytic_subspace_threshold,
            "exact_chi_power_sufficient_norm_beta_0.2": exact_subspace_power_norm,
            "laurent_massart_power_sufficient_norm_beta_0.2": analytic_subspace_power_norm,
            "exact_chi_null_draws": null_draws,
            "exact_chi_exceedances": exact_chi_exceedances,
            "exact_chi_empirical_level": exact_chi_exceedances / null_draws,
            "exact_chi_empirical_level_95_ci": _binomial_interval(
                exact_chi_exceedances, null_draws
            ),
            "laurent_massart_exceedances": analytic_chi_exceedances,
            "laurent_massart_empirical_level": analytic_chi_exceedances / null_draws,
            "laurent_massart_empirical_level_95_ci": _binomial_interval(
                analytic_chi_exceedances, null_draws
            ),
        },
        "finite_sieve_null": {
            "alpha": alpha,
            "threshold": finite_threshold,
            "draws": null_draws,
            "exceedances": finite_exceedances,
            "empirical_fwer": finite_exceedances / null_draws,
            "empirical_fwer_95_ci": _binomial_interval(finite_exceedances, null_draws),
            "null_supremum_mean": float(np.mean(null_max)),
            "null_supremum_q95": float(np.quantile(null_max, 0.95)),
            "continuous_threshold_exceedances_on_sieve": continuous_exceedances,
            "exact_subspace_threshold_exceedances_on_sieve": exact_subspace_exceedances_on_sieve,
            "analytic_subspace_threshold_exceedances_on_sieve": analytic_subspace_exceedances_on_sieve,
        },
        "mesh_audit": {
            "status": "empirical_only_not_a_continuum_certificate",
            "validation_parameter_count": validation_parameters.shape[0],
            "maximum_observed_canonical_radius": float(np.max(empirical_radius)),
            "q99_observed_canonical_radius": float(np.quantile(empirical_radius, 0.99)),
            "parameter_half_cell_radius": float(parameter_mesh_radius),
            "analytic_lipschitz_mesh_upper": float(analytic_mesh_upper),
            "analytic_sieve_condition_eta_lt_sqrt2_met": bool(
                analytic_mesh_upper < np.sqrt(2.0)
            ),
        },
        "finite_sieve_power_sufficient_norm_beta_0.2": finite_sufficient,
        "power_and_localization": power_rows,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    sorted_null = np.sort(null_max)
    axes[0].plot(sorted_null, np.arange(1, null_draws + 1) / null_draws, color="#276FBF")
    axes[0].axvline(finite_threshold, color="#D1495B", linestyle="--", label="finite threshold")
    axes[0].axvline(
        exact_subspace_threshold,
        color="#2A9D8F",
        linestyle=":",
        label="exact subspace threshold",
    )
    axes[0].text(
        0.97,
        0.08,
        f"Dudley threshold = {continuous_threshold:.1f}\n(off scale)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=8,
    )
    axes[0].set_xlim(-0.2, 1.08 * analytic_subspace_threshold)
    axes[0].set(xlabel="null scan supremum", ylabel="empirical CDF", title="Null calibration")
    axes[0].legend(frameon=False, fontsize=8)
    finite_rows = [row for row in power_rows if row["signal_norm"] < 20.0]
    axes[1].plot(
        [row["signal_norm"] for row in finite_rows],
        [row["finite_sieve_empirical_power"] for row in finite_rows],
        "o-",
        label="observed scan power",
    )
    axes[1].plot(
        [row["signal_norm"] for row in finite_rows],
        [row["included_true_template_power_lower"] for row in finite_rows],
        "--",
        label="pointwise lower bound",
    )
    axes[1].axhline(0.8, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="standardized signal norm", ylabel="power", ylim=(0, 1.02), title="Finite-sieve power")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "continuous_lobe_audit.png", dpi=180)
    figure.savefig(output_dir / "continuous_lobe_audit.pdf")
    plt.close(figure)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-draws", type=int, default=50000)
    parser.add_argument("--power-draws", type=int, default=10000)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.null_draws, args.power_draws)


if __name__ == "__main__":
    main()
