#!/usr/bin/env python3
"""Monte Carlo audit of the finite HCRD lobe-dictionary scan bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hcrd.lobe_scan import (
    residualized_lobe_dictionary,
    scan_detection_threshold,
    scan_localization_sufficient_norm,
    scan_power_sufficient_norm,
)


SEED = 20260825


def _dictionary() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 257)
    templates = np.zeros((24, x.size))
    for row, center in enumerate(np.linspace(10, 246, 24).astype(int)):
        support = np.arange(center - 4, center + 5)
        templates[row, support] = 1.0 - np.abs(support - center) / 4.0
    return x, residualized_lobe_dictionary(templates, x=x)


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z**2 / total
    center = (estimate + z**2 / (2.0 * total)) / denominator
    half = z * np.sqrt(estimate * (1.0 - estimate) / total + z**2 / (4.0 * total**2)) / denominator
    return [float(center - half), float(center + half)]


def run(replicates: int, batch_size: int) -> dict[str, object]:
    alpha, beta, delta = 0.05, 0.20, 0.10
    _, dictionary = _dictionary()
    template_count, sample_count = dictionary.shape
    gram = dictionary @ dictionary.T
    maximum_coherence = float(np.max(gram - np.eye(template_count)))
    threshold = scan_detection_threshold(template_count, alpha)
    power_norm = scan_power_sufficient_norm(template_count, alpha, beta)
    localization_norm = scan_localization_sufficient_norm(
        template_count, delta, maximum_coherence
    )
    rng = np.random.default_rng(SEED)
    false_rejections = 0
    misses = 0
    localization_errors = 0
    completed = 0
    while completed < replicates:
        count = min(batch_size, replicates - completed)
        noise = rng.normal(size=(count, sample_count))
        null_scores = noise @ dictionary.T
        false_rejections += int(np.sum(np.max(null_scores, axis=1) > threshold))
        indices = rng.integers(0, template_count, size=count)
        means_at_power = power_norm * gram[indices]
        power_scores = null_scores + means_at_power
        misses += int(np.sum(np.max(power_scores, axis=1) <= threshold))
        means_at_localization = localization_norm * gram[indices]
        localization_scores = null_scores + means_at_localization
        localization_errors += int(
            np.sum(np.argmax(localization_scores, axis=1) != indices)
        )
        completed += count
    return {
        "protocol": "hcrd-lobe-scan-mc-v1",
        "seed": SEED,
        "replicates": replicates,
        "sample_count": sample_count,
        "template_count": template_count,
        "maximum_coherence": maximum_coherence,
        "targets": {"alpha": alpha, "beta": beta, "delta": delta},
        "theorem_bounds": {
            "scan_threshold": threshold,
            "power_sufficient_standardized_norm": power_norm,
            "localization_sufficient_standardized_norm": localization_norm,
        },
        "empirical": {
            "null_false_rejection_rate": false_rejections / replicates,
            "null_false_rejection_wilson_95_ci": _wilson(false_rejections, replicates),
            "miss_rate_at_power_bound": misses / replicates,
            "miss_rate_wilson_95_ci": _wilson(misses, replicates),
            "localization_error_at_bound": localization_errors / replicates,
            "localization_error_wilson_95_ci": _wilson(localization_errors, replicates),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lobe_scan_monte_carlo/summary.json"),
    )
    args = parser.parse_args()
    result = run(args.replicates, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
