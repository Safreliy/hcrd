"""Metrics and dependency-free paired inference for experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def mse(estimate: ArrayLike, target: ArrayLike) -> float:
    difference = np.asarray(estimate, dtype=float) - np.asarray(target, dtype=float)
    return float(np.mean(difference**2))


def nmse(estimate: ArrayLike, target: ArrayLike) -> float:
    target_array = np.asarray(target, dtype=float)
    denominator = float(np.mean((target_array - np.mean(target_array)) ** 2))
    if denominator == 0:
        denominator = float(np.mean(target_array**2))
    if denominator == 0:
        return 0.0 if np.allclose(estimate, target_array) else float("inf")
    return mse(estimate, target_array) / denominator


def scaled_mse(estimate: ArrayLike, target: ArrayLike, reference: ArrayLike) -> float:
    """MSE normalized by the centred power of an external reference signal."""

    reference_array = np.asarray(reference, dtype=float)
    denominator = float(np.mean((reference_array - np.mean(reference_array)) ** 2))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("reference signal has zero centred power")
    return mse(estimate, target) / denominator


def knot_f1(estimate: Sequence[int], target: Sequence[int], tolerance: int = 1) -> float:
    estimated = [int(value) for value in estimate if value not in (estimate[0], estimate[-1])]
    truth = [int(value) for value in target if value not in (target[0], target[-1])]
    if not estimated and not truth:
        return 1.0
    if not estimated or not truth:
        return 0.0
    unmatched = set(range(len(truth)))
    matches = 0
    for value in estimated:
        candidates = [index for index in unmatched if abs(truth[index] - value) <= tolerance]
        if candidates:
            chosen = min(candidates, key=lambda index: abs(truth[index] - value))
            unmatched.remove(chosen)
            matches += 1
    precision = matches / len(estimated)
    recall = matches / len(truth)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def jaccard_with_tolerance(
    first: Sequence[int], second: Sequence[int], tolerance: int = 1
) -> float:
    first_list = list(map(int, first))
    second_list = list(map(int, second))
    if not first_list and not second_list:
        return 1.0
    used: set[int] = set()
    matches = 0
    for value in first_list:
        candidates = [
            index
            for index, other in enumerate(second_list)
            if index not in used and abs(value - other) <= tolerance
        ]
        if candidates:
            chosen = min(candidates, key=lambda index: abs(value - second_list[index]))
            used.add(chosen)
            matches += 1
    union = len(first_list) + len(second_list) - matches
    return matches / union if union else 1.0


def paired_bootstrap_ci(
    differences: ArrayLike,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 20260824,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a nonempty one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def exact_sign_test(differences: ArrayLike) -> float:
    """Two-sided exact sign test after removing ties.

    Negative differences mean that the first method has lower loss.
    """

    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    positives = int(np.sum(nonzero > 0))
    tail = min(positives, n - positives)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)
