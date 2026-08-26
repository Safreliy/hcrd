"""Efficient, threshold-exact VUS metrics for range anomalies.

The implementation follows the TSB-AD/VUS definition but evaluates threshold
sets through score-order cumulative sums.  It is mathematically equivalent to
rebuilding a length-n prediction vector at every threshold, while avoiding the
resulting ``O(buffers * thresholds * n)`` work.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _ranges(labels: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.diff(labels) == 1) + 1
    ends = np.flatnonzero(np.diff(labels) == -1)
    if ends.size and (not starts.size or ends[0] < starts[0]):
        starts = np.concatenate(([0], starts))
    if starts.size and (not ends.size or ends[-1] < starts[-1]):
        ends = np.concatenate((ends, [labels.size - 1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _expanded_ranges(
    original: list[tuple[int, int]], buffer: int, length: int
) -> list[tuple[int, int]]:
    left = max(original[0][0] - buffer // 2, 0)
    output: list[tuple[int, int]] = []
    for current, following in zip(original[:-1], original[1:], strict=True):
        right = current[1] + buffer // 2
        next_left = following[0] - buffer // 2
        if right < next_left:
            output.append((left, right))
            left = next_left
    output.append(
        (left, min(original[-1][1] + buffer // 2, length - 1))
    )
    return output


def _soft_labels(
    labels: np.ndarray,
    ranges: list[tuple[int, int]],
    buffer: int,
) -> np.ndarray:
    extended = labels.astype(float, copy=True)
    if buffer == 0:
        return extended
    length = labels.size
    for start, end in ranges:
        right = np.arange(end + 1, min(end + buffer // 2 + 1, length))
        extended[right] += np.sqrt(1.0 - (right - end) / buffer)
        left = np.arange(max(start - buffer // 2, 0), start)
        extended[left] += np.sqrt(1.0 - (start - left) / buffer)
    return np.minimum(extended, 1.0)


def _cumulative_at(cumulative: np.ndarray, counts: np.ndarray) -> np.ndarray:
    output = np.zeros(counts.size, dtype=float)
    positive = counts > 0
    output[positive] = cumulative[counts[positive] - 1]
    return output


def vus_pr_roc(
    labels: ArrayLike,
    score: ArrayLike,
    *,
    max_buffer: int,
    threshold_count: int = 250,
) -> tuple[float, float]:
    """Return VUS-PR and VUS-ROC for one anomaly score.

    Thresholds are exactly the rank-spaced thresholds used by TSB-AD.  The
    buffer surface is averaged over every integer width from zero through
    ``max_buffer`` inclusive.
    """

    binary = np.asarray(labels, dtype=int)
    values = np.asarray(score, dtype=float)
    if binary.ndim != 1 or values.shape != binary.shape or binary.size < 2:
        raise ValueError("labels and score must be one-dimensional and aligned")
    if not np.all(np.isfinite(values)):
        raise ValueError("score must be finite")
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("labels must be binary")
    if max_buffer < 0 or threshold_count < 2:
        raise ValueError("max_buffer must be nonnegative and threshold_count >= 2")
    anomaly_ranges = _ranges(binary)
    if not anomaly_ranges:
        raise ValueError("labels must contain at least one anomaly")

    order = np.argsort(values)[::-1]
    sorted_scores = values[order]
    sampled = np.linspace(0, values.size - 1, threshold_count).astype(int)
    thresholds = sorted_scores[sampled]
    # Count all tied scores, matching ``score >= threshold`` exactly.
    predicted_counts = np.searchsorted(
        -sorted_scores, -thresholds, side="right"
    )
    original_cumulative = np.cumsum(binary[order], dtype=float)
    original_true_positives = _cumulative_at(
        original_cumulative, predicted_counts
    )
    positives = float(np.sum(binary))

    average_precision = np.empty(max_buffer + 1, dtype=float)
    area_roc = np.empty(max_buffer + 1, dtype=float)
    for buffer in range(max_buffer + 1):
        extended = _soft_labels(binary, anomaly_ranges, buffer)
        extension = extended - binary
        extension_cumulative = np.cumsum(extension[order], dtype=float)
        extension_true_positives = _cumulative_at(
            extension_cumulative, predicted_counts
        )
        true_positives = original_true_positives + extension_true_positives
        effective_positive = positives + 0.5 * extension_true_positives
        recall = np.minimum(true_positives / effective_positive, 1.0)

        expanded = _expanded_ranges(anomaly_ranges, buffer, binary.size)
        maxima = np.asarray(
            [np.max(values[left : right + 1]) for left, right in expanded]
        )
        existence = np.mean(maxima[:, None] >= thresholds[None, :], axis=0)
        true_positive_rate = recall * existence
        precision = true_positives / predicted_counts
        false_positive_rate = (predicted_counts - true_positives) / (
            binary.size - effective_positive
        )

        average_precision[buffer] = np.dot(
            np.diff(np.concatenate(([0.0], true_positive_rate))), precision
        )
        tpr_curve = np.concatenate(([0.0], true_positive_rate, [1.0]))
        fpr_curve = np.concatenate(([0.0], false_positive_rate, [1.0]))
        area_roc[buffer] = np.dot(
            np.diff(fpr_curve), (tpr_curve[1:] + tpr_curve[:-1]) / 2.0
        )
    return float(np.mean(average_precision)), float(np.mean(area_roc))

