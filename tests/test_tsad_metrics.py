import numpy as np
import pytest

from hcrd import vus_pr_roc


def _slow_vus(labels, score, max_buffer, threshold_count):
    # Direct definition used only as an independent small-array test oracle.
    from hcrd.tsad_metrics import _expanded_ranges, _ranges, _soft_labels

    original_ranges = _ranges(labels)
    maximum_ranges = _expanded_ranges(original_ranges, max_buffer, len(labels))
    sorted_score = np.sort(score)[::-1]
    thresholds = sorted_score[
        np.linspace(0, len(score) - 1, threshold_count).astype(int)
    ]
    positives = labels.sum()
    aps = []
    aucs = []
    for buffer in range(max_buffer + 1):
        extended = _soft_labels(labels, original_ranges, buffer)
        current_ranges = _expanded_ranges(original_ranges, buffer, len(labels))
        tpr = []
        fpr = []
        precision = []
        for threshold in thresholds:
            prediction = score >= threshold
            weighted = extended * prediction
            weighted[labels == 1] = 1.0
            tp = sum(
                np.dot(weighted[left : right + 1], prediction[left : right + 1])
                for left, right in maximum_ranges
            )
            extension_predicted = sum(
                np.sum(weighted[left : right + 1])
                for left, right in maximum_ranges
            ) - positives
            effective_positive = positives + 0.5 * extension_predicted
            existence = np.mean(
                [prediction[left : right + 1].any() for left, right in current_ranges]
            )
            tpr.append(min(tp / effective_positive, 1.0) * existence)
            fpr.append((prediction.sum() - tp) / (len(labels) - effective_positive))
            precision.append(tp / prediction.sum())
        tpr_curve = np.asarray([0.0, *tpr, 1.0])
        fpr_curve = np.asarray([0.0, *fpr, 1.0])
        aps.append(np.dot(np.diff(tpr_curve[:-1]), precision))
        aucs.append(
            np.dot(
                np.diff(fpr_curve),
                (tpr_curve[1:] + tpr_curve[:-1]) / 2.0,
            )
        )
    return np.mean(aps), np.mean(aucs)


def test_fast_vus_matches_direct_definition_with_ties_and_two_ranges():
    labels = np.asarray([0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=int)
    score = np.asarray([0.1, 0.7, 0.7, 0.4, 0.2, 0.1, 0.8, 0.3, 0.2, 0.0, 0.1])
    expected = _slow_vus(labels, score, max_buffer=4, threshold_count=7)
    observed = vus_pr_roc(labels, score, max_buffer=4, threshold_count=7)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=1e-14)


def test_vus_input_validation():
    with pytest.raises(ValueError, match="at least one anomaly"):
        vus_pr_roc([0, 0, 0], [0.1, 0.2, 0.3], max_buffer=2)
    with pytest.raises(ValueError, match="binary"):
        vus_pr_roc([0, 2, 1], [0.1, 0.2, 0.3], max_buffer=2)

