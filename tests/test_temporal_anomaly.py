import numpy as np
import pytest

from hcrd import (
    empirical_rank,
    hcrd_temporal_candidate_scores,
    spectral_residual_score,
)


def test_temporal_candidate_family_is_aligned_finite_and_label_free():
    x = np.linspace(0.0, 1.0, 257)
    signal = np.sin(12.0 * np.pi * x)
    signal[131:139] += np.hanning(8) * 2.0
    scores = hcrd_temporal_candidate_scores(signal, max_levels=6)
    assert len(scores) == 12
    for score in scores.values():
        assert score.shape == signal.shape
        assert np.all(np.isfinite(score))
        assert np.all(score >= 0.0)


def test_empirical_rank_preserves_ties_and_spectral_residual_handles_constants():
    ranked = empirical_rank([1.0, 1.0, 3.0, 2.0])
    np.testing.assert_allclose(ranked, [0.375, 0.375, 1.0, 0.75])
    np.testing.assert_array_equal(spectral_residual_score(np.ones(17)), 0.0)
    with pytest.raises(ValueError, match="positive"):
        spectral_residual_score(np.arange(5.0), amplitude_window=0)

