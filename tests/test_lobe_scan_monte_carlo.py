"""Small deterministic smoke test for the lobe-scan Monte Carlo runner."""

from experiments.run_lobe_scan_monte_carlo import run


def test_lobe_scan_monte_carlo_schema_and_bounds() -> None:
    result = run(replicates=1000, batch_size=250)
    assert result["template_count"] == 24
    assert result["sample_count"] == 257
    assert result["theorem_bounds"]["scan_threshold"] > 0.0
    empirical = result["empirical"]
    assert 0.0 <= empirical["null_false_rejection_rate"] <= 1.0
    assert 0.0 <= empirical["miss_rate_at_power_bound"] <= 1.0
    assert 0.0 <= empirical["localization_error_at_bound"] <= 1.0
