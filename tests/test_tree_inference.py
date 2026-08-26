from __future__ import annotations

import pytest

from hcrd import hierarchical_tree_test


def test_chain_reuses_level_after_each_rejected_parent() -> None:
    parents = {"root": None, "middle": "root", "leaf": "middle"}
    result = hierarchical_tree_test(
        {"root": 0.04, "middle": 0.04, "leaf": 0.04}, parents, alpha=0.05
    )
    assert result.local_levels == {"root": 0.05, "middle": 0.05, "leaf": 0.05}
    assert all(result.rejected.values())


def test_branch_levels_conserve_parent_budget_by_leaf_count() -> None:
    parents = {
        "root": None,
        "left": "root",
        "right": "root",
        "left_a": "left",
        "left_b": "left",
        "right_a": "right",
    }
    p_values = {node: 0.0 for node in parents}
    result = hierarchical_tree_test(p_values, parents, alpha=0.06)
    assert result.leaf_counts["root"] == 3
    assert result.local_levels["left"] == pytest.approx(0.04)
    assert result.local_levels["right"] == pytest.approx(0.02)
    assert result.local_levels["left_a"] == pytest.approx(0.02)
    assert result.local_levels["left_b"] == pytest.approx(0.02)
    assert result.local_levels["right_a"] == pytest.approx(0.02)


def test_nonrejected_parent_gates_all_descendants() -> None:
    parents = {"root": None, "child": "root", "leaf": "child"}
    result = hierarchical_tree_test(
        {"root": 0.5, "child": 0.0, "leaf": 0.0}, parents, alpha=0.05
    )
    assert result.tested == {"root": True, "child": False, "leaf": False}
    assert not any(result.rejected.values())


def test_tree_validation_rejects_cycle_and_schema_mismatch() -> None:
    with pytest.raises(ValueError, match="same nodes"):
        hierarchical_tree_test({"root": 0.1}, {"root": None, "x": "root"})
    with pytest.raises(ValueError, match="exactly one"):
        hierarchical_tree_test({"a": 0.1, "b": 0.1}, {"a": "b", "b": "a"})
