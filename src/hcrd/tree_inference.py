"""Strong-FWER gatekeeping for guide-fixed nested HCRD hypothesis trees."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HierarchicalTreeTestResult:
    """Local levels and decisions from a nested gatekeeping procedure."""

    root: Hashable
    local_levels: dict[Hashable, float]
    leaf_counts: dict[Hashable, int]
    tested: dict[Hashable, bool]
    rejected: dict[Hashable, bool]


def _validated_tree(
    parents: Mapping[Hashable, Hashable | None],
) -> tuple[Hashable, dict[Hashable, list[Hashable]], dict[Hashable, int]]:
    nodes = set(parents)
    if not nodes:
        raise ValueError("parents must describe a nonempty tree")
    roots = [node for node, parent in parents.items() if parent is None]
    if len(roots) != 1:
        raise ValueError("parents must describe exactly one rooted tree")
    root = roots[0]
    children = {node: [] for node in nodes}
    for node, parent in parents.items():
        if parent is None:
            continue
        if parent == node:
            raise ValueError("a node cannot be its own parent")
        if parent not in nodes:
            raise ValueError("every non-root parent must be a declared node")
        children[parent].append(node)

    state: dict[Hashable, int] = {}
    leaf_counts: dict[Hashable, int] = {}

    def visit(node: Hashable) -> int:
        marker = state.get(node, 0)
        if marker == 1:
            raise ValueError("parents contain a cycle")
        if marker == 2:
            return leaf_counts[node]
        state[node] = 1
        count = 1 if not children[node] else sum(visit(child) for child in children[node])
        leaf_counts[node] = count
        state[node] = 2
        return count

    visit(root)
    if len(state) != len(nodes):
        raise ValueError("parents contain nodes disconnected from the root")
    return root, children, leaf_counts


def hierarchical_tree_test(
    p_values: Mapping[Hashable, float],
    parents: Mapping[Hashable, Hashable | None],
    *,
    alpha: float = 0.05,
) -> HierarchicalTreeTestResult:
    """Test a logically nested tree with strong FWER control.

    The node null must mean “no signal anywhere in this subtree”, so a true
    parent implies that all descendant nulls are true.  Each child receives its
    parent's level in proportion to its number of descendant leaves; children
    are tested only after their parent rejects.  Under valid marginal p-values,
    the procedure controls FWER at ``alpha`` under arbitrary dependence.

    The tree and p-values must be fixed or selected from an independent guide.
    This procedure is not valid for unrelated per-structure nulls lacking the
    subtree intersection property.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("0 < alpha < 1 is required")
    if set(p_values) != set(parents):
        raise ValueError("p_values and parents must contain the same nodes")
    values = {node: float(value) for node, value in p_values.items()}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("all p-values must be finite and lie in [0, 1]")
    root, children, leaf_counts = _validated_tree(parents)

    local_levels: dict[Hashable, float] = {root: float(alpha)}
    tested: dict[Hashable, bool] = {}
    rejected: dict[Hashable, bool] = {}
    queue: deque[Hashable] = deque([root])
    while queue:
        node = queue.popleft()
        parent = parents[node]
        is_tested = parent is None or rejected[parent]
        tested[node] = bool(is_tested)
        rejected[node] = bool(is_tested and values[node] <= local_levels[node])
        child_nodes = children[node]
        if child_nodes:
            denominator = sum(leaf_counts[child] for child in child_nodes)
            for child in child_nodes:
                local_levels[child] = (
                    local_levels[node] * leaf_counts[child] / denominator
                )
                queue.append(child)

    return HierarchicalTreeTestResult(
        root=root,
        local_levels=local_levels,
        leaf_counts=leaf_counts,
        tested=tested,
        rejected=rejected,
    )
