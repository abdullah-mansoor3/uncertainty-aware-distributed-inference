"""Uncertainty-aware scheduler using PRO score routing."""

from __future__ import annotations

from typing import Callable, Dict, List

from .base_scheduler import BaseScheduler


class UncertaintyAwareScheduler(BaseScheduler):
    """Route high-uncertainty subtasks to node_a and low to node_b."""

    def __init__(
        self,
        uncertainty_threshold: float,
        network_feasible_fn: Callable[[], bool],
    ) -> None:
        """Initialize scheduler thresholds and network feasibility callback."""
        if uncertainty_threshold < 0.0 or uncertainty_threshold > 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")
        self.uncertainty_threshold = uncertainty_threshold
        self.network_feasible_fn = network_feasible_fn

    def schedule(self, subtasks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Assign nodes using stored PRO scores and fallback policy."""
        if not isinstance(subtasks, list):
            raise ValueError("subtasks must be a list")

        if not self.network_feasible_fn():
            return [dict(s, assigned_node="node_a", fallback_to_serial=True) for s in subtasks]

        routed: List[Dict[str, object]] = []
        for subtask in subtasks:
            score = float(subtask.get("pro_score", 1.0))
            assigned = "node_a" if score > self.uncertainty_threshold else "node_b"
            routed.append(dict(subtask, assigned_node=assigned, fallback_to_serial=False))
        return routed
