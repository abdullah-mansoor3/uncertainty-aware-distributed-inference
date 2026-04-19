"""Naive round-robin scheduler baseline without uncertainty awareness."""

from __future__ import annotations

from typing import Dict, List

from .base_scheduler import BaseScheduler


class NaiveParallelScheduler(BaseScheduler):
    """Routes subtasks in round-robin order across node_a and node_b."""

    def schedule(self, subtasks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Assign nodes in alternating order."""
        if not isinstance(subtasks, list):
            raise ValueError("subtasks must be a list")

        nodes = ["node_a", "node_b"]
        routed: List[Dict[str, object]] = []
        for idx, subtask in enumerate(subtasks):
            item = dict(subtask)
            item["assigned_node"] = nodes[idx % 2]
            routed.append(item)
        return routed
