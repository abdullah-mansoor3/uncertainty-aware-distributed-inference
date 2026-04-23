"""Serial scheduler baseline.

Implements the strict serial pipeline where all subtasks are executed on node A
in sequence, providing the reference baseline for later parallel variants.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.scheduler.base_scheduler import BaseScheduler


class SerialScheduler(BaseScheduler):
    """Assign all subtasks to the master node for sequential execution."""

    def schedule(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign every subtask to node_a.

        Args:
            subtasks: Decomposed subtasks.

        Returns:
            Subtasks with assigned_node set to node_a.
        """
        if not isinstance(subtasks, list):
            raise ValueError("subtasks must be a list")

        scheduled: List[Dict[str, Any]] = []
        for subtask in subtasks:
            updated = dict(subtask)
            updated["assigned_node"] = "node_a"
            scheduled.append(updated)
        return scheduled
