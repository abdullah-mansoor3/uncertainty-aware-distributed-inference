"""Serial scheduler baseline that routes all subtasks to node_a."""

from __future__ import annotations

from typing import Dict, List

from .base_scheduler import BaseScheduler


class SerialScheduler(BaseScheduler):
    """Schedules all subtasks to the master node for serial execution."""

    def schedule(self, subtasks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Assign node_a to every subtask."""
        if not isinstance(subtasks, list):
            raise ValueError("subtasks must be a list")

        routed: List[Dict[str, object]] = []
        for subtask in subtasks:
            item = dict(subtask)
            item["assigned_node"] = "node_a"
            routed.append(item)
        return routed
