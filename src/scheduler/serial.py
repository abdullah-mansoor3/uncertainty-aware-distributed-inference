<<<<<<< HEAD
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
=======
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
>>>>>>> 2c641dd (feat: Full project scaffold)
