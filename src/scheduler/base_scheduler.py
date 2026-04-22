<<<<<<< HEAD
"""Base scheduler interface for distributed inference routing."""
=======
"""Scheduler interface for prompt subtask assignment.

Defines the abstract scheduling contract for serial, naive, and uncertainty-aware
policies in the distributed inference framework.
"""
>>>>>>> 2c641dd (feat: Full project scaffold)

from __future__ import annotations

from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import Dict, List


class BaseScheduler(ABC):
    """Abstract interface all schedulers must implement."""

    @abstractmethod
    def schedule(self, subtasks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Assign a target node to each subtask."""
=======
from typing import Any, Dict, List


class BaseScheduler(ABC):
    """Abstract scheduler for subtask node assignment."""

    @abstractmethod
    def schedule(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign subtasks to nodes and return updated subtask dicts.

        Args:
            subtasks: List of subtask dictionaries.

        Returns:
            List of updated subtask dictionaries containing assigned node metadata.
        """
>>>>>>> 2c641dd (feat: Full project scaffold)
        raise NotImplementedError
