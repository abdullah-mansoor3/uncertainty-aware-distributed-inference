"""Base scheduler interface for distributed inference routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseScheduler(ABC):
    """Abstract interface all schedulers must implement."""

    @abstractmethod
    def schedule(self, subtasks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Assign a target node to each subtask."""
        raise NotImplementedError
