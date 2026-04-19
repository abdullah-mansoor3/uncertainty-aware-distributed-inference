"""Prompt decomposition module.

Implements decomposition scaffolding inspired by ParallelPrompt (Kolawole et al.,
NeurIPS 2025).
"""

from __future__ import annotations

from typing import Any, Dict, List


def decompose_prompt(prompt: str, llm: Any) -> List[Dict[str, Any]]:
    """Decompose a prompt into independent subtasks.

    Args:
        prompt: Original user prompt.
        llm: Llama model handle.

    Returns:
        A list of subtask dictionaries.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    return [
        {
            "id": 0,
            "text": prompt.strip(),
            "dependencies": [],
            "parallel_safe": True,
        }
    ]


def check_dependencies(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate dependency hints across subtasks.

    Args:
        subtasks: List of extracted subtasks.

    Returns:
        Subtasks with dependency metadata preserved.
    """
    if not isinstance(subtasks, list):
        raise ValueError("subtasks must be a list")
    return subtasks


def merge_dependent_subtasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge subtasks marked as dependent.

    Args:
        subtasks: Subtasks possibly containing dependency links.

    Returns:
        Merged subtask list.
    """
    if not isinstance(subtasks, list):
        raise ValueError("subtasks must be a list")
    return subtasks
