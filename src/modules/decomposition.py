"""Prompt decomposition for distributed inference.

Implements the decomposition stage inspired by ParallelPrompt (Kolawole et al.,
NeurIPS 2025) with robust fallback logic for constrained CPU settings.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


def decompose_prompt(prompt: str, llm: Any | None = None) -> List[Dict[str, Any]]:
    """Decompose a prompt into potentially parallel-safe subtasks.

    Args:
        prompt: Original input prompt.
        llm: Optional llama-cpp-python model instance for JSON decomposition.

    Returns:
        List of subtask dictionaries with id/text/dependencies/parallel_safe.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    subtasks: List[Dict[str, Any]] = []
    if llm is not None:
        subtasks = _try_llm_decomposition(prompt=prompt, llm=llm)

    if not subtasks:
        LOGGER.info("Decomposition fallback triggered for prompt.")
        subtasks = _rule_based_split(prompt)

    checked = check_dependencies(subtasks)
    return merge_dependent_subtasks(checked)


def check_dependencies(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark likely dependencies using simple lexical overlap heuristics.

    Args:
        subtasks: Candidate subtasks with text fields.

    Returns:
        Updated subtasks with dependency ids and parallel safety flags.
    """
    if not isinstance(subtasks, list):
        raise ValueError("subtasks must be a list")

    result: List[Dict[str, Any]] = []
    prior_tokens: List[set[str]] = []

    for index, subtask in enumerate(subtasks):
        text = str(subtask.get("text", "")).strip()
        current_tokens = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
        dependencies: List[int] = []

        for previous_index, previous_tokens in enumerate(prior_tokens):
            overlap = len(current_tokens & previous_tokens)
            if overlap >= 3:
                dependencies.append(previous_index)

        updated = {
            "id": int(subtask.get("id", index)),
            "text": text,
            "dependencies": dependencies,
            "parallel_safe": len(dependencies) == 0,
        }
        result.append(updated)
        prior_tokens.append(current_tokens)

    return result


def merge_dependent_subtasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge dependent subtasks into serial-only units.

    Args:
        subtasks: Subtasks with dependency metadata.

    Returns:
        Merged list preserving original order semantics.
    """
    if not isinstance(subtasks, list):
        raise ValueError("subtasks must be a list")

    merged: List[Dict[str, Any]] = []
    current_group: List[Dict[str, Any]] = []

    for subtask in subtasks:
        if subtask.get("dependencies"):
            current_group.append(subtask)
        else:
            if current_group:
                merged.append(_merge_group(current_group, len(merged)))
                current_group = []
            merged.append(
                {
                    "id": len(merged),
                    "text": subtask["text"],
                    "dependencies": [],
                    "parallel_safe": True,
                }
            )

    if current_group:
        merged.append(_merge_group(current_group, len(merged)))

    if not merged:
        merged = [{"id": 0, "text": "", "dependencies": [], "parallel_safe": True}]

    return merged


def _try_llm_decomposition(prompt: str, llm: Any) -> List[Dict[str, Any]]:
    """Attempt JSON-array decomposition using the language model.

    Args:
        prompt: Original prompt.
        llm: Model callable compatible with llama-cpp-python.

    Returns:
        Parsed subtask list or empty list on any failure.
    """
    instruction = (
        "Return ONLY a JSON array of independent sub-questions as strings for this prompt:\n"
        f"{prompt}"
    )
    try:
        completion = llm(
            instruction,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
        )
        text = completion["choices"][0]["text"].strip()
        parsed = json.loads(text)
        if not isinstance(parsed, list) or not parsed:
            return []
        return [
            {"id": index, "text": str(item).strip(), "dependencies": [], "parallel_safe": True}
            for index, item in enumerate(parsed)
            if str(item).strip()
        ]
    except Exception as error:  # pragma: no cover - best-effort fallback path
        LOGGER.warning("LLM decomposition failed: %s", error)
        return []


def _rule_based_split(prompt: str) -> List[Dict[str, Any]]:
    """Fallback decomposition by separators and discourse markers.

    Args:
        prompt: Original prompt.

    Returns:
        List of subtask dictionaries.
    """
    normalized = re.sub(r"\s+", " ", prompt).strip()
    split_pattern = r"(?:\n\s*[-*]\s+|\n\s*\d+[\).]\s+|\band also\b|\badditionally\b)"
    parts = [part.strip(" .") for part in re.split(split_pattern, normalized, flags=re.IGNORECASE)]
    parts = [part for part in parts if part]

    if not parts:
        parts = [normalized]

    return [
        {"id": index, "text": text, "dependencies": [], "parallel_safe": True}
        for index, text in enumerate(parts)
    ]


def _merge_group(group: List[Dict[str, Any]], new_id: int) -> Dict[str, Any]:
    """Create a serial-only merged subtask from a dependency group.

    Args:
        group: Group of dependent subtasks.
        new_id: New merged subtask id.

    Returns:
        Single merged subtask dictionary.
    """
    merged_text = " Then ".join(item["text"] for item in group if item.get("text"))
    return {
        "id": new_id,
        "text": merged_text,
        "dependencies": [],
        "parallel_safe": False,
    }
