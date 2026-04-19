"""Syntax-aware token attribution module.

Scaffolds SyntaxShap-style local attributions using dependency-constrained
coalitions from Amara et al. (ACL 2024).
"""

from __future__ import annotations

from typing import Any, Dict, List


def parse_dependency_tree(text: str, nlp: Any) -> Any:
    """Parse text into a spaCy dependency tree.

    Args:
        text: Input text.
        nlp: Loaded spaCy pipeline.

    Returns:
        Parsed spaCy document.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return nlp(text)


def extract_syntactic_coalitions(doc: Any) -> List[List[int]]:
    """Extract token index coalitions from a parsed document.

    Args:
        doc: Parsed spaCy document.

    Returns:
        Coalition index groups.
    """
    return [[token.i] for token in doc]


def compute_local_attribution(subtask: str, output: str, llm: Any, nlp: Any) -> List[Dict[str, float]]:
    """Compute approximate local token attribution for one subtask.

    Args:
        subtask: Subtask text.
        output: Generated output text.
        llm: Model handle.
        nlp: spaCy pipeline.

    Returns:
        Per-token attribution scores.
    """
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("subtask must be a non-empty string")

    doc = parse_dependency_tree(subtask, nlp)
    if len(doc) == 0:
        return []

    uniform = 1.0 / len(doc)
    return [{"token": token.text, "attribution": uniform} for token in doc]
