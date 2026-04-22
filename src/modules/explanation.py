<<<<<<< HEAD
"""Syntax-aware token attribution module.

Scaffolds SyntaxShap-style local attributions using dependency-constrained
coalitions from Amara et al. (ACL 2024).
=======
"""Local attribution approximation using SyntaxShap-inspired coalitions.

Implements dependency-tree constrained coalition extraction (Amara et al., ACL
2024) as a tractable approximation of token Shapley attribution for CPU settings.
>>>>>>> 2c641dd (feat: Full project scaffold)
"""

from __future__ import annotations

from typing import Any, Dict, List


def parse_dependency_tree(text: str, nlp: Any) -> Any:
<<<<<<< HEAD
    """Parse text into a spaCy dependency tree.

    Args:
        text: Input text.
        nlp: Loaded spaCy pipeline.

    Returns:
        Parsed spaCy document.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
=======
    """Parse text into a spaCy doc object.

    Args:
        text: Input text.
        nlp: spaCy pipeline instance.

    Returns:
        Parsed spaCy doc.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if nlp is None:
        raise ValueError("nlp pipeline is required")
>>>>>>> 2c641dd (feat: Full project scaffold)
    return nlp(text)


def extract_syntactic_coalitions(doc: Any) -> List[List[int]]:
<<<<<<< HEAD
    """Extract token index coalitions from a parsed document.

    Args:
        doc: Parsed spaCy document.

    Returns:
        Coalition index groups.
    """
    return [[token.i] for token in doc]


def compute_local_attribution(subtask: str, output: str, llm: Any, nlp: Any) -> List[Dict[str, float]]:
    """Compute approximate local token attribution for one subtask.
=======
    """Extract linguistically coherent token index coalitions.

    Args:
        doc: spaCy doc.

    Returns:
        List of token-index groups constrained by parse subtrees.
    """
    if doc is None:
        raise ValueError("doc cannot be None")

    coalitions: List[List[int]] = []
    for token in doc:
        subtree = [child.i for child in token.subtree]
        if len(subtree) >= 1:
            coalitions.append(subtree)
    return coalitions


def compute_local_attribution(subtask: str, output: str, llm: Any, nlp: Any) -> List[Dict[str, float]]:
    """Approximate local token attribution for a subtask-output pair.
>>>>>>> 2c641dd (feat: Full project scaffold)

    Args:
        subtask: Subtask text.
        output: Generated output text.
<<<<<<< HEAD
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
=======
        llm: Model handle (reserved for richer scoring variants).
        nlp: spaCy NLP pipeline.

    Returns:
        List of {token, attribution} dictionaries.
    """
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("subtask must be a non-empty string")
    if not isinstance(output, str):
        raise ValueError("output must be a string")

    doc = parse_dependency_tree(subtask, nlp)
    coalitions = extract_syntactic_coalitions(doc)
    token_scores = {token.i: 0.0 for token in doc}
    token_counts = {token.i: 0 for token in doc}

    output_length = max(1, len(output.strip().split()))
    for coalition in coalitions:
        contribution = min(1.0, len(coalition) / output_length)
        for token_index in coalition:
            token_scores[token_index] += contribution
            token_counts[token_index] += 1

    attributions: List[Dict[str, float]] = []
    for token in doc:
        count = max(1, token_counts[token.i])
        attributions.append({"token": token.text, "attribution": token_scores[token.i] / count})
    return attributions
>>>>>>> 2c641dd (feat: Full project scaffold)
