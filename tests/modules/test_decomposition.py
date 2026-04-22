"""Tests for decomposition module fallback and dependency logic."""

from src.modules.decomposition import decompose_prompt


def test_decompose_prompt_rule_based_split() -> None:
    """Rule-based fallback should split prompt on discourse markers."""
    prompt = "Explain entropy and also give one routing example additionally mention one limitation"
    subtasks = decompose_prompt(prompt, llm=None)
    assert len(subtasks) >= 2
    assert all("text" in item for item in subtasks)
