"""Tests for uncertainty scoring and classification."""

from src.modules.uncertainty import classify_uncertainty, compute_pro_score


def test_compute_pro_score_valid_range() -> None:
    """PRO score must be normalized to [0, 1] when defined."""
    logprobs = [-0.2, -1.1, -1.8, -2.0]
    score = compute_pro_score(logprobs)
    assert 0.0 <= score <= 1.0


def test_classify_uncertainty_threshold() -> None:
    """Classification should switch around threshold."""
    assert classify_uncertainty(0.8, tau=0.5) == "high"
    assert classify_uncertainty(0.2, tau=0.5) == "low"
