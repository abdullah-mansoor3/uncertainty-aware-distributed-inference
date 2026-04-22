"""Tests for evaluation metric helpers."""

from src.utils.metrics import compute_latency_stats, compute_rouge


def test_compute_latency_stats_basic() -> None:
    """Latency stats should return deterministic basic values."""
    stats = compute_latency_stats([10.0, 20.0, 30.0])
    assert stats["mean"] == 20.0
    assert stats["max"] == 30.0


def test_compute_rouge_empty_inputs() -> None:
    """ROUGE should return NaN values for empty inputs, not crash."""
    result = compute_rouge("", [])
    assert "rouge1" in result and "rougeL" in result
