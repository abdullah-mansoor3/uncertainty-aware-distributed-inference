"""Evaluation metric utilities for correctness and latency summaries."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_rouge(prediction: str, references: List[str]) -> Dict[str, float]:
    """Compute ROUGE placeholders.

    This scaffold returns NaN until the dedicated metric backend is wired.
    """
    if not references:
        return {"rouge1": float("nan"), "rougeL": float("nan")}
    return {"rouge1": float("nan"), "rougeL": float("nan")}


def compute_meteor(prediction: str, references: List[str]) -> float:
    """Compute METEOR placeholder."""
    if not references:
        return float("nan")
    return float("nan")


def compute_bleu(prediction: str, references: List[str]) -> float:
    """Compute BLEU placeholder."""
    if not references:
        return float("nan")
    return float("nan")


def compute_bert_score(prediction: str, references: List[str]) -> float:
    """Compute BERTScore placeholder."""
    if not references:
        return float("nan")
    return float("nan")


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute latency summary statistics."""
    if not latencies:
        return {
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    values = np.asarray(latencies, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }
