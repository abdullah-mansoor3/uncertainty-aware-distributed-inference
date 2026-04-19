"""Uncertainty and calibration module.

Implements PRO-style entropy scoring and rank calibration metrics based on
Nguyen et al. (2025) and Huang et al. (EMNLP 2024).
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency in early scaffold.
    roc_auc_score = None


def compute_pro_score(completion_logprobs: List[float], adaptive_k: bool = True) -> float:
    """Compute normalized entropy from top-K log-probabilities.

    Args:
        completion_logprobs: Token log-probabilities.
        adaptive_k: Whether to prune weak probabilities before entropy.

    Returns:
        Normalized entropy score in [0, 1].
    """
    if not completion_logprobs:
        return float("nan")

    probs = [math.exp(lp) for lp in completion_logprobs]
    if adaptive_k:
        probs = [p for p in probs if p > 1e-8]
    if not probs:
        return float("nan")

    total = sum(probs)
    norm = [p / total for p in probs]
    k = len(norm)
    if k <= 1:
        return 0.0

    # PRO entropy: H = -sum(p_i * log(p_i)), normalized by log(K).
    entropy = -sum(p * math.log(p) for p in norm if p > 0.0)
    return float(max(0.0, min(1.0, entropy / math.log(k))))


def classify_uncertainty(score: float, tau: float = 0.5) -> str:
    """Classify uncertainty into high or low bucket.

    Args:
        score: PRO score.
        tau: Decision threshold.

    Returns:
        "high" when score > tau, otherwise "low".
    """
    if math.isnan(score):
        return "high"
    return "high" if score > tau else "low"


def compute_erce(scores: List[float], correctness: List[float], n_bins: int = 10) -> float:
    """Compute Expected Rank Calibration Error (ERCE).

    Args:
        scores: Uncertainty scores.
        correctness: Correctness values in [0, 1].
        n_bins: Number of rank bins.

    Returns:
        ERCE value, or NaN when inputs are invalid.
    """
    if len(scores) != len(correctness) or len(scores) == 0:
        return float("nan")

    s = np.asarray(scores, dtype=float)
    c = np.asarray(correctness, dtype=float)
    order = np.argsort(s)
    c_sorted = c[order]

    bins = np.array_split(np.arange(len(c_sorted)), n_bins)
    if not bins:
        return float("nan")

    bin_means = []
    for bucket in bins:
        if len(bucket) == 0:
            continue
        bin_means.append(float(np.nanmean(c_sorted[bucket])))

    if not bin_means:
        return float("nan")

    correctness_cdf = np.cumsum(bin_means) / max(np.sum(bin_means), 1e-12)
    uncertainty_cdf = np.linspace(1.0 / len(bin_means), 1.0, len(bin_means))
    return float(np.mean(np.abs(correctness_cdf - uncertainty_cdf)))


def compute_auroc(scores: List[float], correctness: List[float]) -> float:
    """Compute AUROC for uncertainty-to-error ranking quality.

    Args:
        scores: Uncertainty scores.
        correctness: Correctness values in [0, 1].

    Returns:
        AUROC value or NaN if unavailable.
    """
    if roc_auc_score is None:
        return float("nan")
    if len(scores) != len(correctness) or len(scores) == 0:
        return float("nan")

    labels = [1 if c < 0.5 else 0 for c in correctness]
    if len(set(labels)) < 2:
        return float("nan")

    return float(roc_auc_score(labels, scores))
