<<<<<<< HEAD
"""Uncertainty and calibration module.

Implements PRO-style entropy scoring and rank calibration metrics based on
Nguyen et al. (2025) and Huang et al. (EMNLP 2024).
=======
"""Uncertainty scoring and calibration diagnostics.

Implements PRO-style entropy scoring from top-K token probabilities (Nguyen et al.,
2025) and rank-calibration helpers aligned with ERCE/AUROC workflows (Huang et
al., EMNLP 2024).
>>>>>>> 2c641dd (feat: Full project scaffold)
"""

from __future__ import annotations

<<<<<<< HEAD
import math
=======
>>>>>>> 2c641dd (feat: Full project scaffold)
from typing import List

import numpy as np

<<<<<<< HEAD
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
=======

def compute_pro_score(completion_logprobs: List[float], adaptive_k: bool = True) -> float:
    """Compute normalized entropy PRO score from token log-probabilities.

    Args:
        completion_logprobs: List of log-probabilities from top-K tokens.
        adaptive_k: Whether to filter very low-probability items.

    Returns:
        Normalized entropy in [0, 1], or NaN when undefined.
    """
    if not isinstance(completion_logprobs, list) or len(completion_logprobs) == 0:
        return float("nan")

    logprob_array = np.array(completion_logprobs, dtype=float)
    logprob_array = logprob_array[~np.isnan(logprob_array)]
    if logprob_array.size == 0:
        return float("nan")

    probability_array = np.exp(logprob_array)

    if adaptive_k:
        probability_array = probability_array[probability_array >= 1e-6]

    if probability_array.size < 2:
        return float("nan")

    probability_sum = np.sum(probability_array)
    if probability_sum <= 0.0:
        return float("nan")
    probability_array = probability_array / probability_sum
    entropy = -np.sum(probability_array * np.log(probability_array + 1e-12))
    normalizer = np.log(float(probability_array.size))
    if normalizer <= 0.0:
        return float("nan")

    score = float(entropy / normalizer)
    return max(0.0, min(1.0, score))


def classify_uncertainty(score: float, tau: float = 0.5) -> str:
    """Classify uncertainty as high or low.

    Args:
        score: PRO score in [0, 1].
        tau: Threshold above which uncertainty is high.

    Returns:
        "high" when score > tau, else "low".
    """
    if not isinstance(score, (float, int)):
        raise ValueError("score must be numeric")
    return "high" if float(score) > float(tau) else "low"


def compute_erce(scores: List[float], correctness: List[float], n_bins: int = 10) -> float:
    """Compute expected rank calibration error (ERCE).

    Args:
        scores: Uncertainty scores.
        correctness: Correctness values (0..1) aligned with scores.
        n_bins: Number of equal-width bins on rank-normalized uncertainty.

    Returns:
        ERCE value where lower is better calibration.
    """
    if len(scores) != len(correctness) or len(scores) == 0:
        return float("nan")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    rank_order = np.argsort(np.array(scores, dtype=float))
    sorted_correctness = np.array(correctness, dtype=float)[rank_order]
    sample_count = len(sorted_correctness)
    uncertainty_cdf = np.linspace(1.0 / sample_count, 1.0, sample_count)
    correctness_cdf = np.cumsum(sorted_correctness) / (np.sum(sorted_correctness) + 1e-12)

    edges = np.linspace(0, sample_count, n_bins + 1, dtype=int)
    bin_errors = []
    for index in range(n_bins):
        start = edges[index]
        end = edges[index + 1]
        if end <= start:
            continue
        deviation = np.abs(correctness_cdf[start:end] - uncertainty_cdf[start:end])
        bin_errors.append(np.mean(deviation))

    if not bin_errors:
        return float("nan")
    return float(np.mean(bin_errors))


def compute_auroc(scores: List[float], correctness: List[float]) -> float:
    """Compute AUROC for uncertainty as an error detector.

    Args:
        scores: Uncertainty scores where higher means less confidence.
        correctness: Correctness values in [0, 1].

    Returns:
        AUROC of classifying incorrect items, or NaN if undefined.
    """
    if len(scores) != len(correctness) or len(scores) == 0:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")

    is_error = np.array([1 if value < 0.5 else 0 for value in correctness], dtype=int)
    if np.unique(is_error).size < 2:
        return float("nan")
    return float(roc_auc_score(is_error, np.array(scores, dtype=float)))
>>>>>>> 2c641dd (feat: Full project scaffold)
