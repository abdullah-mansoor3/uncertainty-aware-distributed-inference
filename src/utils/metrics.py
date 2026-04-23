"""Evaluation metrics for correctness and latency.

Provides ROUGE, BLEU, METEOR, BERTScore wrappers and latency summary helpers for
experiment reporting.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_rouge(prediction: str, references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1 and ROUGE-L F-measure.

    Args:
        prediction: Generated text.
        references: List of reference answers.

    Returns:
        Dictionary with rouge1 and rougeL scores.
    """
    if not prediction or not references:
        return {"rouge1": float("nan"), "rougeL": float("nan")}

    try:
        from rouge_score import rouge_scorer
    except Exception:
        return {"rouge1": float("nan"), "rougeL": float("nan")}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rouge_l_scores = []
    for reference in references:
        values = scorer.score(reference, prediction)
        rouge1_scores.append(values["rouge1"].fmeasure)
        rouge_l_scores.append(values["rougeL"].fmeasure)
    return {"rouge1": float(np.mean(rouge1_scores)), "rougeL": float(np.mean(rouge_l_scores))}


def compute_meteor(prediction: str, references: List[str]) -> float:
    """Compute METEOR over multiple references.

    Args:
        prediction: Generated text.
        references: List of references.

    Returns:
        Mean METEOR score or NaN for empty input.
    """
    if not prediction or not references:
        return float("nan")
    try:
        from nltk.translate.meteor_score import meteor_score
    except Exception:
        return float("nan")
    pred_tokens = prediction.split()
    scores = [meteor_score([reference.split()], pred_tokens) for reference in references if reference]
    return float(np.mean(scores)) if scores else float("nan")


def compute_bleu(prediction: str, references: List[str]) -> float:
    """Compute sentence BLEU with multiple references.

    Args:
        prediction: Generated text.
        references: List of references.

    Returns:
        BLEU score or NaN for empty input.
    """
    if not prediction or not references:
        return float("nan")
    try:
        from nltk.translate.bleu_score import sentence_bleu
    except Exception:
        return float("nan")
    reference_tokens = [reference.split() for reference in references if reference]
    if not reference_tokens:
        return float("nan")
    return float(sentence_bleu(reference_tokens, prediction.split()))


def compute_bert_score(prediction: str, references: List[str]) -> float:
    """Compute BERTScore F1 against references.

    Args:
        prediction: Generated text.
        references: Reference answers.

    Returns:
        Mean BERTScore F1 or NaN if inputs are invalid.
    """
    if not prediction or not references:
        return float("nan")
    try:
        from bert_score import score as bert_score
        _, _, f1_values = bert_score([prediction] * len(references), references, lang="en", verbose=False)
        return float(f1_values.mean().item())
    except Exception:
        return float("nan")


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute latency summary statistics.

    Args:
        latencies: Latency values in milliseconds.

    Returns:
        mean, p95, p99, max latency dictionary.
    """
    if not latencies:
        return {"mean": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}

    latency_array = np.array(latencies, dtype=float)
    return {
        "mean": float(np.mean(latency_array)),
        "p95": float(np.percentile(latency_array, 95)),
        "p99": float(np.percentile(latency_array, 99)),
        "max": float(np.max(latency_array)),
    }
