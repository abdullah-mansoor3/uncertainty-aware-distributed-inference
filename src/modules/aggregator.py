"""Aggregation module for distributed outputs and attributions.

Implements master-side merge and weighted attribution aggregation.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def merge_outputs(subtask_outputs: List[str], order: List[int]) -> str:
    """Merge subtask outputs in original prompt order.

    Args:
        subtask_outputs: Generated outputs.
        order: Original subtask indices.

    Returns:
        Combined output string.
    """
    if len(subtask_outputs) != len(order):
        raise ValueError("subtask_outputs and order must have the same length")

    ordered = [text for _, text in sorted(zip(order, subtask_outputs), key=lambda x: x[0])]
    return "\n".join(ordered)


def aggregate_attributions(local_maps: List[List[Dict[str, float]]], uncertainty_scores: List[float]) -> List[Dict[str, float]]:
    """Aggregate local attribution maps using uncertainty-based weighting.

    Args:
        local_maps: Attribution maps from each subtask/node.
        uncertainty_scores: PRO scores aligned to local_maps.

    Returns:
        Global normalized token attribution map.
    """
    if len(local_maps) != len(uncertainty_scores):
        raise ValueError("local_maps and uncertainty_scores must have same length")

    weighted: List[Dict[str, float]] = []
    total = 0.0
    for local_map, score in zip(local_maps, uncertainty_scores):
        weight = max(0.0, 1.0 - float(score))
        for item in local_map:
            value = float(item.get("attribution", 0.0)) * weight
            weighted.append({"token": str(item.get("token", "")), "attribution": value})
            total += value

    if total <= 0.0:
        return weighted

    return [{"token": x["token"], "attribution": x["attribution"] / total} for x in weighted]


def render_heatmap(global_map: List[Dict[str, float]]) -> None:
    """Render a simple attribution heatmap.

    Args:
        global_map: Normalized token attribution map.
    """
    if not global_map:
        return

    tokens = [x["token"] for x in global_map]
    values = [x["attribution"] for x in global_map]

    plt.figure(figsize=(max(8, len(tokens) * 0.4), 2.5))
    plt.imshow([values], aspect="auto", cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=60, ha="right")
    plt.yticks([])
    plt.tight_layout()
