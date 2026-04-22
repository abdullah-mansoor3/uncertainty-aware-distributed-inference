"""Master-side output and attribution aggregation.

Combines subtask outputs in prompt order and aggregates local attribution maps
weighted by confidence proxy (1 - PRO uncertainty score).
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def merge_outputs(subtask_outputs: List[str], order: List[int]) -> str:
    """Merge subtask outputs according to original order.

    Args:
        subtask_outputs: Generated subtask outputs.
        order: Output ordering indices.

    Returns:
        Single merged response string.
    """
    if len(subtask_outputs) != len(order):
        raise ValueError("subtask_outputs and order must have equal length")
    ordered_pairs = sorted(zip(order, subtask_outputs), key=lambda item: item[0])
    return "\n".join(text.strip() for _, text in ordered_pairs if text.strip())


def aggregate_attributions(local_maps: List[List[Dict[str, float]]], uncertainty_scores: List[float]) -> List[Dict[str, float]]:
    """Aggregate local attribution vectors into a normalized global map.

    Args:
        local_maps: List of local token attribution maps.
        uncertainty_scores: PRO uncertainty score per map.

    Returns:
        Global normalized token attribution list.
    """
    if len(local_maps) != len(uncertainty_scores):
        raise ValueError("local_maps and uncertainty_scores must have equal length")

    weighted_items: List[Dict[str, float]] = []
    for local_map, score in zip(local_maps, uncertainty_scores):
        confidence_weight = max(0.0, 1.0 - float(score))
        for token_info in local_map:
            weighted_items.append(
                {
                    "token": token_info["token"],
                    "attribution": max(0.0, float(token_info["attribution"])) * confidence_weight,
                }
            )

    total_weight = sum(item["attribution"] for item in weighted_items) or 1.0
    for item in weighted_items:
        item["attribution"] = item["attribution"] / total_weight
    return weighted_items


def render_heatmap(global_map: List[Dict[str, float]]) -> None:
    """Render a token-level attribution heatmap.

    Args:
        global_map: Token attribution items.

    Returns:
        None.
    """
    if not global_map:
        return
    tokens = [item["token"] for item in global_map]
    weights = [item["attribution"] for item in global_map]

    plt.figure(figsize=(max(6, len(tokens) * 0.35), 2.5))
    plt.bar(range(len(tokens)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=70, ha="right")
    plt.tight_layout()
    plt.show()

