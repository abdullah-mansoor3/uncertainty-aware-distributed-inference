"""Prompt decomposition for distributed inference.

Implements the schema-based decomposition stage from ParallelPrompt (Kolawole et al.,
NeurIPS 2025). The paper's core insight: a parallelizable prompt has a template with
{data} placeholder, a shared context, and a list of data items (or count n). Decomposition
means filling the template once per item to get N independent subtasks.

Key design decisions vs the paper:
  - When the dataset provides pre-extracted iterations (ground_truth), we use them directly.
    Re-running LLM decomposition on a prompt whose schema is already known is wasteful.
  - LLM decomposition is used as a fallback for novel prompts not from the dataset.
  - Rule-based fallback covers the most common ParallelPrompt surface patterns:
    bracket arrays, numbered lists, quoted-item lists, and repeated-generation counts.
  - Dependency check uses a conservative threshold so template-sharing subtasks are NOT
    falsely marked as dependent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# How many tokens two subtasks must share to be flagged as dependent.
# Keep this HIGH — subtasks from the same template share template tokens
# and should NOT be marked as dependent because of that.
_DEPENDENCY_OVERLAP_THRESHOLD = 8


def decompose_prompt(
    prompt: str,
    llm: Any | None = None,
) -> List[Dict[str, Any]]:
    """Decompose a prompt into parallel-safe subtasks.

    Priority order:
      1. Use known_items from dataset (ground_truth / iterations field) if provided.
      2. Try LLM schema extraction.
      3. Fall back to rule-based pattern matching.

    Args:
        prompt: Original input prompt.
        llm: Optional llama-cpp-python model instance.
        
    Returns:
        List of subtask dicts with id/text/dependencies/parallel_safe fields.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    subtasks: List[Dict[str, Any]] = []

    # Path 2: try LLM extraction.
    if not subtasks and llm is not None:
        subtasks = _try_llm_decomposition(prompt=prompt, llm=llm)

    # Path 3: rule-based fallback.
    if not subtasks:
        LOGGER.info("Decomposition fallback triggered for prompt.")
        subtasks = _rule_based_split(prompt)

    checked = check_dependencies(subtasks)
    return merge_dependent_subtasks(checked)


def check_dependencies(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark sequential dependencies using conservative lexical overlap.

    Uses a high overlap threshold (_DEPENDENCY_OVERLAP_THRESHOLD) so subtasks that
    merely share template boilerplate are NOT flagged as dependent. Only flag when
    one subtask uniquely references content that could only come from another subtask's
    output (not the original prompt).

    Args:
        subtasks: Candidate subtasks with text fields.

    Returns:
        Updated subtasks with dependency ids and parallel_safe flags.
    """
    if not isinstance(subtasks, list):
        raise ValueError("subtasks must be a list")

    result: List[Dict[str, Any]] = []
    prior_tokens: List[set[str]] = []

    # Tokens that appear in ALL subtasks are template tokens, not dependency signals.
    # We compute common tokens and exclude them from the overlap check.
    all_token_sets = [
        set(re.findall(r"[A-Za-z0-9_]{4,}", str(s.get("text", "")).lower()))
        for s in subtasks
    ]
    # Tokens shared by every subtask = template boilerplate
    if all_token_sets:
        common_tokens = set.intersection(*all_token_sets) if len(all_token_sets) > 1 else set()
    else:
        common_tokens = set()

    for index, subtask in enumerate(subtasks):
        text = str(subtask.get("text", "")).strip()
        current_tokens = set(re.findall(r"[A-Za-z0-9_]{4,}", text.lower())) - common_tokens
        dependencies: List[int] = []

        for previous_index, previous_tokens in enumerate(prior_tokens):
            # Only count overlap of non-template tokens
            overlap = len(current_tokens & previous_tokens)
            if overlap >= _DEPENDENCY_OVERLAP_THRESHOLD:
                dependencies.append(previous_index)

        result.append({
            "id": int(subtask.get("id", index)),
            "text": text,
            "dependencies": dependencies,
            "parallel_safe": len(dependencies) == 0,
        })
        prior_tokens.append(current_tokens)

    return result


def merge_dependent_subtasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge sequentially-dependent subtasks into serial-only units.

    Args:
        subtasks: Subtasks with dependency metadata.

    Returns:
        Merged list. Parallel-safe subtasks pass through unchanged.
        Dependent groups are concatenated into a single serial task.
    """
    if not isinstance(subtasks, list):
        raise ValueError("subtasks must be a list")

    merged: List[Dict[str, Any]] = []
    current_group: List[Dict[str, Any]] = []

    for subtask in subtasks:
        if subtask.get("dependencies"):
            current_group.append(subtask)
        else:
            if current_group:
                merged.append(_merge_group(current_group, len(merged)))
                current_group = []
            merged.append({
                "id": len(merged),
                "text": subtask["text"],
                "dependencies": [],
                "parallel_safe": True,
            })

    if current_group:
        merged.append(_merge_group(current_group, len(merged)))

    if not merged:
        merged = [{"id": 0, "text": "", "dependencies": [], "parallel_safe": True}]

    return merged


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _items_to_subtasks(items: List[str]) -> List[Dict[str, Any]]:
    """Convert a flat list of item strings into subtask dicts.

    Args:
        items: Iteration items from the dataset.

    Returns:
        Subtask dicts, all marked parallel_safe=True.
    """
    return [
        {"id": index, "text": text, "dependencies": [], "parallel_safe": True}
        for index, text in enumerate(items)
    ]


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from LLM output.

    Args:
        text: Raw LLM completion text.

    Returns:
        Cleaned string ready for JSON parsing.
    """
    # Remove opening fence with optional language tag
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _try_llm_decomposition(prompt: str, llm: Any) -> List[Dict[str, Any]]:
    """Extract parallel subtasks via LLM schema decomposition.

    The instruction asks the model to identify the repeating data items in the prompt
    and return each as a short independent task string. This matches the ParallelPrompt
    schema approach (template + data list) without requiring the model to return the full
    5-field schema — we just want the filled subtask strings.

    Args:
        prompt: Original user prompt.
        llm: Model compatible with llama-cpp-python completion interface.

    Returns:
        Parsed subtask list or empty list on any failure.
    """
    # The instruction is carefully worded to handle both:
    #   - data-list prompts: "Translate X, Y, Z" -> ["Translate X", "Translate Y", "Translate Z"]
    #   - count prompts: "Generate 3 stories" -> ["Generate story 1", "Generate story 2", ...]
    instruction = (
        "You are a task decomposition assistant. Your job is to identify the independent "
        "parallel sub-tasks inside a prompt and return them as a JSON array of strings.\n\n"
        "Rules:\n"
        "- Each string in the array must be a complete, self-contained task.\n"
        "- Tasks must be truly independent — no task should need the output of another.\n"
        "- If the prompt lists N items to process with the same operation, return N tasks.\n"
        "- If the prompt asks to generate N variations, return N generation requests.\n"
        "- If the prompt cannot be decomposed, return a JSON array with one element.\n"
        "- Return ONLY the JSON array. No explanation, no markdown, no extra text.\n\n"
        f"Prompt to decompose:\n{prompt}\n\n"
        "JSON array:"
    )
    try:
        completion = llm(
            instruction,
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
        )
        raw_text = completion["choices"][0]["text"].strip()

        # Strip markdown fences — the most common failure mode
        cleaned = _strip_markdown_fences(raw_text)

        # Find the JSON array even if there's preamble text
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)

        parsed = json.loads(cleaned)
        if not isinstance(parsed, list) or not parsed:
            return []

        items = [str(item).strip() for item in parsed if str(item).strip()]
        if not items:
            return []

        LOGGER.debug("LLM decomposition produced %s subtasks.", len(items))
        return _items_to_subtasks(items)

    except Exception as error:
        LOGGER.warning("LLM decomposition failed: %s", error)
        return []


def _rule_based_split(prompt: str) -> List[Dict[str, Any]]:
    """Fallback decomposition covering the most common ParallelPrompt surface patterns.

    Handles in priority order:
      1. Python/JSON bracket arrays: ["item1", "item2"] or ['item1', 'item2']
      2. Numbered lists: 1. task one  2. task two
      3. Quoted comma-separated items: "item1", "item2", "item3"
      4. "for each of X, Y, Z" patterns
      5. Sentence-splitting on semicolons or newlines as last resort

    Args:
        prompt: Original user prompt.

    Returns:
        List of subtask dicts.
    """
    normalized = re.sub(r"\s+", " ", prompt).strip()
    items: List[str] = []

    # Pattern 1: bracket array ["item1", "item2", ...] — very common in ParallelPrompt
    bracket_match = re.search(r'\[(["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*)\]', normalized)
    if bracket_match:
        raw_items = re.findall(r'["\']([^"\']+)["\']', bracket_match.group(1))
        items = [item.strip() for item in raw_items if item.strip()]
        if len(items) >= 2:
            LOGGER.debug("Rule-based: found %s items via bracket array.", len(items))
            return _items_to_subtasks(items)

    # Pattern 2: numbered list  1. ... 2. ... or 1) ... 2) ...
    numbered = re.split(r"\n\s*\d+[\).]\s+", "\n" + normalized)
    numbered = [part.strip() for part in numbered if part.strip()]
    if len(numbered) >= 2:
        LOGGER.debug("Rule-based: found %s items via numbered list.", len(numbered))
        return _items_to_subtasks(numbered)

    # Pattern 3: quoted comma-separated items "X", "Y", "Z" anywhere in the prompt
    quoted_items = re.findall(r'"([^"]{2,})"', normalized)
    if len(quoted_items) >= 2:
        LOGGER.debug("Rule-based: found %s quoted items.", len(quoted_items))
        return _items_to_subtasks(quoted_items)

    # Pattern 4: "for each of X, Y and Z" or "for X, Y, and Z"
    for_each_match = re.search(
        r"(?:for each of|for|translate|analyze|summarize|classify)\s+(.+?)(?:\.|,\s+(?:and\s+)?(?:provide|give|return|write|list))",
        normalized,
        re.IGNORECASE,
    )
    if for_each_match:
        candidates = re.split(r",\s*(?:and\s+)?", for_each_match.group(1))
        candidates = [c.strip().strip('"\'') for c in candidates if c.strip()]
        if len(candidates) >= 2:
            LOGGER.debug("Rule-based: found %s items via for-each pattern.", len(candidates))
            return _items_to_subtasks(candidates)

    # Pattern 5: bullet points  - item or * item
    bullets = re.split(r"\n\s*[-*•]\s+", "\n" + normalized)
    bullets = [part.strip() for part in bullets if part.strip()]
    if len(bullets) >= 2:
        LOGGER.debug("Rule-based: found %s items via bullet list.", len(bullets))
        return _items_to_subtasks(bullets)

    # Last resort: return the whole prompt as a single task.
    # Better to have one task than a broken split.
    LOGGER.debug("Rule-based: no parallel pattern found, returning single task.")
    return [{"id": 0, "text": normalized, "dependencies": [], "parallel_safe": False}]


def _merge_group(group: List[Dict[str, Any]], new_id: int) -> Dict[str, Any]:
    """Create a serial-only merged subtask from a dependency group.

    Args:
        group: Dependent subtasks to merge.
        new_id: ID for the resulting merged task.

    Returns:
        Single merged subtask dict with parallel_safe=False.
    """
    merged_text = " Then ".join(item["text"] for item in group if item.get("text"))
    return {
        "id": new_id,
        "text": merged_text,
        "dependencies": [],
        "parallel_safe": False,
    }