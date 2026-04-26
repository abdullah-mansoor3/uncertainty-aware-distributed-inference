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
    if llm is not None:
        subtasks = _try_llm_decomposition(prompt=prompt, llm=llm)

    # Path 3: rule-based fallback.
    # If LLM returns only one task for an obviously decomposable prompt,
    # prefer the richer rule-based decomposition.
    rule_based_subtasks = _rule_based_split(prompt)
    if not subtasks:
        LOGGER.info("Decomposition fallback triggered for prompt.")
        subtasks = rule_based_subtasks
    elif len(subtasks) <= 1 < len(rule_based_subtasks):
        LOGGER.debug(
            "Overriding single-task LLM decomposition with rule-based split (%s tasks).",
            len(rule_based_subtasks),
        )
        subtasks = rule_based_subtasks

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
    index_to_group: Dict[int, int] = {}

    for index, subtask in enumerate(subtasks):
        text = str(subtask.get("text", "")).strip()
        if not text:
            continue

        # If this subtask depends on earlier subtasks, merge into that anchor group.
        dependencies = [int(dep) for dep in subtask.get("dependencies", []) if int(dep) in index_to_group]
        if dependencies:
            group_index = index_to_group[dependencies[0]]
            merged[group_index]["text"] = f"{merged[group_index]['text']} Then {text}"
            merged[group_index]["parallel_safe"] = False
            index_to_group[index] = group_index
            continue

        # New independent group.
        merged.append({
            "id": len(merged),
            "text": text,
            "dependencies": [],
            "parallel_safe": True,
        })
        index_to_group[index] = len(merged) - 1

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
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _parse_llm_item_list(text: str) -> List[str]:
    """Parse a list of subtasks from LLM output text.

    This is intentionally robust for noisy outputs from llama-cpp-python. We prefer
    strict JSON arrays, but also accept quoted item lists or line-separated items.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Prefer the first JSON array if the model returns extra text.
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    candidate = array_match.group(0) if array_match else text

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    quoted_items: List[str] = []
    for match in re.findall(r'"([^"\n]{2,})"|\'([^\'\n]{2,})\'|“([^”\n]{2,})”|‘([^’\n]{2,})’', candidate):
        item = next((group for group in match if group), "").strip()
        if item:
            quoted_items.append(item)
    if len(quoted_items) >= 2:
        return quoted_items

    lines = [line.strip(' -•*"\'“”‘’') for line in candidate.splitlines() if line.strip()]
    if len(lines) >= 2:
        return [line for line in lines if len(line) > 2]

    return []

def _try_llm_decomposition(prompt: str, llm: Any) -> List[Dict[str, Any]]:
    """Extract parallel subtasks via LLM with chain-of-thought and few-shot examples.
 
    Key improvements over original:
    - Chain-of-thought: model reasons about independence before outputting JSON
    - More few-shot examples covering edge cases (single task, answer options, NER)
    - Explicit negative examples so model learns what NOT to split
    - JSON output is on its own line after reasoning, easier to extract
    """
 
    instruction = (
        "You are a task decomposition assistant. "
        "Decide if a prompt contains multiple INDEPENDENT tasks that can run in parallel, "
        "then output them as a JSON array.\n\n"
 
        "Rules:\n"
        "- Tasks are independent if each one can be completed WITHOUT the result of any other.\n"
        "- Answer options / choices are NOT independent tasks. A question with options is ONE task.\n"
        "- If the whole prompt is one task, return a JSON array with that one task as a string.\n"
        "- Output ONLY the JSON array on the last line. No markdown, no extra text after the array.\n\n"
 
        "--- EXAMPLES ---\n\n"
 
        "Prompt: Translate each of these: [\"Hello\", \"Goodbye\", \"Thank you\"]\n"
        "Reasoning: Three separate translation jobs, each fully independent.\n"
        'Output: ["Translate: Hello", "Translate: Goodbye", "Translate: Thank you"]\n\n'
 
        "Prompt: Rate each sentence 1-10: 1. The book is brown. 2. The book are brown.\n"
        "Reasoning: Two independent rating tasks, one per sentence.\n"
        'Output: ["Rate 1-10: The book is brown.", "Rate 1-10: The book are brown."]\n\n'
 
        "Prompt: Extract all named entities from: Apple was founded by Steve Jobs in Cupertino.\n"
        "Reasoning: One extraction task over one document. Cannot be split further.\n"
        '["Extract named entities from: Apple was founded by Steve Jobs in Cupertino."]\n\n'
 
        'Prompt: "Where is Paris?" Options are: "France", "Germany", "Italy". Choose one.\n'
        "Reasoning: This is ONE question with answer options. Options are not independent tasks.\n"
        '["Answer: Where is Paris? Options: France, Germany, Italy."]\n\n'
 
        "Prompt: Extract every drug, its target, and mechanism from this document as "
        "( drug | Target | mechanism ) triplets. Document mentions trastuzumab, pertuzumab, "
        "and Les-4367.\n"
        "Reasoning: Three drugs are mentioned, each triplet is independent.\n"
        '["Extract triplet for trastuzumab", "Extract triplet for pertuzumab", "Extract triplet for Les-4367"]\n\n'
 
        'Prompt: "What color is the sky?" The options are "red", "blue", "green", "yellow". Choose one.\n'
        "Reasoning: ONE question asking to pick from options. Options are not independent tasks.\n"
        '["Answer this question by choosing one option: What color is the sky? Options: red, blue, green, yellow."]\n\n'
 
        "--- NOW DECOMPOSE THIS ---\n\n"
 
        f"Prompt: {prompt}\n"
        "Reasoning:"
    )
 
    try:
        completion = llm(
            instruction,
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
        raw_text = completion["choices"][0]["text"].strip()
        LOGGER.debug("LLM raw output: %s", raw_text)
 
        # The model reasons first, then outputs the JSON array on the last line.
        # Find the last [...] block in the output.
        all_arrays = re.findall(r"\[.*?\]", raw_text, re.DOTALL)
        if not all_arrays:
            LOGGER.warning("LLM produced no JSON array in output: %s", raw_text[:200])
            return []
 
        # Take the last array found — after the chain-of-thought reasoning
        raw_array = all_arrays[-1]
        cleaned = _strip_markdown_fences(raw_array)
 
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
    """Fallback decomposition covering ParallelPrompt surface patterns.

    Args:
        prompt: Original user prompt.

    Returns:
        List of subtask dicts.
    """
    raw_prompt = str(prompt).strip()
    normalized = re.sub(r"\s+", " ", raw_prompt).strip()
    items: List[str] = []

    # Pattern 1: bracket array ["item1", "item2"] — most common in ParallelPrompt
    # Looser match: find [...] block then extract quoted strings inside it
    bracket_match = re.search(r'\[([^\[\]]{5,})\]', normalized)
    if bracket_match:
        inner = bracket_match.group(1)
        # Try double-quoted items first
        raw_items = re.findall(r'"([^"]{2,})"', inner)
        if not raw_items:
            # Try single-quoted items
            raw_items = re.findall(r"'([^']{2,})'", inner)
        if not raw_items:
            # Comma-separated unquoted items
            raw_items = [x.strip() for x in inner.split(",") if x.strip()]
        items = [item.strip() for item in raw_items if item.strip()]
        if len(items) >= 2:
            LOGGER.debug("Rule-based: found %s items via bracket array.", len(items))
            return _items_to_subtasks(items)

    # Pattern 2: numbered list  1. ... 2. ... or 1) ... 2) ...
    numbered = re.findall(
        r"(?:^|\n)\s*\d+[\).]\s+(.*?)(?=(?:\n\s*\d+[\).]\s+)|\Z)",
        raw_prompt,
        flags=re.DOTALL,
    )
    numbered = [part.strip().rstrip(".") for part in numbered if part.strip()]
    if len(numbered) >= 2:
        LOGGER.debug("Rule-based: found %s items via numbered list.", len(numbered))
        return _items_to_subtasks(numbered)

    # Pattern 3: bullet points - item or * item
    bullets = re.split(r"(?:^|\n)\s*[-*•]\s+", raw_prompt)
    bullets = [part.strip() for part in bullets if part.strip()]
    if len(bullets) >= 2:
        LOGGER.debug("Rule-based: found %s items via bullet list.", len(bullets))
        return _items_to_subtasks(bullets)

    # Pattern 4: repeated quoted items on separate lines
    quoted_lines: List[str] = []
    for line in raw_prompt.splitlines():
        stripped = line.strip()
        if re.match(r'^(?:["“‘]).+(?:["”’])$', stripped):
            quoted_lines.append(stripped.strip('"“”‘’'))
    if len(quoted_lines) >= 2:
        LOGGER.debug("Rule-based: found %s items via repeated quoted lines.", len(quoted_lines))
        return _items_to_subtasks(quoted_lines)

    # Pattern 5: sentences separated by semicolons (common in NER/sentiment prompts)
    semicolons = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(semicolons) >= 2:
        LOGGER.debug("Rule-based: found %s items via semicolons.", len(semicolons))
        return _items_to_subtasks(semicolons)

    # Pattern 5: discourse connectors often used for independent asks
    # e.g. "Explain X and also give Y additionally mention Z"
    connector_split = re.split(
        r"\b(?:and also|additionally|also|as well as|plus)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    connector_split = [part.strip(" ,.-") for part in connector_split if part.strip(" ,.-")]
    if len(connector_split) >= 2:
        LOGGER.debug("Rule-based: found %s items via discourse connectors.", len(connector_split))
        return _items_to_subtasks(connector_split)

    # Pattern 6: labeled tasks (e.g., "Easy: ... Medium: ... Hard: ...")
    labeled_chunks = re.findall(
        r"([A-Za-z][A-Za-z0-9_\-/ ]{1,20}:\s*.*?)(?=(?:\s+[A-Za-z][A-Za-z0-9_\-/ ]{1,20}:\s*)|$)",
        normalized,
    )
    labeled_chunks = [chunk.strip() for chunk in labeled_chunks if chunk.strip()]
    if len(labeled_chunks) >= 2:
        LOGGER.debug("Rule-based: found %s items via labeled segments.", len(labeled_chunks))
        return _items_to_subtasks(labeled_chunks)

    # Last resort: single task, not parallel
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