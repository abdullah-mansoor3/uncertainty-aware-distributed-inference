"""Download and sample ParallelPrompt dataset.

Keeps 100 rows stratified by task_type distribution.
Outputs canonical JSONL for the project pipeline.

Usage:
    python data/prepare_parallelprompt.py --verbose
    python data/prepare_parallelprompt.py --from-local data/raw/parallelprompt_full.parquet --verbose
    python data/prepare_parallelprompt.py --sample-size 50 --output data/processed/parallelprompt_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

HF_PATH = "hf://datasets/forgelab/ParallelPrompt/data/train-00000-of-00001.parquet"
HF_DATASET_ID = "forgelab/ParallelPrompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ParallelPrompt subset")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/parallelprompt_100.jsonl")
    parser.add_argument(
        "--raw-output",
        default="data/raw/parallelprompt_full.parquet",
        help="Cache raw parquet locally so you don't re-download next time",
    )
    parser.add_argument(
        "--from-local",
        default=None,
        help="Skip download entirely, load from this local parquet path",
    )
    parser.add_argument(
        "--use-datasets-lib",
        action="store_true",
        help="Use HuggingFace datasets library instead of direct parquet read",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_from_local(path: str) -> pd.DataFrame:
    """Load from a local parquet file."""
    LOGGER.info("Loading from local path: %s", path)
    return pd.read_parquet(path)


def load_via_parquet(hf_path: str, local_cache: str) -> pd.DataFrame:
    """Download via direct parquet read from HuggingFace. Can be slow (~40MB stream)."""
    cache_path = Path(local_cache)
    if cache_path.exists():
        LOGGER.info("Local cache found, loading: %s", cache_path)
        return pd.read_parquet(cache_path)

    LOGGER.info("Downloading via parquet stream from HuggingFace (may take a while)...")
    df = pd.read_parquet(hf_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    LOGGER.info("Cached to %s (%s rows)", cache_path, len(df))
    return df


def load_via_datasets_lib(local_cache: str) -> pd.DataFrame:
    """Download using HuggingFace datasets library. More reliable than direct parquet stream."""
    cache_path = Path(local_cache)
    if cache_path.exists():
        LOGGER.info("Local cache found, loading: %s", cache_path)
        return pd.read_parquet(cache_path)

    LOGGER.info("Downloading via datasets library...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    ds = load_dataset(HF_DATASET_ID, split="train")
    df = ds.to_pandas()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    LOGGER.info("Cached to %s (%s rows)", cache_path, len(df))
    return df


def load_raw(args: argparse.Namespace) -> pd.DataFrame:
    """Route to correct loader based on args."""
    if args.from_local:
        return load_from_local(args.from_local)
    if args.use_datasets_lib:
        return load_via_datasets_lib(args.raw_output)
    return load_via_parquet(HF_PATH, args.raw_output)


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Log actual columns and rename to canonical names based on column name patterns."""
    LOGGER.info("Raw columns: %s", df.columns.tolist())
    LOGGER.info("Shape: %s", df.shape)
    LOGGER.debug("First row:\n%s", df.iloc[0].to_string())

    rename_map: dict[str, str] = {}
    already_assigned: set[str] = set()

    def assign(col: str, canonical: str) -> None:
        if canonical not in already_assigned:
            rename_map[col] = canonical
            already_assigned.add(canonical)

    for col in df.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ("prompt", "instruction", "query", "text")) and "prompt" not in already_assigned:
            assign(col, "prompt")
        elif "task" in col_lower and "type" in col_lower:
            assign(col, "task_type")
        elif "parallel" in col_lower:
            assign(col, "is_parallel")
        elif "confidence" in col_lower:
            assign(col, "confidence")
        elif any(k in col_lower for k in ("iteration", "data_list", "subtask")):
            assign(col, "iterations")
        elif "context" in col_lower:
            assign(col, "context")
        elif "source" in col_lower:
            assign(col, "source")

    LOGGER.info("Column rename map: %s", rename_map)
    df = df.rename(columns=rename_map)
    return df


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample preserving task_type distribution. Filters to parallel-only rows first."""

    # Filter to rows confirmed as parallel if column exists
    if "is_parallel" in df.columns:
        parallel_df = df[df["is_parallel"] == True].copy()
        LOGGER.info("Parallel rows: %s / %s total", len(parallel_df), len(df))
        if len(parallel_df) >= sample_size:
            df = parallel_df
        else:
            LOGGER.warning(
                "Only %s parallel rows found, need %s — using all rows.",
                len(parallel_df), sample_size,
            )

    if "task_type" not in df.columns or df["task_type"].isna().all():
        LOGGER.info("No task_type column found — using plain random sample.")
        return df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

    counts = df["task_type"].value_counts()
    LOGGER.info("Task type distribution in source:\n%s", counts.to_string())

    total = len(df)
    allocations: dict[str, int] = {}
    for task_type, count in counts.items():
        proportion = count / total
        allocations[task_type] = max(1, round(proportion * sample_size))

    # Fix rounding drift — adjust largest group
    drift = sum(allocations.values()) - sample_size
    if drift != 0:
        biggest = counts.index[0]
        allocations[biggest] -= drift

    LOGGER.info("Planned allocations per task_type: %s", allocations)

    rng = random.Random(seed)
    parts = []
    for task_type, n in allocations.items():
        group = df[df["task_type"] == task_type]
        n = min(n, len(group))
        if n <= 0:
            continue
        parts.append(group.sample(n=n, random_state=rng.randint(0, 99999)))

    result = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    LOGGER.info("Sampled shape: %s", result.shape)
    LOGGER.info("Final task_type distribution:\n%s", result["task_type"].value_counts().to_string())
    return result


def row_to_canonical(row: pd.Series, new_id: int) -> dict:
    """Convert one ParallelPrompt row to the project canonical JSONL schema."""
    prompt = str(row.get("prompt", "")).strip()

    # iterations = decomposition ground truth (expected independent subtasks).
    iterations = row.get("iterations")
    if isinstance(iterations, list):
        decomposition_ground_truth = [str(item).strip() for item in iterations if str(item).strip()]
    elif iterations is not None:
        decomposition_ground_truth = [str(iterations).strip()]
    else:
        decomposition_ground_truth = []

    # context is treated as answer/reference ground truth when present.
    context_value = row.get("context")
    if isinstance(context_value, str) and context_value.strip():
        answer_ground_truth = [context_value.strip()]
    else:
        answer_ground_truth = []

    return {
        "id": new_id,
        "original_prompt": prompt,
        # Canonical answer references used for ROUGE/BLEU/METEOR/BERT.
        "ground_truth": answer_ground_truth,
        # Canonical decomposition references used to score decomposition quality.
        "decomposition_ground_truth": decomposition_ground_truth,
        # meta is kept for analysis scripts but not used by pipeline runners
        "meta": {
            "source": str(row.get("source", "")),
            "task_type": str(row.get("task_type", "")),
            "is_parallel": bool(row.get("is_parallel", False)),
            "confidence": str(row.get("confidence", "")),
            "context": str(context_value) if context_value else None,
        },
    }

def detect_language(text: str) -> str:
    """Detect language of text. Returns ISO 639-1 code or 'unknown' on failure.

    Args:
        text: Input text to detect language for.

    Returns:
        Language code string e.g. 'en', 'fr', 'unknown'.
    """
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return "unknown"


NON_LATIN_SCRIPT_RE = re.compile(
    r"["
    r"\u0400-\u04FF"  # Cyrillic
    r"\u0370-\u03FF"  # Greek
    r"\u0590-\u05FF"  # Hebrew
    r"\u0600-\u06FF"  # Arabic
    r"\u0900-\u097F"  # Devanagari
    r"\u0E00-\u0E7F"  # Thai
    r"\u3040-\u309F"  # Hiragana
    r"\u30A0-\u30FF"  # Katakana
    r"\u31F0-\u31FF"  # Katakana Phonetic Extensions
    r"\u3400-\u4DBF"  # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs
    r"\uAC00-\uD7AF"  # Hangul Syllables
    r"]"
)


def filter_english_only(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where prompt is not detected as English.

    Args:
        df: DataFrame with resolved columns including 'prompt'.

    Returns:
        Filtered DataFrame with only English prompts.
    """
    if "prompt" not in df.columns:
        LOGGER.warning("No prompt column found, skipping language filter.")
        return df

    LOGGER.info("Removing prompts with non-Latin scripts...")

    def has_non_latin_script(text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return True
        return NON_LATIN_SCRIPT_RE.search(text) is not None

    non_latin_mask = df["prompt"].apply(has_non_latin_script)
    latin_df = df[~non_latin_mask].copy()

    LOGGER.info(
        "Script filter: kept %s / %s rows (dropped %s non-Latin script)",
        len(latin_df), len(df), len(df) - len(latin_df),
    )

    LOGGER.info("Running language detection on %s rows...", len(latin_df))

    def is_english(text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        lang = detect_language(text[:300])  # first 300 chars is enough for detection
        return lang == "en"

    mask = latin_df["prompt"].apply(is_english)
    english_df = latin_df[mask].copy()

    LOGGER.info(
        "Language filter: kept %s / %s rows (dropped %s non-English)",
        len(english_df), len(latin_df), len(latin_df) - len(english_df),
    )
    return english_df


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    df = load_raw(args)
    df = resolve_columns(df)
    df = filter_english_only(df)
    sampled = stratified_sample(df, sample_size=args.sample_size, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in sampled.iterrows():
        record = row_to_canonical(row, new_id=len(records) + 1)
        if not record["original_prompt"]:
            LOGGER.warning("Skipping row — empty prompt")
            continue
        records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Wrote %s records to %s", len(records), output_path)

    LOGGER.info("--- Sample output records ---")
    for record in records[:3]:
        LOGGER.info(
            "id=%s | task=%s | decomp_gt=%s | answer_gt=%s | prompt=%s...",
            record["id"],
            record["meta"]["task_type"],
            len(record["decomposition_ground_truth"]),
            len(record["ground_truth"]),
            record["original_prompt"][:80],
        )


if __name__ == "__main__":
    main()