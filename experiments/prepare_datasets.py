"""Dataset download and preprocessing pipeline for ANN+PDC project.

This script prepares NQ-Open, MMLU-Pro, and synthetic prompts into the canonical
JSONL format defined in copilot_context/instructions.txt Section 5.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset download and preprocessing."""
    parser = argparse.ArgumentParser(description="Download and prepare project datasets")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw dataset directory")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Processed dataset directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples per processed dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw datasets to JSONL and stop",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Skip downloads and only prepare processed outputs from raw JSONL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure logger format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to JSONL file."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def normalize_ground_truth(value: Any) -> List[str]:
    """Normalize answer payload into a non-empty list of strings."""
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned
    if value is None:
        return []

    text = str(value).strip()
    return [text] if text else []


def download_nq_open(raw_dir: Path) -> Path:
    """Download NQ-Open split and store as raw JSONL.

    Source: google-research-datasets/nq_open, config=nq_open, split=validation.
    """
    dataset = load_dataset("google-research-datasets/nq_open", "nq_open", split="validation")
    output_path = raw_dir / "nq_open_validation.jsonl"
    write_jsonl(output_path, dataset)
    LOGGER.info("Downloaded NQ-Open raw rows: %s", len(dataset))
    return output_path


def download_mmlu_pro(raw_dir: Path) -> Path:
    """Download MMLU-Pro split and store as raw JSONL.

    Source: TIGER-Lab/MMLU-Pro, config=default, split=test.
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default", split="test")
    output_path = raw_dir / "mmlu_pro_test.jsonl"
    write_jsonl(output_path, dataset)
    LOGGER.info("Downloaded MMLU-Pro raw rows: %s", len(dataset))
    return output_path


def sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    """Shuffle rows deterministically and return first sample_size elements."""
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    if not rows:
        return []

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[: min(sample_size, len(shuffled))]


def format_multiple_choice(options: Any) -> str:
    """Format options list into A/B/C-style prompt suffix."""
    if not isinstance(options, list) or not options:
        return ""

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: List[str] = []
    for idx, option in enumerate(options):
        label = labels[idx] if idx < len(labels) else f"O{idx + 1}"
        lines.append(f"{label}. {str(option).strip()}")
    return "\n".join(lines)


def resolve_mmlu_answer(row: Dict[str, Any]) -> List[str]:
    """Resolve MMLU-Pro answer into text ground truth list."""
    options = row.get("options")
    answer_index = row.get("answer_index")
    answer_value = row.get("answer")

    if isinstance(answer_index, int) and isinstance(options, list):
        if 0 <= answer_index < len(options):
            return normalize_ground_truth(options[answer_index])

    if isinstance(answer_value, str):
        text = answer_value.strip()
        if len(text) == 1 and text.isalpha() and isinstance(options, list):
            idx = ord(text.upper()) - ord("A")
            if 0 <= idx < len(options):
                return normalize_ground_truth(options[idx])
        return normalize_ground_truth(text)

    return normalize_ground_truth(answer_value)


def convert_nq_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    """Convert sampled NQ-Open rows to canonical project JSONL schema."""
    sampled = sample_rows(rows, sample_size=len(rows), seed=seed)
    output: List[Dict[str, Any]] = []

    for row in sampled:
        prompt = str(row.get("question", "")).strip()
        answers = normalize_ground_truth(row.get("answer"))
        if not prompt or not answers:
            continue
        output.append(
            {
                "id": len(output) + 1,
                "original_prompt": prompt,
                "ground_truth": answers,
            }
        )
        if len(output) >= sample_size:
            break
    return output


def convert_mmlu_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    """Convert sampled MMLU-Pro rows to canonical project JSONL schema."""
    sampled = sample_rows(rows, sample_size=len(rows), seed=seed)
    output: List[Dict[str, Any]] = []

    for row in sampled:
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        choices_text = format_multiple_choice(row.get("options"))
        prompt = f"{question}\nOptions:\n{choices_text}" if choices_text else question
        answers = resolve_mmlu_answer(row)
        if not answers:
            continue

        output.append(
            {
                "id": len(output) + 1,
                "original_prompt": prompt,
                "ground_truth": answers,
            }
        )
        if len(output) >= sample_size:
            break
    return output


def build_synthetic_rows(sample_size: int, seed: int) -> List[Dict[str, Any]]:
    """Create synthetic structured prompts in canonical schema.

    These are deterministic templates designed for decomposition experiments.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    cities = [
        "Lahore",
        "Karachi",
        "Islamabad",
        "Peshawar",
        "Quetta",
        "Faisalabad",
        "Rawalpindi",
        "Multan",
        "Hyderabad",
        "Sialkot",
    ]
    topics = [
        "energy",
        "traffic",
        "health",
        "education",
        "water",
        "waste management",
        "public transport",
        "disaster response",
    ]
    styles = ["bullet list", "short paragraph", "table-like text", "numbered steps"]

    candidates: List[Dict[str, Any]] = []
    for city in cities:
        for topic in topics:
            for style in styles:
                prompt = (
                    f"For {city}, provide three actions to improve {topic}, "
                    f"then add one risk and one mitigation. Use {style}."
                )
                truth = [
                    "Three actions, one risk, and one mitigation tied to the requested topic."
                ]
                candidates.append({"original_prompt": prompt, "ground_truth": truth})

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[: min(sample_size, len(candidates))]

    output: List[Dict[str, Any]] = []
    for row in selected:
        output.append(
            {
                "id": len(output) + 1,
                "original_prompt": row["original_prompt"],
                "ground_truth": row["ground_truth"],
            }
        )
    return output


def run_download(raw_dir: Path) -> None:
    """Download all external datasets into raw JSONL files."""
    ensure_dir(raw_dir)
    download_nq_open(raw_dir)
    download_mmlu_pro(raw_dir)


def run_prepare(raw_dir: Path, processed_dir: Path, sample_size: int, seed: int) -> None:
    """Load raw data, sample, and write canonical processed JSONL outputs."""
    ensure_dir(processed_dir)

    nq_raw = read_jsonl(raw_dir / "nq_open_validation.jsonl")
    mmlu_raw = read_jsonl(raw_dir / "mmlu_pro_test.jsonl")

    nq_rows = convert_nq_rows(nq_raw, sample_size=sample_size, seed=seed)
    mmlu_rows = convert_mmlu_rows(mmlu_raw, sample_size=sample_size, seed=seed + 1)
    synthetic_rows = build_synthetic_rows(sample_size=sample_size, seed=seed + 2)

    write_jsonl(processed_dir / "nq_open_100.jsonl", nq_rows)
    write_jsonl(processed_dir / "mmlu_pro_100.jsonl", mmlu_rows)
    write_jsonl(processed_dir / "synthetic_prompts.jsonl", synthetic_rows)

    LOGGER.info("Wrote processed NQ-Open rows: %s", len(nq_rows))
    LOGGER.info("Wrote processed MMLU-Pro rows: %s", len(mmlu_rows))
    LOGGER.info("Wrote processed synthetic rows: %s", len(synthetic_rows))


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    configure_logging(args.verbose)

    if args.download_only and args.prepare_only:
        raise ValueError("Choose either --download-only or --prepare-only, not both")

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    if not args.prepare_only:
        run_download(raw_dir)

    if not args.download_only:
        run_prepare(raw_dir, processed_dir, sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
