"""Dataset downloader and preprocessor for project JSONL format.

Builds 100-sample processed datasets for NQ-Open and MMLU-Pro plus synthetic
prompts under data/processed/ for fair cross-pipeline comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Download and preprocess datasets")
    parser.add_argument("--samples", type=int, default=100, help="Samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write records in JSONL format.

    Args:
        path: Output path.
        rows: Record list.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_nq_open(sample_count: int, seed: int) -> List[Dict[str, object]]:
    """Create NQ-Open processed samples.

    Args:
        sample_count: Number of samples.
        seed: Random seed.

    Returns:
        Project-format rows.
    """
    dataset = load_dataset("nq_open", split="validation")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = indices[:sample_count]

    rows: List[Dict[str, object]] = []
    for row_id, index in enumerate(selected, start=1):
        item = dataset[index]
        rows.append(
            {
                "id": row_id,
                "original_prompt": item["question"],
                "ground_truth": item["answer"] if isinstance(item["answer"], list) else [str(item["answer"])],
            }
        )
    return rows


def build_mmlu_pro(sample_count: int, seed: int) -> List[Dict[str, object]]:
    """Create MMLU-Pro processed samples.

    Args:
        sample_count: Number of samples.
        seed: Random seed.

    Returns:
        Project-format rows.
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = indices[:sample_count]

    rows: List[Dict[str, object]] = []
    for row_id, index in enumerate(selected, start=1):
        item = dataset[index]
        prompt = f"{item['question']}\nChoices: {item['options']}"
        rows.append(
            {
                "id": row_id,
                "original_prompt": prompt,
                "ground_truth": [str(item.get("answer", ""))],
            }
        )
    return rows


def build_synthetic() -> List[Dict[str, object]]:
    """Create synthetic decomposition-focused prompts.

    Returns:
        Project-format synthetic rows.
    """
    prompts = [
        {
            "id": 1,
            "original_prompt": "Summarize Pakistan's GDP trend and additionally list two key export sectors.",
            "ground_truth": ["GDP trend summary and major exports include textiles and rice."],
        },
        {
            "id": 2,
            "original_prompt": "Explain what entropy means in uncertainty estimation and also provide one practical routing example.",
            "ground_truth": ["Entropy measures uncertainty; high-entropy tasks should route to stronger nodes."],
        },
        {
            "id": 3,
            "original_prompt": "Define heterogeneous scheduling and additionally compare round-robin vs uncertainty-aware routing.",
            "ground_truth": ["Heterogeneous scheduling matches tasks to node capabilities; uncertainty-aware routing is adaptive."],
        },
    ]
    return prompts


def main() -> None:
    """Download and preprocess all required datasets.

    Returns:
        None.
    """
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    nq_rows = build_nq_open(sample_count=args.samples, seed=args.seed)
    write_jsonl(PROCESSED_DIR / "nq_open_100.jsonl", nq_rows)

    mmlu_rows = build_mmlu_pro(sample_count=args.samples, seed=args.seed)
    write_jsonl(PROCESSED_DIR / "mmlu_pro_100.jsonl", mmlu_rows)

    synthetic_rows = build_synthetic()
    write_jsonl(PROCESSED_DIR / "synthetic_prompts.jsonl", synthetic_rows)

    print("Processed datasets written to data/processed/")


if __name__ == "__main__":
    main()
