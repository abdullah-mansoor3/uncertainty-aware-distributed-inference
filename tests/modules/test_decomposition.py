"""tests/modules/test_decomposition.py

Loads the actual Llama model and runs decompose_prompt on the first 2 rows
read directly from data/processed/parallelprompt_100.jsonl.

Run:
    python -m pytest tests/modules/test_decomposition.py -v -s
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from llama_cpp import Llama
from src.modules.decomposition import decompose_prompt

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

with open("configs/cluster_config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_CFG = CFG["model"]
DATASET_PATH = CFG["datasets"]["parallelprompt"]

# ---------------------------------------------------------------------------
# Load first 2 rows from dataset
# ---------------------------------------------------------------------------

def load_rows(path: str, n: int = 2):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows

ROWS = load_rows(DATASET_PATH, n=2)

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------

print("\nLoading model...")
LLM = Llama(
    model_path=MODEL_CFG["path"],
    n_ctx=MODEL_CFG.get("n_ctx", 1024),
    n_threads=CFG["nodes"]["master"]["n_threads"],
    verbose=False,
)
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def _print_result(row, subtasks):
    print(f"\n{'='*60}")
    print(f"  ROW id={row['id']} | source={row['meta']['source']}")
    print(f"  PROMPT (first 120 chars):")
    print(f"  {row['original_prompt'][:120]}")
    print(f"  --> {len(subtasks)} subtask(s)")
    print(f"{'='*60}")
    for s in subtasks:
        tag = "PARALLEL" if s["parallel_safe"] else "SERIAL"
        print(f"  [{s['id']}] [{tag}]")
        print(f"      {s['text']}")
    print()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMDecompositionOnDataset(unittest.TestCase):

    def test_row1(self):
        row = ROWS[0]
        result = decompose_prompt(row["original_prompt"], llm=LLM)
        _print_result(row, result)
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    def test_row2(self):
        row = ROWS[1]
        result = decompose_prompt(row["original_prompt"], llm=LLM)
        _print_result(row, result)
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)