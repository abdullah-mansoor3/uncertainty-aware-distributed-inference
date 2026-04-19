"""Run naive parallel inference experiment pipeline."""

from __future__ import annotations

import argparse
import logging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for naive experiment."""
    parser = argparse.ArgumentParser(description="Run naive parallel pipeline")
    parser.add_argument("--config", required=True, help="Path to cluster config")
    parser.add_argument("--dataset", required=True, help="Path to input jsonl")
    parser.add_argument("--output", required=True, help="Path to output jsonl")
    return parser.parse_args()


def main() -> None:
    """Entry point for naive experiment execution."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("Naive scaffold ready")
    logging.info("config=%s dataset=%s output=%s", args.config, args.dataset, args.output)


if __name__ == "__main__":
    main()
