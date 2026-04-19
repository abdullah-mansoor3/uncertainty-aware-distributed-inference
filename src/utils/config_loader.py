"""Configuration loading utilities for cluster_config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_KEYS = ["nodes", "model", "scheduler", "datasets", "evaluation"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration.

    Args:
        config_path: Path to YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")

    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing config key: {key}")

    return config
