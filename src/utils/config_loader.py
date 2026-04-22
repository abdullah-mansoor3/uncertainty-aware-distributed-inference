<<<<<<< HEAD
"""Configuration loading utilities for cluster_config.yaml."""

from __future__ import annotations

=======
"""Configuration loading utilities.

Loads project configuration from YAML and validates required keys used by the
serial experiment runner and core modules.
"""

from __future__ import annotations

import hashlib
import json
>>>>>>> 2c641dd (feat: Full project scaffold)
from pathlib import Path
from typing import Any, Dict

import yaml


<<<<<<< HEAD
REQUIRED_KEYS = ["nodes", "model", "scheduler", "datasets", "evaluation"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration.

    Args:
        config_path: Path to YAML file.
=======
REQUIRED_TOP_LEVEL_KEYS = ["nodes", "model", "scheduler", "datasets", "evaluation", "runtime"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML.

    Args:
        config_path: Path to YAML config file.
>>>>>>> 2c641dd (feat: Full project scaffold)

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
<<<<<<< HEAD
        raise ValueError("Config root must be a mapping")

    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing config key: {key}")

    return config
=======
        raise ValueError("Config must parse to a dictionary")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise KeyError(f"Missing config key '{key}' in {config_path}")
    return config


def config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of configuration content.

    Args:
        config: Configuration dictionary.

    Returns:
        SHA256 hash hex digest for reproducibility tracking.
    """
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
>>>>>>> 2c641dd (feat: Full project scaffold)
