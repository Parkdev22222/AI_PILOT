"""Configuration helpers for CLAM + CloseAirCombat."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    env: Dict[str, Any]
    algorithm: Dict[str, Any]
    training: Dict[str, Any]


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


def load_config(path: str | Path) -> Config:
    """Load YAML or JSON config into Config.

    Raises:
        ConfigError: if the config is missing required sections.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    if config_path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(config_path.read_text())
    elif config_path.suffix == ".json":
        data = json.loads(config_path.read_text())
    else:
        raise ConfigError("Config must be YAML (.yaml/.yml) or JSON (.json)")

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping/object")

    env = data.get("env")
    algorithm = data.get("algorithm")
    training = data.get("training")

    sections = (("env", env), ("algorithm", algorithm), ("training", training))
    missing = [key for key, value in sections if value is None]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    if not isinstance(env, dict) or not isinstance(algorithm, dict) or not isinstance(training, dict):
        raise ConfigError("'env', 'algorithm', and 'training' must be mappings")

    return Config(env=env, algorithm=algorithm, training=training)
