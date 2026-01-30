from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import yaml


class IntegrationError(RuntimeError):
    """Raised when integration setup fails."""


def add_to_syspath(repo_path: Path) -> None:
    if not repo_path.exists():
        raise IntegrationError(f"Repo path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise IntegrationError(f"Repo path is not a directory: {repo_path}")
    candidate_paths = [repo_path, repo_path.parent]
    for path in candidate_paths:
        path_str = str(path)
        if path_str not in os.sys.path:
            os.sys.path.insert(0, path_str)


def load_entrypoint(entry: str) -> Callable[..., Any]:
    """Load `module:callable` entrypoint string."""
    if ":" not in entry:
        raise IntegrationError(
            f"Invalid entrypoint '{entry}'. Expected format 'module:callable'."
        )
    module_name, attr = entry.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - explicit error reporting
        raise IntegrationError(
            f"Failed to import module '{module_name}' for entry '{entry}'."
        ) from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise IntegrationError(
            f"Module '{module_name}' does not define '{attr}' for entry '{entry}'."
        ) from exc
    if not callable(target):
        raise IntegrationError(f"Entry '{entry}' is not callable.")
    return target


def load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise IntegrationError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(config_path.read_text()) or {}
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text())
    raise IntegrationError(
        f"Unsupported config format for {config_path}. Use .json or .yaml/.yml."
    )


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def dump_config(config: Any, path: Path) -> None:
    payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
