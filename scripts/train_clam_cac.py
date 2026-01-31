#!/usr/bin/env python3
"""Train CLAM on the CloseAirCombat 2v2 environment.

This script intentionally keeps the integration points flexible so you can
plug in the actual CloseAirCombat environment entrypoint and CLAM-RL trainer
entrypoint without modifying source code. Use --env-entry and --clam-entry
(or set them in the config file) to point at the real callables.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class TrainConfig:
    env_entry: Optional[str]
    env_kwargs: Dict[str, Any]
    clam_entry: Optional[str]
    clam_kwargs: Dict[str, Any]
    total_steps: Optional[int]
    log_dir: Optional[str]
    seed: Optional[int]
    device: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLAM on CloseAirCombat 2v2 with flexible entrypoints."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file with env/clam settings.",
    )
    parser.add_argument(
        "--cac-root",
        type=str,
        default=None,
        help="Path to the CloseAirCombat repo (added to PYTHONPATH).",
    )
    parser.add_argument(
        "--clam-root",
        type=str,
        default=None,
        help="Path to the CLAM-RL repo (added to PYTHONPATH).",
    )
    parser.add_argument(
        "--env-entry",
        type=str,
        default=None,
        help="Environment entrypoint in 'module:function' form.",
    )
    parser.add_argument(
        "--clam-entry",
        type=str,
        default=None,
        help="CLAM trainer entrypoint in 'module:callable' form.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create env + trainer and exit without training.",
    )
    return parser.parse_args()


def add_repo_to_syspath(path: Optional[str], label: str) -> None:
    if not path:
        return
    repo_path = Path(path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {repo_path}")
    sys.path.insert(0, str(repo_path))
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{repo_path}{os.pathsep}{existing}" if existing else str(repo_path)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not installed. Install it or provide a JSON config file."
            )
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(
        f"Unsupported config extension {config_path.suffix!r}. Use .yaml/.yml or .json."
    )


def load_entrypoint(entry: str) -> Callable[..., Any]:
    if ":" not in entry:
        raise ValueError(
            f"Entrypoint {entry!r} must be in 'module:callable' format."
        )
    module_name, attr_name = entry.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' does not define '{attr_name}'."
        ) from exc


def call_with_supported_args(func: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return func(**supported)


def build_train_config(config: Dict[str, Any], args: argparse.Namespace) -> TrainConfig:
    env_config = config.get("env", {})
    clam_config = config.get("clam", {})
    train_config = config.get("train", {})

    return TrainConfig(
        env_entry=args.env_entry or env_config.get("entrypoint"),
        env_kwargs=env_config.get("kwargs", {}) or {},
        clam_entry=args.clam_entry or clam_config.get("entrypoint"),
        clam_kwargs=clam_config.get("kwargs", {}) or {},
        total_steps=train_config.get("total_steps"),
        log_dir=args.log_dir or train_config.get("log_dir"),
        seed=args.seed or train_config.get("seed"),
        device=args.device or train_config.get("device"),
    )


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def validate_env(env: Any) -> None:
    missing = [name for name in ("reset", "step") if not hasattr(env, name)]
    if missing:
        raise TypeError(
            f"Environment is missing required methods: {', '.join(missing)}"
        )


def instantiate_trainer(entry: Callable[..., Any], config: TrainConfig, env: Any) -> Any:
    if inspect.isclass(entry):
        return call_with_supported_args(
            entry,
            env=env,
            config=config,
            **config.clam_kwargs,
        )
    return call_with_supported_args(
        entry,
        env=env,
        config=config,
        **config.clam_kwargs,
    )


def run_training(trainer: Any, config: TrainConfig, env: Any) -> None:
    for method_name in ("train", "learn", "run"):
        if hasattr(trainer, method_name):
            method = getattr(trainer, method_name)
            call_with_supported_args(
                method,
                env=env,
                total_steps=config.total_steps,
                log_dir=config.log_dir,
                device=config.device,
                seed=config.seed,
            )
            return
    raise AttributeError(
        "Trainer does not expose train/learn/run; provide a callable entrypoint instead."
    )


def main() -> None:
    args = parse_args()
    add_repo_to_syspath(args.cac_root, "CloseAirCombat")
    add_repo_to_syspath(args.clam_root, "CLAM-RL")

    raw_config = load_config(args.config)
    config = build_train_config(raw_config, args)

    if not config.env_entry:
        raise ValueError(
            "Environment entrypoint is required. Set --env-entry or env.entrypoint in config."
        )
    if not config.clam_entry:
        raise ValueError(
            "CLAM entrypoint is required. Set --clam-entry or clam.entrypoint in config."
        )

    set_seed(config.seed)

    env_factory = load_entrypoint(config.env_entry)
    env = call_with_supported_args(env_factory, **config.env_kwargs)
    validate_env(env)

    trainer_entry = load_entrypoint(config.clam_entry)
    trainer = instantiate_trainer(trainer_entry, config, env)

    if args.dry_run:
        print("Dry run completed: environment and trainer instantiated.")
    else:
        run_training(trainer, config, env)

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
