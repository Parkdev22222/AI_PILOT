from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .clam_adapter import (
    instantiate_clam,
    load_clam_entry,
    resolve_clam_rl_path,
    run_training,
)
from .config import TrainingConfig
from .env_adapter import make_env, resolve_close_air_combat_path
from .utils import IntegrationError, dump_config, load_config_file, merge_config

LOGGER = logging.getLogger("clam_closeaircombat")


def build_config(args: argparse.Namespace) -> TrainingConfig:
    file_config = load_config_file(args.config)
    cli_config = {
        "env_entry": args.env_entry,
        "clam_entry": args.clam_entry,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "log_dir": args.log_dir,
        "close_air_combat_path": args.close_air_combat_path,
        "clam_rl_path": args.clam_rl_path,
        "env_kwargs": args.env_kwargs or {},
        "clam_kwargs": args.clam_kwargs or {},
    }
    merged = merge_config(file_config, {k: v for k, v in cli_config.items() if v is not None})
    env_kwargs = merged.get("env_kwargs", {})
    clam_kwargs = merged.get("clam_kwargs", {})
    extra = {
        key: value
        for key, value in merged.items()
        if key not in {
            "env_entry",
            "clam_entry",
            "total_steps",
            "seed",
            "log_dir",
            "close_air_combat_path",
            "clam_rl_path",
            "env_kwargs",
            "clam_kwargs",
        }
    }
    return TrainingConfig(
        env_entry=merged["env_entry"],
        clam_entry=merged["clam_entry"],
        total_steps=int(merged.get("total_steps", 500_000)),
        seed=int(merged.get("seed", 42)),
        log_dir=merged.get("log_dir", "runs/clam_closeaircombat"),
        close_air_combat_path=merged.get("close_air_combat_path"),
        clam_rl_path=merged.get("clam_rl_path"),
        env_kwargs=env_kwargs,
        clam_kwargs=clam_kwargs,
        extra=extra,
    )


def parse_kv(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        import json

        return json.loads(text)
    except Exception as exc:
        raise IntegrationError(
            "--env-kwargs/--clam-kwargs must be JSON string."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLAM x CloseAirCombat 2v2 trainer")
    parser.add_argument("--env-entry", required=True, help="module:callable for env")
    parser.add_argument("--clam-entry", required=True, help="module:callable for CLAM trainer")
    parser.add_argument("--config", help="Path to JSON/YAML config file")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="runs/clam_closeaircombat")
    parser.add_argument("--close-air-combat-path")
    parser.add_argument("--clam-rl-path")
    parser.add_argument("--env-kwargs", type=parse_kv, default=None)
    parser.add_argument("--clam-kwargs", type=parse_kv, default=None)
    return parser


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "train.log"),
            logging.StreamHandler(),
        ],
    )


def run(args: argparse.Namespace) -> None:
    config = build_config(args)
    setup_logging(config.log_dir)
    LOGGER.info("Resolved config: %s", config)
    dump_config(config, Path(config.log_dir) / "resolved_config.yaml")

    close_air_path = resolve_close_air_combat_path(config.close_air_combat_path)
    clam_rl_path = resolve_clam_rl_path(config.clam_rl_path)
    LOGGER.info("CloseAirCombat path: %s", close_air_path)
    LOGGER.info("CLAM-RL path: %s", clam_rl_path)

    env = make_env(config.env_entry, close_air_path, config.env_kwargs)
    clam_entry = load_clam_entry(config.clam_entry, clam_rl_path)
    trainer = instantiate_clam(clam_entry, config.clam_kwargs)

    if hasattr(trainer, "seed"):
        trainer.seed(config.seed)
    if hasattr(env, "seed"):
        env.seed(config.seed)

    run_training(trainer, env, config.total_steps)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
