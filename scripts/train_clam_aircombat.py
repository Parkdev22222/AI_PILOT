"""Entry point for CLAM training on CloseAirCombat 2v2 environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from clam_aircombat import load_config, resolve_repo_path
from clam_aircombat.training.runner import build_training_spec


def _parse_overrides(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for overrides: {exc}")


def _extend_sys_path(*paths: Path) -> None:
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeaircombat-path", required=True, help="Path to CloseAirCombat repo")
    parser.add_argument("--clam-rl-path", required=True, help="Path to CLAM-RL repo")
    parser.add_argument("--env-factory", required=True, help="Environment factory 'module:function'")
    parser.add_argument("--clam-class", required=True, help="CLAM class 'module:ClassName'")
    parser.add_argument("--env-config", required=True, help="YAML/JSON config for environment")
    parser.add_argument("--output-dir", required=True, help="Output directory for logs/checkpoints")
    parser.add_argument(
        "--env-overrides",
        default=None,
        help="JSON overrides for env config (merged after config load)",
    )
    parser.add_argument(
        "--clam-overrides",
        default=None,
        help="JSON overrides for algorithm config (merged after config load)",
    )
    parser.add_argument(
        "--training-overrides",
        default=None,
        help="JSON overrides for training config (merged after config load)",
    )
    return parser.parse_args()


def merge_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return base
    merged = dict(base)
    merged.update(overrides)
    return merged


def main() -> None:
    args = parse_args()
    closeaircombat_path = resolve_repo_path(args.closeaircombat_path, "CloseAirCombat")
    clam_rl_path = resolve_repo_path(args.clam_rl_path, "CLAM-RL")
    _extend_sys_path(closeaircombat_path, clam_rl_path)

    config = load_config(args.env_config)
    env_config = merge_overrides(config.env, _parse_overrides(args.env_overrides))
    clam_config = merge_overrides(config.algorithm, _parse_overrides(args.clam_overrides))
    training_config = merge_overrides(config.training, _parse_overrides(args.training_overrides))

    training_spec = build_training_spec(
        output_dir=args.output_dir,
        env_factory=args.env_factory,
        env_config=env_config,
        clam_class=args.clam_class,
        clam_config=clam_config,
        training_config=training_config,
    )

    training_spec.run()


if __name__ == "__main__":
    main()
