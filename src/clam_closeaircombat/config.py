from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    env_entry: str
    clam_entry: str
    total_steps: int = 500_000
    seed: int = 42
    log_dir: str = "runs/clam_closeaircombat"
    close_air_combat_path: str | None = None
    clam_rl_path: str | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    clam_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
