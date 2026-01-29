"""Training loop runner for CLAM + CloseAirCombat."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from clam_aircombat.algorithms.clam_adapter import CLAMSpec, Trainable
from clam_aircombat.envs.close_air_combat import EnvSpec


class TrainingError(RuntimeError):
    """Raised when training loop cannot proceed."""


@dataclass
class TrainingSpec:
    output_dir: Path
    training_config: Dict[str, Any]
    env_spec: EnvSpec
    clam_spec: CLAMSpec

    def run(self) -> Any:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env = self.env_spec.create()
        algorithm = self.clam_spec.build()
        if not hasattr(algorithm, "train"):
            raise TrainingError("Loaded CLAM algorithm does not implement 'train' method")

        # Forward extra training config into train() so CLAM-RL can control the loop.
        return algorithm.train(env=env, output_dir=str(self.output_dir), **self.training_config)


def build_training_spec(
    output_dir: str | Path,
    env_factory: str,
    env_config: Dict[str, Any],
    clam_class: str,
    clam_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> TrainingSpec:
    return TrainingSpec(
        output_dir=Path(output_dir).expanduser().resolve(),
        training_config=training_config,
        env_spec=EnvSpec(factory=env_factory, config=env_config),
        clam_spec=CLAMSpec(class_path=clam_class, config=clam_config),
    )
