from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .utils import IntegrationError, add_to_syspath, load_entrypoint


def resolve_clam_rl_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    import os

    value = os.environ.get("CLAM_RL_PATH")
    if value:
        return Path(value).expanduser().resolve()
    return Path("external/CLAM-RL").resolve()


def load_clam_entry(entry: str, repo_path: Path) -> Callable[..., Any]:
    add_to_syspath(repo_path)
    return load_entrypoint(entry)


def instantiate_clam(entry: Callable[..., Any], clam_kwargs: dict[str, Any]) -> Any:
    try:
        return entry(**clam_kwargs)
    except TypeError as exc:
        raise IntegrationError(
            f"Failed to instantiate CLAM entry with kwargs: {clam_kwargs}."
        ) from exc
    except Exception as exc:
        raise IntegrationError("CLAM instantiation failed.") from exc


def run_training(trainer: Any, env: Any, total_steps: int) -> None:
    """Attempt to run training with common method names."""
    if hasattr(trainer, "train"):
        trainer.train(env=env, total_steps=total_steps)
        return
    if hasattr(trainer, "learn"):
        trainer.learn(env=env, total_steps=total_steps)
        return
    if hasattr(trainer, "fit"):
        trainer.fit(env=env, total_steps=total_steps)
        return
    raise IntegrationError(
        "Trainer does not expose train/learn/fit. Please adapt run_training."
    )
