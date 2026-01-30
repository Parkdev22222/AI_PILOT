from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import IntegrationError, add_to_syspath, load_entrypoint


def resolve_close_air_combat_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    import os

    env_var = "CLOSE_AIR_COMBAT_PATH"
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser().resolve()
    return Path("external/CloseAirCombat").resolve()


def make_env(entry: str, repo_path: Path, env_kwargs: dict[str, Any]) -> Any:
    add_to_syspath(repo_path)
    factory = load_entrypoint(entry)
    try:
        return factory(**env_kwargs)
    except TypeError as exc:
        raise IntegrationError(
            f"Failed to create env via '{entry}'. Check env_kwargs: {env_kwargs}."
        ) from exc
    except Exception as exc:
        raise IntegrationError(
            f"Env creation failed via '{entry}': {exc}"
        ) from exc
