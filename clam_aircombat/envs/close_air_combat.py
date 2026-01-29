"""CloseAirCombat environment adapter.

This module provides a flexible factory loader. You can specify the environment
factory as a string: "module:function".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict


class EnvFactoryError(RuntimeError):
    """Raised when environment factory cannot be loaded or executed."""


@dataclass
class EnvSpec:
    factory: str
    config: Dict[str, Any]

    def create(self) -> Any:
        factory_fn = load_factory(self.factory)
        try:
            return factory_fn(**self.config)
        except TypeError as exc:
            raise EnvFactoryError(
                f"Failed to create environment with factory '{self.factory}'. "
                "Check factory signature and config keys."
            ) from exc


def load_factory(factory_path: str) -> Callable[..., Any]:
    if ":" not in factory_path:
        raise EnvFactoryError(
            "env-factory must be in the form 'module:function', "
            f"got '{factory_path}'"
        )
    module_name, attr_name = factory_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise EnvFactoryError(
            f"Could not import env module '{module_name}'. Ensure CloseAirCombat is on PYTHONPATH."
        ) from exc
    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise EnvFactoryError(
            f"Module '{module_name}' does not provide '{attr_name}'."
        ) from exc
    if not callable(factory):
        raise EnvFactoryError(f"Environment factory '{factory_path}' is not callable")
    return factory
