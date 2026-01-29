"""Adapter for CLAM-RL algorithms.

This adapter dynamically loads a CLAM algorithm class from the CLAM-RL repository
so we can keep the integration flexible.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Protocol


class CLAMLoadError(RuntimeError):
    """Raised when CLAM algorithm cannot be loaded."""


class Trainable(Protocol):
    def train(self, env: Any, **kwargs: Any) -> Any:
        ...


@dataclass
class CLAMSpec:
    class_path: str
    config: Dict[str, Any]

    def build(self) -> Trainable:
        cls = load_class(self.class_path)
        try:
            return cls(**self.config)
        except TypeError as exc:
            raise CLAMLoadError(
                f"Failed to construct CLAM class '{self.class_path}'. "
                "Check init signature and config keys."
            ) from exc


def load_class(class_path: str) -> type:
    if ":" not in class_path:
        raise CLAMLoadError(
            "clam-class must be in the form 'module:ClassName', "
            f"got '{class_path}'"
        )
    module_name, class_name = class_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise CLAMLoadError(
            f"Could not import CLAM module '{module_name}'. Ensure CLAM-RL is on PYTHONPATH."
        ) from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise CLAMLoadError(
            f"Module '{module_name}' does not provide class '{class_name}'."
        ) from exc
    if not isinstance(cls, type):
        raise CLAMLoadError(f"'{class_path}' is not a class")
    return cls
