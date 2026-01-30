from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class FlatActionSpec:
    nvec: tuple[int, ...]

    @property
    def size(self) -> int:
        total = 1
        for value in self.nvec:
            total *= int(value)
        return total

    def flatten(self, action: Sequence[int]) -> int:
        index = 0
        for value, base in zip(action, self.nvec):
            index = index * base + int(value)
        return index

    def unflatten(self, index: int) -> np.ndarray:
        values = []
        remaining = int(index)
        for base in reversed(self.nvec):
            values.append(remaining % base)
            remaining //= base
        return np.array(list(reversed(values)), dtype=np.int32)


class CloseAirCombatAdapter:
    """Adapter to expose CloseAirCombat envs with MPE-style interfaces."""

    def __init__(self, env: Any, flatten_actions: bool = True):
        self.env = env
        self.flatten_actions = flatten_actions
        if not hasattr(env, "num_agents"):
            raise ValueError("CloseAirCombat env must expose num_agents.")
        self.n = int(env.num_agents)
        self._obs_space = [env.observation_space for _ in range(self.n)]
        self._flat_action_spec = self._build_action_spec(env.action_space)
        if flatten_actions:
            self._action_space = [spaces.Discrete(self._flat_action_spec.size) for _ in range(self.n)]
        else:
            self._action_space = [env.action_space for _ in range(self.n)]

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed: int | None = None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def reset(self):
        obs = self.env.reset()
        return self._split(obs)

    def step(self, actions: Sequence[int]):
        if self.flatten_actions:
            packed_actions = [self._flat_action_spec.unflatten(action) for action in actions]
        else:
            packed_actions = list(actions)
        obs, rewards, dones, info = self.env.step(packed_actions)
        return self._split(obs), self._split(rewards), self._split(dones), info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def _split(self, data: np.ndarray | Sequence[Any]):
        array = np.asarray(data)
        if array.shape[0] != self.n:
            return list(data)
        return [array[idx] for idx in range(self.n)]

    def _build_action_spec(self, space: spaces.Space) -> FlatActionSpec:
        if isinstance(space, spaces.MultiDiscrete):
            return FlatActionSpec(tuple(int(value) for value in space.nvec))
        if isinstance(space, spaces.Tuple):
            nvec: list[int] = []
            for subspace in space.spaces:
                if isinstance(subspace, spaces.MultiDiscrete):
                    nvec.extend(int(value) for value in subspace.nvec)
                elif isinstance(subspace, spaces.Discrete):
                    nvec.append(int(subspace.n))
                else:
                    raise ValueError(f"Unsupported subspace type: {type(subspace)}")
            return FlatActionSpec(tuple(nvec))
        if isinstance(space, spaces.Discrete):
            return FlatActionSpec((int(space.n),))
        raise ValueError(f"Unsupported action space type: {type(space)}")


def _resolve_env_class(env_class: str):
    from envs.JSBSim.envs import MultipleCombatEnv, SingleCombatEnv, SingleControlEnv

    mapping = {
        "SingleCombatEnv": SingleCombatEnv,
        "MultipleCombatEnv": MultipleCombatEnv,
        "SingleControlEnv": SingleControlEnv,
    }
    if env_class not in mapping:
        raise ValueError(f"Unknown env_class '{env_class}'. Expected one of {sorted(mapping.keys())}.")
    return mapping[env_class]


def make_jsbsim_env(
    env_class: str = "MultipleCombatEnv",
    config_name: str = "2v2/NoWeapon/HierarchySelfplay",
    flatten_actions: bool = True,
    **kwargs: Any,
):
    """Create a JSBSim env and wrap it with the CLAM adapter."""
    env_type = _resolve_env_class(env_class)
    env = env_type(config_name, **kwargs)
    return CloseAirCombatAdapter(env, flatten_actions=flatten_actions)


def make_1v1_env(
    config_name: str = "1v1/NoWeapon/HierarchySelfplay",
    flatten_actions: bool = True,
    **kwargs: Any,
):
    return make_jsbsim_env(
        env_class="SingleCombatEnv",
        config_name=config_name,
        flatten_actions=flatten_actions,
        **kwargs,
    )


def make_2v2_env(
    config_name: str = "2v2/NoWeapon/HierarchySelfplay",
    flatten_actions: bool = True,
    **kwargs: Any,
):
    return make_jsbsim_env(
        env_class="MultipleCombatEnv",
        config_name=config_name,
        flatten_actions=flatten_actions,
        **kwargs,
    )
