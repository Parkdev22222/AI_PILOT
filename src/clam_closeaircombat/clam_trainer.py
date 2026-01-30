from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from clam.models.ppo_fnn import PPO_FNN


@dataclass
class TrainMetrics:
    episode: int = 0
    steps: int = 0


class CLAMTrainer:
    """Minimal CLAM trainer adapted for CloseAirCombat."""

    def __init__(
        self,
        embedding_dim: int = 16,
        update_timestep: int = 2048,
        max_episode_steps: int = 1000,
        gamma: float = 0.99,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        eps_clip: float = 0.2,
        k_epochs: int = 80,
        has_vq: bool = False,
        n_rules: int = 3,
        order: int = 3,
        device: str | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.update_timestep = update_timestep
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.has_vq = has_vq
        self.n_rules = n_rules
        self.order = order
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, env: Any, total_steps: int) -> TrainMetrics:
        state_dim = int(env.observation_space[0].shape[0])
        action_dim = int(env.action_space[0].n)
        ppo = PPO_FNN(
            self.device,
            state_dim,
            action_dim,
            self.embedding_dim,
            self.has_vq,
            self.lr_actor,
            self.lr_critic,
            self.gamma,
            self.k_epochs,
            self.eps_clip,
            False,
            action_std_init=0.6,
            n_rules=self.n_rules,
            order=self.order,
        )
        metrics = TrainMetrics()
        obs = env.reset()
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        while metrics.steps < total_steps:
            episode_steps = 0
            done_flags = [False] * env.n
            while episode_steps < self.max_episode_steps and metrics.steps < total_steps:
                actions = []
                for agent_id in range(env.n):
                    state = np.concatenate((obs[agent_id], embedding))
                    if self.has_vq:
                        action, _ = ppo.select_action(state)
                    else:
                        action = ppo.select_action(state)
                    actions.append(int(action))
                next_obs, rewards, dones, _ = env.step(actions)
                for agent_id in range(env.n):
                    reward_value = float(np.asarray(rewards[agent_id]).squeeze())
                    done_value = bool(np.asarray(dones[agent_id]).squeeze())
                    ppo.buffer.rewards.append(reward_value)
                    ppo.buffer.is_terminals.append(done_value)
                    done_flags[agent_id] = done_value
                metrics.steps += 1
                episode_steps += 1
                obs = next_obs
                if metrics.steps % self.update_timestep == 0:
                    ppo.update()
                if all(done_flags):
                    break
            metrics.episode += 1
            obs = env.reset()
        return metrics
