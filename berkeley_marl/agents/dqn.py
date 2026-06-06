"""A compact shared DQN baseline for Berkeley Forest.

This is not a full QMIX implementation. It is a deliberately small baseline that
shares one Q-network across agents and trains from each agent's local transition
with the shared team reward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

# Small MLP updates are faster and more predictable with one CPU thread.
torch.set_num_threads(1)
import torch.nn.functional as F

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class DQNConfig:
    observation_size: int
    n_actions: int
    hidden_size: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.97
    batch_size: int = 64
    replay_size: int = 20_000
    target_update_interval: int = 250
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, n_actions: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Parameter-shared independent DQN policy."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.online = QNetwork(config.observation_size, config.n_actions, config.hidden_size).to(self.device)
        self.target = QNetwork(config.observation_size, config.n_actions, config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: Deque[Transition] = deque(maxlen=config.replay_size)
        self.train_steps = 0

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.config.n_actions)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(obs_t)
            return int(torch.argmax(q_values, dim=-1).item())

    def remember(self, transition: Transition) -> None:
        self.replay.append(transition)

    def remember_many(self, transitions: Iterable[Transition]) -> None:
        self.replay.extend(transitions)

    def update(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None

        batch: List[Transition] = random.sample(self.replay, self.config.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.online(obs_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target(next_obs_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str, metadata: dict | None = None) -> None:
        torch.save(
            {
                "model_state_dict": self.online.state_dict(),
                "config": self.config.__dict__,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        checkpoint = torch.load(path, map_location=device)
        config_dict = dict(checkpoint["config"])
        config_dict["device"] = device
        agent = cls(DQNConfig(**config_dict))
        agent.online.load_state_dict(checkpoint["model_state_dict"])
        agent.target.load_state_dict(agent.online.state_dict())
        return agent
