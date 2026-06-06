"""Berkeley Forest environment.

This module intentionally avoids a hard dependency on PettingZoo so the project
can run in small environments. The API follows the spirit of PettingZoo's
ParallelEnv: reset returns per-agent observations and step accepts a dict of
per-agent actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Position = Tuple[int, int]
ObservationDict = Dict[str, np.ndarray]
ActionDict = Dict[str, int]


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for :class:`BerkeleyForestEnv`."""

    condition: str = "berkeley"
    grid_size: int = 7
    n_agents: int = 2
    max_steps: int = 40
    observation_radius: int = 1
    step_penalty: float = -0.02
    false_report_penalty: float = -0.2
    success_reward: float = 10.0
    seed: int | None = None


class BerkeleyForestEnv:
    """A tiny cooperative gridworld with two ontological conditions.

    Agents search for a hidden tree. Observations are local and decentralized;
    rewards are shared by the team.

    Conditions
    ----------
    berkeley:
        Reaching the tree is not enough. The task succeeds only when an agent
        can currently perceive the tree and chooses the REPORT action.
    materialist:
        The tree exists independent of perception. The task succeeds as soon as
        any agent physically reaches the tree location.

    Actions
    -------
    0: stay
    1: up
    2: down
    3: left
    4: right
    5: report
    """

    ACTIONS = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    REPORT_ACTION = 5

    metadata = {"name": "berkeley_forest_v0", "is_parallelizable": True}

    def __init__(self, config: EnvConfig | None = None, **kwargs):
        if config is None:
            config = EnvConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either EnvConfig or keyword arguments, not both.")

        if config.condition not in {"berkeley", "materialist"}:
            raise ValueError("condition must be 'berkeley' or 'materialist'")
        if config.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if config.n_agents < 1:
            raise ValueError("n_agents must be at least 1")

        self.config = config
        self.possible_agents = [f"agent_{i}" for i in range(config.n_agents)]
        self.agents: List[str] = list(self.possible_agents)
        self.rng = np.random.default_rng(config.seed)
        self.step_count = 0
        self.agent_positions: Dict[str, Position] = {}
        self.tree_position: Position = (0, 0)
        self.last_reports: Dict[str, int] = {}
        self.done = False

    @property
    def n_actions(self) -> int:
        return 6

    @property
    def observation_size(self) -> int:
        # own row/col, tree_visible, relative tree row/col, other-agent mean
        # row/col, last own report, condition flag
        return 8

    def reset(self, seed: int | None = None) -> tuple[ObservationDict, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.done = False
        self.agents = list(self.possible_agents)
        self.last_reports = {agent: 0 for agent in self.agents}

        occupied: set[Position] = set()
        self.tree_position = self._sample_empty(occupied)
        occupied.add(self.tree_position)

        self.agent_positions = {}
        for agent in self.agents:
            pos = self._sample_empty(occupied)
            self.agent_positions[agent] = pos
            occupied.add(pos)

        return self._observations(), self._info()

    def step(
        self, actions: ActionDict
    ) -> tuple[ObservationDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        if self.done:
            raise RuntimeError("step() called after episode finished. Call reset().")

        self.step_count += 1
        self.last_reports = {agent: 0 for agent in self.agents}

        for agent in self.agents:
            action = int(actions.get(agent, 0))
            if action in self.ACTIONS:
                self.agent_positions[agent] = self._move(self.agent_positions[agent], action)
            elif action == self.REPORT_ACTION:
                self.last_reports[agent] = 1
            else:
                raise ValueError(f"Invalid action {action!r} for {agent}")

        reward = self.config.step_penalty
        success = False
        false_reports = 0

        if self.config.condition == "materialist":
            success = any(pos == self.tree_position for pos in self.agent_positions.values())
        else:
            for agent, reported in self.last_reports.items():
                if reported and self._can_perceive_tree(agent):
                    success = True
                    break
                if reported and not self._can_perceive_tree(agent):
                    false_reports += 1

        if success:
            reward += self.config.success_reward
            self.done = True
        else:
            reward += false_reports * self.config.false_report_penalty

        timeout = self.step_count >= self.config.max_steps
        if timeout:
            self.done = True

        rewards = {agent: float(reward) for agent in self.agents}
        terminations = {agent: bool(success) for agent in self.agents}
        truncations = {agent: bool(timeout and not success) for agent in self.agents}
        infos = {agent: self._agent_info(agent, success) for agent in self.agents}
        return self._observations(), rewards, terminations, truncations, infos

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)]
        tr, tc = self.tree_position
        grid[tr][tc] = "T"
        for idx, (agent, (row, col)) in enumerate(self.agent_positions.items()):
            grid[row][col] = str(idx)
        return "\n".join(" ".join(row) for row in grid)

    def state_for_demo(self) -> dict:
        return {
            "condition": self.config.condition,
            "grid_size": self.config.grid_size,
            "step": self.step_count,
            "max_steps": self.config.max_steps,
            "tree": {"row": self.tree_position[0], "col": self.tree_position[1]},
            "agents": {
                agent: {
                    "row": pos[0],
                    "col": pos[1],
                    "can_perceive_tree": self._can_perceive_tree(agent),
                    "last_report": bool(self.last_reports.get(agent, 0)),
                }
                for agent, pos in self.agent_positions.items()
            },
            "done": self.done,
        }

    def _sample_empty(self, occupied: set[Position]) -> Position:
        while True:
            pos = (
                int(self.rng.integers(0, self.config.grid_size)),
                int(self.rng.integers(0, self.config.grid_size)),
            )
            if pos not in occupied:
                return pos

    def _move(self, pos: Position, action: int) -> Position:
        dr, dc = self.ACTIONS[action]
        row = min(max(pos[0] + dr, 0), self.config.grid_size - 1)
        col = min(max(pos[1] + dc, 0), self.config.grid_size - 1)
        return (row, col)

    def _can_perceive_tree(self, agent: str) -> bool:
        ar, ac = self.agent_positions[agent]
        tr, tc = self.tree_position
        return abs(ar - tr) + abs(ac - tc) <= self.config.observation_radius

    def _observations(self) -> ObservationDict:
        return {agent: self._observation(agent) for agent in self.agents}

    def _observation(self, agent: str) -> np.ndarray:
        size = float(max(self.config.grid_size - 1, 1))
        row, col = self.agent_positions[agent]
        tr, tc = self.tree_position
        visible = self._can_perceive_tree(agent)

        others = [pos for name, pos in self.agent_positions.items() if name != agent]
        if others:
            mean_other = np.mean(np.asarray(others, dtype=np.float32), axis=0) / size
        else:
            mean_other = np.asarray([row / size, col / size], dtype=np.float32)

        rel_tree = np.asarray([(tr - row) / size, (tc - col) / size], dtype=np.float32) if visible else np.zeros(2)
        condition_flag = 1.0 if self.config.condition == "berkeley" else 0.0

        obs = np.asarray(
            [
                row / size,
                col / size,
                1.0 if visible else 0.0,
                rel_tree[0],
                rel_tree[1],
                mean_other[0],
                mean_other[1],
                condition_flag,
            ],
            dtype=np.float32,
        )
        return obs

    def _info(self) -> dict:
        return {
            "condition": self.config.condition,
            "grid_size": self.config.grid_size,
            "n_agents": self.config.n_agents,
            "max_steps": self.config.max_steps,
        }

    def _agent_info(self, agent: str, success: bool) -> dict:
        return {
            "success": bool(success),
            "can_perceive_tree": self._can_perceive_tree(agent),
            "tree_position": self.tree_position,
            "agent_position": self.agent_positions[agent],
            "last_report": bool(self.last_reports.get(agent, 0)),
        }
