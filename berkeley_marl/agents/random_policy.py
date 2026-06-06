from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Uniform random policy used for smoke tests and the demo."""

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, _obs) -> int:
        return int(self.rng.integers(0, self.n_actions))
