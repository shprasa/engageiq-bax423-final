from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BetaBandit:
    """
    Thompson sampling over discrete arms (e.g., domains).
    Success=fdbk engage/bookmark, Failure=skip.
    """

    arms: list[str]
    alpha0: float = 1.0
    beta0: float = 1.0

    def __post_init__(self) -> None:
        n = len(self.arms)
        self.alpha = np.full(n, self.alpha0, dtype=np.float64)
        self.beta = np.full(n, self.beta0, dtype=np.float64)
        self._idx = {a: i for i, a in enumerate(self.arms)}

    def sample_weights(self, rng: np.random.Generator) -> dict[str, float]:
        w = rng.beta(self.alpha, self.beta)
        return {a: float(w[i]) for a, i in self._idx.items()}

    def update(self, arm: str, reward01: int) -> None:
        i = self._idx.get(arm)
        if i is None:
            return
        if int(reward01) == 1:
            self.alpha[i] += 1.0
        else:
            self.beta[i] += 1.0

    def mean(self, arm: str) -> float:
        i = self._idx[arm]
        return float(self.alpha[i] / (self.alpha[i] + self.beta[i]))

