from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .bandit import BetaBandit

Policy = Literal["thompson", "epsilon_greedy"]
FeedbackAction = Literal["engage", "bookmark", "skip", "unbookmark"]

REWARD_MAP: dict[str, float] = {
    "engage": 1.0,
    "bookmark": 0.85,
    "skip": 0.0,
    "unbookmark": 0.0,
}


@dataclass
class EngagementRLAgent:
    """
    Reinforcement learning agent for adaptive engagement recommendations.

    Formulation (contextual multi-armed bandit):
    - **State**: user interest profile + session feedback history (implicit in ranking query)
    - **Actions**: recommend opportunities from domain arms (15 technical domains)
    - **Policy**: Thompson sampling (Beta-Bernoulli) or epsilon-greedy over domain success rates
    - **Reward**: engage=+1.0, bookmark=+0.85, skip=0.0 — used to update domain value estimates
    """

    arms: list[str]
    policy: Policy = "thompson"
    epsilon: float = 0.12
    alpha0: float = 1.0
    beta0: float = 1.0
    _bandit: BetaBandit = field(init=False, repr=False)
    reward_history: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    rounds: int = 0

    def __post_init__(self) -> None:
        self._bandit = BetaBandit(arms=self.arms, alpha0=self.alpha0, beta0=self.beta0)

    def sample_weights(self, rng: np.random.Generator) -> dict[str, float]:
        if self.policy == "epsilon_greedy":
            if rng.random() < self.epsilon:
                return {a: float(rng.random()) for a in self.arms}
            q = self.q_values()
            best = max(q.values()) if q else 1.0
            return {a: (1.0 if q.get(a, 0) >= best - 1e-9 else 0.15) for a in self.arms}
        return self._bandit.sample_weights(rng)

    def update(self, arm: str, reward01: float) -> None:
        self._bandit.update(arm, 1 if float(reward01) >= 0.5 else 0)
        self.rounds += 1
        r = float(reward01)
        self.total_reward += r
        self.reward_history.append(
            {
                "round": self.rounds,
                "arm": arm,
                "reward": r,
                "cumulative_reward": self.total_reward,
                "policy": self.policy,
            }
        )

    def observe_feedback(self, domain: str, action: FeedbackAction) -> float:
        reward = REWARD_MAP.get(action, 0.0)
        self.update(domain, reward)
        return reward

    def mean(self, arm: str) -> float:
        return self._bandit.mean(arm)

    def q_values(self) -> dict[str, float]:
        return {a: self.mean(a) for a in self.arms}

    def top_domains(self, k: int = 8) -> list[tuple[str, float]]:
        ranked = sorted(self.q_values().items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def summary(self) -> dict[str, float | str | int]:
        q = list(self.q_values().values())
        entropy = 0.0
        if q and sum(q) > 0:
            p = np.asarray(q, dtype=float)
            p = p / p.sum()
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
        return {
            "policy": self.policy,
            "rounds": self.rounds,
            "total_reward": round(self.total_reward, 3),
            "avg_reward": round(self.total_reward / max(1, self.rounds), 3),
            "policy_entropy": round(entropy, 3),
        }


def persona_reward(interest: str, domain: str, row: pd.Series | None = None) -> float:
    """Binary reward for RL benchmark — 1.0 if domain matches persona interest."""
    dom_l = domain.lower()
    it = interest.lower()

    if any(k in it for k in ("devops", "kubernetes", "terraform", "infra")):
        return 1.0 if "devops" in dom_l else 0.0
    if any(k in it for k in ("developer tools", "api", "cli", "saas", "startup", "b2b")):
        return 1.0 if any(x in dom_l for x in ("developer tools", "b2b saas", "cloud api")) else 0.0
    if any(k in it for k in ("machine learning", "nlp", "ml", "good first", "portfolio")):
        base = 1.0 if any(x in dom_l for x in ("machine learning", "ai research")) else 0.0
        if row is not None:
            try:
                if int(float(row.get("good_first_issue") or 0)) == 1:
                    base = min(1.0, base + 0.2)
            except (TypeError, ValueError):
                pass
        return base
    if any(k in it for k in ("trend", "velocity", "viral", "journalist")):
        return 0.85 if row is not None and float(row.get("score_visibility") or 0) > 0.5 else 0.35

    return 1.0 if any(tok in dom_l for tok in it.split()[:8] if len(tok) > 3) else 0.0


def _select_row_for_round(
    ranked: pd.DataFrame,
    agent: EngagementRLAgent | None,
    rng: np.random.Generator,
    explore: bool,
    work_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Domain-guided selection: RL learns; baseline picks uniformly random domains."""
    if explore and work_df is not None and len(work_df):
        dom = str(rng.choice(sorted(work_df["domain"].dropna().unique())))
        sub = work_df[work_df["domain"] == dom]
        if len(sub):
            return sub.sample(1, random_state=int(rng.integers(0, 2**31))).iloc[0]

    pool = ranked.head(min(50, len(ranked)))
    if pool.empty:
        return ranked.iloc[0]

    if agent is not None:
        weights = agent.sample_weights(rng)
        for dom, _ in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            sub = pool[pool["domain"] == dom]
            if len(sub):
                return sub.iloc[0]

    return pool.iloc[0]


def run_rl_simulation(
    ranked_fn,
    interest: str,
    arms: list[str],
    rounds: int = 60,
    use_rl: bool = True,
    policy: Policy = "thompson",
    seed: int = 42,
    work_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run an RL episode for benchmarking.
    With RL: Thompson sampling over domains guides which item to surface each round.
    Without RL: random domain exploration each round (cold start — no learning).
    """
    rng = np.random.default_rng(seed)
    agent = EngagementRLAgent(arms=arms, policy=policy) if use_rl else None
    ndcgs: list[float] = []
    rewards: list[float] = []
    cumulative: list[float] = []
    total = 0.0

    for t in range(rounds):
        ranked = ranked_fn(agent, rng)
        if ranked.empty:
            break

        row = _select_row_for_round(ranked, agent, rng, explore=not use_rl, work_df=work_df)
        dom = str(row["domain"])
        reward = persona_reward(interest, dom, row)

        # Raj persona: learn to avoid low-engagement / off-topic threads after mid-training
        if t > rounds // 3 and "blockchain" in dom.lower() and "blockchain" not in interest.lower():
            reward = 0.0
        if t > rounds // 2 and float(row.get("comments") or 0) < 5 and "startup" in interest.lower():
            reward = min(reward, 0.25)

        if agent is not None:
            agent.update(dom, reward)

        total += reward
        rewards.append(reward)
        cumulative.append(total)

        labels = [
            1 if persona_reward(interest, str(ranked.loc[i, "domain"])) >= 0.85 else 0
            for i in range(min(10, len(ranked)))
        ]
        from .ranking import ndcg_at_k

        ndcgs.append(ndcg_at_k(labels, 10))

    return {
        "ndcgs": ndcgs,
        "rewards": rewards,
        "cumulative_reward": cumulative,
        "total_reward": total,
        "avg_reward_last10": float(np.mean(rewards[-10:])) if rewards else 0.0,
        "avg_reward_first10": float(np.mean(rewards[:10])) if rewards else 0.0,
        "agent": agent,
    }
