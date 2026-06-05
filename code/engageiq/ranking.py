from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bandit import BetaBandit


@dataclass
class RankConfig:
    candidate_k: int = 200
    final_k: int = 20

    w_relevance: float = 0.55
    w_health: float = 0.20
    w_visibility: float = 0.15
    w_effort: float = 0.10


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_health(df: pd.DataFrame) -> np.ndarray:
    # crude but deterministic health signals across sources
    up = df["upvotes"].fillna(0).astype(float).to_numpy()
    com = df["comments"].fillna(0).astype(float).to_numpy()
    stars = df["stars"].fillna(0).astype(float).to_numpy()
    forks = df["forks"].fillna(0).astype(float).to_numpy()

    s = (
        0.35 * np.log1p(up)
        + 0.25 * np.log1p(com)
        + 0.25 * np.log1p(stars)
        + 0.15 * np.log1p(forks)
    )
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def score_visibility(df: pd.DataFrame) -> np.ndarray:
    # proxy: high upvotes/comments/stars => visibility; penalize extremely old items mildly
    up = df["upvotes"].fillna(0).astype(float).to_numpy()
    com = df["comments"].fillna(0).astype(float).to_numpy()
    stars = df["stars"].fillna(0).astype(float).to_numpy()

    raw = 0.5 * np.log1p(up) + 0.35 * np.log1p(com) + 0.15 * np.log1p(stars)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def score_effort(df: pd.DataFrame) -> np.ndarray:
    # proxy: more comments/issues => more effort; "good first issue" reduces effort
    com = df["comments"].fillna(0).astype(float).to_numpy()
    issues = df["issues_open"].fillna(0).astype(float).to_numpy()
    gfi = df["good_first_issue"].fillna(0).astype(float).to_numpy()

    raw = 0.5 * np.log1p(com) + 0.5 * np.log1p(issues) - 0.75 * gfi
    eff = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return np.clip(eff, 0.0, 1.0)


def rerank(
    candidates: pd.DataFrame,
    relevance01: np.ndarray,
    bandit: BetaBandit | None,
    rng: np.random.Generator,
    cfg: RankConfig,
    interest_text: str = "",
) -> pd.DataFrame:
    health = score_health(candidates)
    vis = score_visibility(candidates)
    effort = score_effort(candidates)

    # bandit provides a prior boost by domain
    if bandit is None:
        dom_boost = np.zeros(len(candidates), dtype=np.float64)
    else:
        w = bandit.sample_weights(rng)
        dom_boost = candidates["domain"].map(lambda d: w.get(str(d), 0.0)).to_numpy()
        dom_boost = (dom_boost - dom_boost.min()) / (dom_boost.max() - dom_boost.min() + 1e-9)

    # persona keyword boost (e.g., Kubernetes → DevOps/K8s domain)
    kw_boost = np.zeros(len(candidates), dtype=np.float64)
    if interest_text.strip():
        it = interest_text.lower()
        for i, dom in enumerate(candidates["domain"].astype(str)):
            d = dom.lower()
            if "devops" in it or "kubernetes" in it or "terraform" in it:
                if "devops" in d:
                    kw_boost[i] += 0.35
            if "machine learning" in it or "nlp" in it or "ml" in it:
                if "machine learning" in d or "ai research" in d:
                    kw_boost[i] += 0.35
            if "developer tools" in it or "api" in it or "cli" in it:
                if "developer tools" in d or "b2b saas" in d or "cloud api" in d:
                    kw_boost[i] += 0.35
            if "trend" in it or "viral" in it:
                kw_boost[i] += 0.15 * float(vis[i])
        if kw_boost.max() > 0:
            kw_boost = kw_boost / (kw_boost.max() + 1e-9)

    final = (
        cfg.w_relevance * relevance01
        + cfg.w_health * health
        + cfg.w_visibility * vis
        - cfg.w_effort * effort
        + 0.10 * dom_boost
        + 0.12 * kw_boost
    )

    out = candidates.copy()
    out["score_relevance"] = relevance01
    out["score_health"] = health
    out["score_visibility"] = vis
    out["score_effort"] = effort
    out["score_final"] = final

    out = out.sort_values("score_final", ascending=False).head(cfg.final_k).reset_index(drop=True)
    return out


def ndcg_at_k(relevances: list[int] | np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum((2**rel - 1) / denom))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum((2**ideal - 1) / denom))
    return dcg / idcg if idcg > 0 else 0.0

