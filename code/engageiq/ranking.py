from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def score_recency(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["created_at"], errors="coerce")
    now = pd.Timestamp.now()
    age_days = (now - ts).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(9999).to_numpy()
    raw = 1.0 / (1.0 + age_days)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def augment_candidates(
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    interest_text: str,
    max_extra: int = 50,
) -> pd.DataFrame:
    """Ensure good-first-issue GitHub items enter the candidate set for portfolio-builder personas."""
    it = interest_text.lower()
    if not any(k in it for k in ("good first issue", "beginner", "portfolio", "open source")):
        return candidates

    existing_urls = set(candidates["url"].astype(str))
    gfi = pool[
        (pool["source"].astype(str) == "github")
        & (pool["good_first_issue"].fillna(0).astype(float) == 1)
        & (~pool["url"].astype(str).isin(existing_urls))
    ]
    if "machine learning" in it or "nlp" in it:
        gfi = gfi[gfi["domain"].astype(str).str.contains("Machine Learning|AI Research", case=False)]
    extra = gfi.head(max_extra)
    if extra.empty:
        return candidates
    return pd.concat([candidates, extra], ignore_index=True).drop_duplicates(subset=["url"])


def rerank(
    candidates: pd.DataFrame,
    relevance01: np.ndarray,
    bandit: BetaBandit | Any | None,
    rng: np.random.Generator,
    cfg: RankConfig,
    interest_text: str = "",
) -> pd.DataFrame:
    health = score_health(candidates)
    vis = score_visibility(candidates)
    effort = score_effort(candidates)
    recency = score_recency(candidates)

    trend_mode = any(k in interest_text.lower() for k in ("trend", "velocity", "viral", "recency"))

    # bandit provides a prior boost by domain
    if bandit is None:
        dom_boost = np.zeros(len(candidates), dtype=np.float64)
    else:
        w = bandit.sample_weights(rng)
        dom_boost = candidates["domain"].map(lambda d: w.get(str(d), 0.0)).to_numpy()
        dom_boost = (dom_boost - dom_boost.min()) / (dom_boost.max() - dom_boost.min() + 1e-9)

    # persona keyword boost (e.g., Kubernetes → DevOps/K8s domain)
    kw_boost = np.zeros(len(candidates), dtype=np.float64)
    gfi_boost = np.zeros(len(candidates), dtype=np.float64)
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
            if any(k in it for k in ("good first issue", "beginner", "portfolio")):
                gfi_val = candidates.iloc[i].get("good_first_issue")
                try:
                    gfi_ok = int(float(gfi_val)) == 1 if gfi_val == gfi_val else False
                except (TypeError, ValueError):
                    gfi_ok = False
                if gfi_ok:
                    gfi_boost[i] += 0.85
                lang = str(candidates.iloc[i].get("lang") or "").lower()
                if lang in ("c++", "rust"):
                    kw_boost[i] -= 0.25
        if kw_boost.max() > 0:
            kw_boost = kw_boost / (kw_boost.max() + 1e-9)

    if trend_mode:
        final = (
            0.25 * relevance01
            + 0.15 * health
            + 0.35 * vis
            + 0.25 * recency
            - cfg.w_effort * effort
            + 0.10 * dom_boost
            + 0.12 * kw_boost
            + 0.35 * gfi_boost
        )
    else:
        final = (
            cfg.w_relevance * relevance01
            + cfg.w_health * health
            + cfg.w_visibility * vis
            - cfg.w_effort * effort
            + 0.10 * dom_boost
            + 0.12 * kw_boost
            + 0.35 * gfi_boost
        )

    # Prefer real API-sourced URLs over offline synthetic backup rows
    live = ~candidates["url"].astype(str).str.contains("example.local", na=False)
    final = final + np.where(live.to_numpy(), 0.18, 0.0)

    out = candidates.copy()
    out["score_relevance"] = relevance01
    out["score_health"] = health
    out["score_visibility"] = vis
    out["score_effort"] = effort
    out["score_recency"] = recency
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

