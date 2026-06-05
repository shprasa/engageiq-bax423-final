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
    up = df["upvotes"].fillna(0).astype(float).to_numpy()
    com = df["comments"].fillna(0).astype(float).to_numpy()
    stars = df["stars"].fillna(0).astype(float).to_numpy()
    forks = df["forks"].fillna(0).astype(float).to_numpy()
    is_hn = df["source"].astype(str).str.lower().eq("hackernews").to_numpy()

    s_gh = 0.35 * np.log1p(up) + 0.25 * np.log1p(com) + 0.25 * np.log1p(stars) + 0.15 * np.log1p(forks)
    s_hn = 0.55 * np.log1p(up) + 0.45 * np.log1p(com)
    s = np.where(is_hn, s_hn, s_gh)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def score_visibility(df: pd.DataFrame) -> np.ndarray:
    up = df["upvotes"].fillna(0).astype(float).to_numpy()
    com = df["comments"].fillna(0).astype(float).to_numpy()
    stars = df["stars"].fillna(0).astype(float).to_numpy()
    is_hn = df["source"].astype(str).str.lower().eq("hackernews").to_numpy()

    raw_gh = 0.5 * np.log1p(up) + 0.35 * np.log1p(com) + 0.15 * np.log1p(stars)
    raw_hn = 0.55 * np.log1p(up) + 0.45 * np.log1p(com)
    raw = np.where(is_hn, raw_hn, raw_gh)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def score_effort(df: pd.DataFrame) -> np.ndarray:
    # proxy: more comments/issues => more effort; "good first issue" reduces effort
    com = df["comments"].fillna(0).astype(float).to_numpy()
    issues = df["issues_open"].fillna(0).astype(float).to_numpy()
    gfi = df["good_first_issue"].fillna(0).astype(float).to_numpy()

    raw = 0.5 * np.log1p(com) + 0.5 * np.log1p(issues) - 0.75 * gfi
    eff = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    eff = np.clip(eff, 0.0, 1.0)
    # GFI items should always rank as low-effort for portfolio-builder personas
    eff = np.where(gfi >= 1, np.minimum(eff, 0.25), eff)
    return eff


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
    extras: list[pd.DataFrame] = []

    if any(k in it for k in ("good first issue", "beginner", "portfolio", "open source")):
        gfi = pool[
            (pool["source"].astype(str) == "github")
            & (pool["good_first_issue"].fillna(0).astype(float) == 1)
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if "machine learning" in it or "nlp" in it:
            gfi = gfi[gfi["domain"].astype(str).str.contains("Machine Learning|AI Research", case=False)]
        if not gfi.empty:
            extras.append(gfi.head(max_extra // 2))

    if any(k in it for k in ("machine learning", "nlp", "hacker news", "ml threads")):
        hn = pool[
            (pool["source"].astype(str) == "hackernews")
            & (pool["domain"].astype(str).str.contains("Machine Learning|AI Research", case=False))
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if not hn.empty:
            extras.append(hn.head(max_extra // 3))

    hn_interest = any(
        k in it
        for k in ("hacker news", "hackernews", " hn ", "hn threads", "infra threads", "ml threads", "discussions")
    )
    if hn_interest:
        hn_pool = pool[
            (pool["source"].astype(str) == "hackernews")
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if any(k in it for k in ("kubernetes", "terraform", "devops", "ci/cd", "observability", "infra")):
            hn_pool = hn_pool[hn_pool["domain"].astype(str).str.contains("DevOps", case=False)]
        elif any(k in it for k in ("developer tools", "api", "cli", "b2b", "saas")):
            hn_pool = hn_pool[
                hn_pool["domain"].astype(str).str.contains("Developer Tools|B2B SaaS|Cloud APIs", case=False)
            ]
        elif any(k in it for k in ("trend", "viral", "velocity", "recency")):
            hn_pool = hn_pool.sort_values(["upvotes", "comments"], ascending=False)
        if not hn_pool.empty:
            extras.append(hn_pool.head(max_extra // 2))

    if any(k in it for k in ("kubernetes", "terraform", "devops", "ci/cd", "observability", "infra")):
        hn_infra = pool[
            (pool["source"].astype(str) == "hackernews")
            & (pool["domain"].astype(str).str.contains("DevOps", case=False))
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if not hn_infra.empty:
            extras.append(hn_infra.head(max_extra // 3))
        stars = pool["stars"].fillna(0).astype(float)
        gh_niche = pool[
            (pool["source"].astype(str) == "github")
            & (pool["domain"].astype(str).str.contains("DevOps", case=False))
            & (stars >= 500)
            & (stars <= 25000)
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if not gh_niche.empty:
            extras.append(gh_niche.head(max_extra // 3))

    if not extras:
        return candidates
    extra = pd.concat(extras, ignore_index=True).drop_duplicates(subset=["url"])
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
    portfolio_mode = any(k in interest_text.lower() for k in ("good first issue", "beginner", "portfolio"))
    devops_mode = any(
        k in interest_text.lower() for k in ("kubernetes", "terraform", "devops", "ci/cd", "observability")
    )

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
    hn_boost = np.zeros(len(candidates), dtype=np.float64)
    if interest_text.strip():
        it = interest_text.lower()
        hn_interest = any(
            k in it
            for k in ("hacker news", "hackernews", " hn ", "hn threads", "infra threads", "ml threads", "discussions")
        )
        for i, dom in enumerate(candidates["domain"].astype(str)):
            d = dom.lower()
            src = str(candidates.iloc[i].get("source", "")).lower()
            if hn_interest and src == "hackernews":
                hn_boost[i] += 0.55 if devops_mode or trend_mode else 0.40
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

    niche_boost = np.zeros(len(candidates), dtype=np.float64)
    if devops_mode:
        stars = candidates["stars"].fillna(0).astype(float).to_numpy()
        # Prefer active but not mega repos — stand-out contributor opportunity
        niche_boost = np.exp(-((np.log1p(stars) - 8.0) ** 2) / 12.0)
        niche_boost = niche_boost / (niche_boost.max() + 1e-9)
        niche_boost = np.where(stars > 30000, niche_boost * 0.35, niche_boost)

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
            + 0.12 * niche_boost
            + 0.18 * hn_boost
        )
    else:
        effort_term = cfg.w_effort * effort
        if portfolio_mode:
            effort_term = cfg.w_effort * 1.35 * effort
        final = (
            cfg.w_relevance * relevance01
            + cfg.w_health * health
            + cfg.w_visibility * vis
            - effort_term
            + 0.10 * dom_boost
            + 0.12 * kw_boost
            + 0.35 * gfi_boost
            + 0.12 * niche_boost
            + 0.18 * hn_boost
        )
        if portfolio_mode:
            final = final + 0.10 * (1.0 - effort)

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

