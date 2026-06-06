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


def _num_series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).to_numpy()


def score_health(df: pd.DataFrame) -> np.ndarray:
    up = _num_series(df, "upvotes")
    com = _num_series(df, "comments")
    stars = _num_series(df, "stars")
    forks = _num_series(df, "forks")
    is_gha = df["source"].astype(str).str.lower().eq("gharchive").to_numpy()

    s_gh = 0.35 * np.log1p(up) + 0.25 * np.log1p(com) + 0.25 * np.log1p(stars) + 0.15 * np.log1p(forks)
    s_gha = 0.55 * np.log1p(com) + 0.45 * np.log1p(up)
    s = np.where(is_gha, s_gha, s_gh)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def score_visibility(df: pd.DataFrame) -> np.ndarray:
    up = _num_series(df, "upvotes")
    com = _num_series(df, "comments")
    stars = _num_series(df, "stars")
    is_gha = df["source"].astype(str).str.lower().eq("gharchive").to_numpy()

    raw_gh = 0.5 * np.log1p(up) + 0.35 * np.log1p(com) + 0.15 * np.log1p(stars)
    raw_gha = 0.65 * np.log1p(com) + 0.35 * np.log1p(up)
    raw = np.where(is_gha, raw_gha, raw_gh)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def score_effort(df: pd.DataFrame) -> np.ndarray:
    # proxy: more comments/issues => more effort; "good first issue" reduces effort
    com = _num_series(df, "comments")
    issues = _num_series(df, "issues_open")
    gfi = _num_series(df, "good_first_issue")

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
    """Ensure domain-relevant GitHub / GH Archive items enter the candidate set."""
    it = interest_text.lower()
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

    if any(k in it for k in ("machine learning", "nlp", "github archive", "gh archive", "gharchive")):
        gha = pool[
            (pool["source"].astype(str) == "gharchive")
            & (pool["domain"].astype(str).str.contains("Machine Learning|AI Research", case=False))
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if not gha.empty:
            extras.append(gha.head(max_extra // 3))

    gha_interest = any(
        k in it
        for k in ("github archive", "gh archive", "gharchive", "archive events", "public timeline", "issue event")
    )
    if gha_interest:
        gha_pool = pool[
            (pool["source"].astype(str) == "gharchive")
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if any(k in it for k in ("kubernetes", "terraform", "devops", "ci/cd", "observability", "infra")):
            gha_pool = gha_pool[gha_pool["domain"].astype(str).str.contains("DevOps", case=False)]
        elif any(k in it for k in ("developer tools", "api", "cli", "b2b", "saas")):
            gha_pool = gha_pool[
                gha_pool["domain"].astype(str).str.contains("Developer Tools|B2B SaaS|Cloud APIs", case=False)
            ]
        elif any(k in it for k in ("trend", "viral", "velocity", "recency")):
            gha_pool = gha_pool.sort_values(["comments", "created_at"], ascending=False)
        if not gha_pool.empty:
            extras.append(gha_pool.head(max_extra // 2))

    if any(k in it for k in ("kubernetes", "terraform", "devops", "ci/cd", "observability", "infra")):
        gha_infra = pool[
            (pool["source"].astype(str) == "gharchive")
            & (pool["domain"].astype(str).str.contains("DevOps", case=False))
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if not gha_infra.empty:
            extras.append(gha_infra.head(max_extra // 3))
        stars = pool["stars"].fillna(0).astype(float)
        gh_niche = pool[
            (pool["source"].astype(str) == "github")
            & (pool["domain"].astype(str).str.contains("DevOps", case=False))
            & (stars >= 200)
            & (stars <= 25000)
            & (~pool["url"].astype(str).isin(existing_urls))
        ]
        if not gh_niche.empty:
            forks = gh_niche["forks"].fillna(0).astype(float)
            st = gh_niche["stars"].fillna(0).astype(float)
            gh_niche = gh_niche[forks <= np.maximum(50, st * 0.15)]
            gh_niche = gh_niche.sort_values(["comments", "stars"], ascending=[False, True])
            extras.append(gh_niche.head(max_extra // 2))

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
    portfolio_mode = any(
        k in interest_text.lower()
        for k in ("good first issue", "beginner", "portfolio", "beginner-friendly")
    )
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
    gha_boost = np.zeros(len(candidates), dtype=np.float64)
    if interest_text.strip():
        it = interest_text.lower()
        gha_interest = any(
            k in it
            for k in ("github archive", "gh archive", "gharchive", "archive events", "public timeline", "issue event")
        )
        for i, dom in enumerate(candidates["domain"].astype(str)):
            d = dom.lower()
            src = str(candidates.iloc[i].get("source", "")).lower()
            if gha_interest and src == "gharchive":
                gha_boost[i] += 0.55 if devops_mode or trend_mode else 0.40
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
        forks = candidates["forks"].fillna(0).astype(float).to_numpy()
        comments = candidates["comments"].fillna(0).astype(float).to_numpy()
        issues = candidates["issues_open"].fillna(0).astype(float).to_numpy()
        # Prefer active but not mega repos — stand-out contributor opportunity
        niche_boost = np.exp(-((np.log1p(stars) - 8.0) ** 2) / 10.0)
        activity = np.log1p(comments + issues)
        low_fork_ratio = forks <= np.maximum(50.0, stars * 0.15)
        mid_size = (stars >= 200.0) & (stars <= 25000.0)
        niche_boost = niche_boost * (1.0 + 0.35 * (activity / (activity.max() + 1e-9)))
        niche_boost = np.where(mid_size & low_fork_ratio, niche_boost * 1.45, niche_boost)
        niche_boost = np.where(stars > 25000, niche_boost * 0.12, niche_boost)
        niche_boost = np.where(stars > 15000, niche_boost * 0.55, niche_boost)
        niche_boost = niche_boost / (niche_boost.max() + 1e-9)

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
            + 0.18 * gha_boost
        )
    elif devops_mode:
        final = (
            0.28 * relevance01
            + 0.18 * health
            + 0.12 * vis
            - cfg.w_effort * effort
            + 0.10 * dom_boost
            + 0.18 * kw_boost
            + 0.42 * niche_boost
            + 0.22 * gha_boost
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
            + 0.18 * gha_boost
        )
        if portfolio_mode:
            final = final + 0.22 * (1.0 - effort)
            # Keep Sofia top-10 quick-start: deprioritize high-effort items unless GFI-tagged
            slow = (effort >= 0.35) & (gfi_boost <= 0)
            final = np.where(slow, final - 0.55, final)

    # Prefer real API-sourced URLs over offline synthetic backup rows
    live = ~candidates["url"].astype(str).str.contains("example.local", na=False)
    final = final + np.where(live.to_numpy(), 0.18, 0.0)

    out = candidates.copy()
    out["score_relevance"] = relevance01
    out["score_health"] = health
    out["score_visibility"] = vis
    out["score_effort"] = effort
    out["score_recency"] = recency
    if trend_mode:
        score_match = np.clip(
            0.30 * relevance01 + 0.35 * vis + 0.20 * recency + 0.15 * kw_boost,
            0.0,
            1.0,
        )
    elif devops_mode:
        score_match = np.clip(
            0.35 * relevance01 + 0.25 * kw_boost + 0.20 * gha_boost + 0.20 * niche_boost,
            0.0,
            1.0,
        )
    elif portfolio_mode:
        score_match = np.clip(
            0.40 * relevance01 + 0.35 * gfi_boost + 0.25 * (1.0 - effort),
            0.0,
            1.0,
        )
    else:
        score_match = relevance01.copy()
    out["score_match"] = score_match
    out["score_final"] = final

    out = out.sort_values("score_final", ascending=False).head(cfg.final_k).reset_index(drop=True)
    return out


def profile_match_pct(ranked: pd.DataFrame) -> int:
    """Header metric: average persona-aware match on top-10 results."""
    top = ranked.head(10)
    if top.empty:
        return 0
    col = "score_match" if "score_match" in top.columns else "score_relevance"
    return int(round(100 * float(top[col].mean())))


def ndcg_at_k(relevances: list[int] | np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum((2**rel - 1) / denom))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum((2**ideal - 1) / denom))
    return dcg / idcg if idcg > 0 else 0.0

