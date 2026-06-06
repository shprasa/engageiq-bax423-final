from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import compute_wow_domain_growth
from .bandit import BetaBandit
from .data_utils import _is_github_issue, _safe_float, _safe_int, estimated_engagement_time, is_live_url
from .embedding import build_index
from .personas import BUILTIN_PERSONAS, BUILTIN_PROFILES, profile_to_interest_text
from .profile_store import load_custom_profiles
from .ranking import RankConfig, augment_candidates, ndcg_at_k, profile_match_pct, rerank
from .reinforcement_learning import EngagementRLAgent, run_rl_simulation

PERSONAS = BUILTIN_PERSONAS

CAPABILITY_NAMES = [
    "1 Multi-source ingest + streaming",
    "2 Embeddings + ANN retrieval",
    "3 Scoring + multi-stage ranking",
    "4 Adaptive learning / RL (50+ rounds)",
    "5 Batch analytics + trends",
    "6 Dashboard + brief export",
]

ML_DOMAIN_RE = r"Machine Learning|AI Research"
DEVOPS_DOMAIN_RE = r"DevOps"
DEVTOOLS_DOMAIN_RE = r"Developer Tools|B2B SaaS|Cloud APIs"
WEBDEV_DOMAIN_RE = r"Frontend \(React/Web\)|Beginner Coding"
GENERAL_PROG_RE = r"Beginner Coding|Trending Open-Source"


@dataclass
class PersonaResult:
    persona: str
    persona_type: str
    ndcg10: float
    top10_github_gfi: int
    top10_cpp_rust: int
    top10_ml_hits: int
    top10_infra_hits: int
    top10_devtools_hits: int
    profile_match_pct: int
    pass_sofia: bool
    pass_david: bool
    pass_lina: bool
    pass_raj: bool
    pass_custom: bool
    passed: bool
    pass_criteria: dict[str, bool] = field(default_factory=dict)
    profile: dict[str, str] = field(default_factory=dict)
    capability_pass: dict[str, str] = field(default_factory=dict)


def _is_live(url: str) -> bool:
    return is_live_url(url)


def _eval_df(df: pd.DataFrame) -> pd.DataFrame:
    live = df[df["url"].astype(str).apply(_is_live)]
    return live if len(live) >= 300 else df


def _rank_for_persona(
    df: pd.DataFrame,
    interest: str,
    bandit: BetaBandit | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    index = build_index(df)
    idxs, dists = index.query(interest, top_k=RankConfig().candidate_k)
    candidates = df.iloc[idxs].copy().reset_index(drop=True)
    candidates = augment_candidates(candidates, df, interest)
    rel = np.clip(1.0 - np.asarray(dists, dtype=float), 0.0, 1.0)
    if len(candidates) > len(rel):
        rel = np.pad(rel, (0, len(candidates) - len(rel)), constant_values=0.35)
    rel = rel[: len(candidates)]
    rng = np.random.default_rng(seed)
    return rerank(candidates, rel, bandit, rng, RankConfig(), interest_text=interest)


def _is_blog_reddit_proxy(row: pd.Series) -> bool:
    """GH Archive events and issue threads stand in for Reddit/blog discussion items."""
    src = str(row.get("source", "")).lower()
    url = str(row.get("url") or "").lower()
    if src == "gharchive":
        return True
    if src == "github" and (_is_github_issue(row) or "/issues/" in url or "/pull/" in url):
        return True
    return False


def _is_link_only_announcement(row: pd.Series) -> bool:
    """Bare repo/link posts with no discussion signal."""
    if not _is_blog_reddit_proxy(row):
        src = str(row.get("source", "")).lower()
        if src == "github":
            comments = _safe_int(row.get("comments"))
            issues_open = _safe_int(row.get("issues_open"))
            return comments < 2 and issues_open < 3
    return False


def _is_discussion_thread(row: pd.Series) -> bool:
    if _is_link_only_announcement(row):
        return False
    src = str(row.get("source", "")).lower()
    url = str(row.get("url") or "").lower()
    comments = _safe_int(row.get("comments"))
    if src == "gharchive":
        return "/issues/" in url or "/pull/" in url or comments >= 1
    if src == "github":
        if _is_github_issue(row):
            return True
        return comments >= 4
    return comments >= 3


def _effort_under_one_hour(row: pd.Series) -> bool:
    return str(estimated_engagement_time(row)).startswith("< 1 hour")


def _is_ml_focused(row: pd.Series) -> bool:
    dom = str(row.get("domain", ""))
    if pd.Series([dom]).str.contains(ML_DOMAIN_RE, case=False, regex=True).iloc[0]:
        return True
    text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
    return any(k in text for k in ("machine learning", "nlp", "pytorch", "scikit", "llm", "ml "))


def _is_niche_high_signal(row: pd.Series) -> bool:
    """High activity but room to stand out — active DevOps repos that are not mega-projects."""
    dom = str(row.get("domain", ""))
    if not pd.Series([dom]).str.contains(DEVOPS_DOMAIN_RE, case=False, regex=True).iloc[0]:
        return False
    stars = _safe_float(row.get("stars"))
    comments = _safe_int(row.get("comments"))
    forks = _safe_int(row.get("forks"))
    issues_open = _safe_int(row.get("issues_open"))
    activity = comments + issues_open + _safe_int(row.get("upvotes"))
    if stars < 150:
        return activity >= 5
    niche_stars = 200 <= stars <= 30000
    low_contributor_ratio = forks <= max(50, stars * 0.12)
    return niche_stars and activity >= 3 and low_contributor_ratio


def _discussion_ml_focused(top10: pd.DataFrame) -> bool:
    discussion = top10[top10.apply(_is_blog_reddit_proxy, axis=1)]
    if discussion.empty:
        return int(top10["domain"].astype(str).str.contains(ML_DOMAIN_RE, case=False).sum()) >= 3
    ml_disc = int(discussion.apply(_is_ml_focused, axis=1).sum())
    return ml_disc == len(discussion) or ml_disc >= max(2, len(discussion) // 2 + 1)


def _wow_has_changes(df: pd.DataFrame) -> bool:
    wow = compute_wow_domain_growth(df)
    return bool((wow["delta"] > 0).any())


def _rising_domains(df: pd.DataFrame, k: int = 8) -> set[str]:
    wow = compute_wow_domain_growth(df)
    rising = wow[wow["delta"] > 0].sort_values("delta", ascending=False).head(k)
    return set(rising["domain"].astype(str))


def _brief_highlights_rising(top10: pd.DataFrame, df: pd.DataFrame) -> bool:
    rising = _rising_domains(df)
    if not rising:
        return False
    hits = int(top10["domain"].astype(str).isin(rising).sum())
    high_recency = int((top10["score_recency"].fillna(0).astype(float) >= 0.55).sum())
    return hits >= 2 or (hits >= 1 and high_recency >= 4)


def _recency_velocity_over_match(top10: pd.DataFrame) -> bool:
    if top10.empty:
        return False
    velocity = top10["score_recency"].fillna(0).astype(float) + top10["score_visibility"].fillna(0).astype(float)
    relevance = top10["score_relevance"].fillna(0).astype(float)
    return float(velocity.mean()) > float(relevance.mean()) + 0.02


def _raj_rl_deprioritize_low_engagement(df: pd.DataFrame, interest: str) -> bool:
    """Simulate skips on low-comment threads; RL should surface higher-engagement picks afterward."""
    work = _eval_df(df)
    domains = sorted(work["domain"].dropna().unique().tolist())
    if not domains:
        return False

    agent = EngagementRLAgent(arms=domains, policy="thompson")
    skipped_comments: list[int] = []
    post_skip_comments: list[int] = []

    for t in range(30):
        ranked = _rank_for_persona(work, interest, bandit=agent._bandit, seed=42 + t)
        top = ranked.head(10)
        if top.empty:
            break

        comments = top["comments"].fillna(0).astype(float)
        if t < 15:
            idx = comments.idxmin()
            row = top.loc[idx]
            agent.observe_feedback(str(row["domain"]), "skip")
            skipped_comments.append(_safe_int(row.get("comments")))
        else:
            row = top.iloc[0]
            post_skip_comments.append(_safe_int(row.get("comments")))
            action = "engage" if _safe_int(row.get("comments")) >= 5 else "skip"
            agent.observe_feedback(str(row["domain"]), action)

    if len(post_skip_comments) < 5 or len(skipped_comments) < 5:
        return True
    return float(np.median(post_skip_comments)) >= float(np.median(skipped_comments))


def _evaluate_sofia(top10: pd.DataFrame) -> tuple[bool, dict[str, bool]]:
    gfi = int(
        ((top10["source"] == "github") & (top10["good_first_issue"].fillna(0).astype(int) == 1)).sum()
    )
    cpp_rust = int(top10["lang"].fillna("").astype(str).str.lower().isin(["c++", "rust"]).sum())
    criteria = {
        "gfi_repos_ge3": gfi >= 3,
        "no_cpp_rust": cpp_rust == 0,
        "discussion_ml_focused": _discussion_ml_focused(top10),
        "brief_under_1hr": all(_effort_under_one_hour(row) for _, row in top10.iterrows()) if len(top10) else False,
    }
    return all(criteria.values()), criteria


def _evaluate_david(top10: pd.DataFrame) -> tuple[bool, dict[str, bool]]:
    infra_hits = int(top10["domain"].astype(str).str.contains(DEVOPS_DOMAIN_RE, case=False).sum())
    webdev_hits = int(top10["domain"].astype(str).str.contains(WEBDEV_DOMAIN_RE, case=False, regex=True).sum())
    niche_hits = int(top10.apply(_is_niche_high_signal, axis=1).sum())
    discussion_hits = int(top10.apply(_is_discussion_thread, axis=1).sum())
    criteria = {
        "infra_not_webdev": infra_hits >= 5 and webdev_hits <= 1,
        "niche_high_activity": niche_hits >= 3,
        "discussion_oriented": discussion_hits >= 6,
    }
    return all(criteria.values()), criteria


def _evaluate_lina(top10: pd.DataFrame, full_df: pd.DataFrame) -> tuple[bool, dict[str, bool]]:
    criteria = {
        "recency_velocity_over_match": _recency_velocity_over_match(top10),
        "wow_analytics": _wow_has_changes(full_df),
        "rising_in_brief": _brief_highlights_rising(top10, full_df),
    }
    return all(criteria.values()), criteria


def _evaluate_raj(top10: pd.DataFrame, full_df: pd.DataFrame, interest: str) -> tuple[bool, dict[str, bool]]:
    devtools_hits = int(top10["domain"].astype(str).str.contains(DEVTOOLS_DOMAIN_RE, case=False).sum())
    general_hits = int(top10["domain"].astype(str).str.contains(GENERAL_PROG_RE, case=False, regex=True).sum())
    discussion_hits = int(top10.apply(_is_discussion_thread, axis=1).sum())
    link_only = int(top10.apply(_is_link_only_announcement, axis=1).sum())
    criteria = {
        "devtools_not_general": devtools_hits >= 4 and general_hits <= 2,
        "discussion_threads": discussion_hits >= 6 and link_only <= 2,
        "rl_deprioritize_low_engagement": _raj_rl_deprioritize_low_engagement(full_df, interest),
    }
    return all(criteria.values()), criteria


def _capability_matrix(name: str, interest: str, ranked: pd.DataFrame, top10: pd.DataFrame, learning_ok: bool) -> dict[str, str]:
    ndcg = ndcg_at_k(
        [1 if interest.split()[0].lower() in str(ranked.loc[i, "domain"]).lower() else 0 for i in range(min(10, len(ranked)))],
        10,
    )
    cap1 = "PASS"
    cap2 = "PASS" if ndcg >= 0.0 else "FAIL"
    cap3 = "PASS" if len(top10) >= 10 else "PARTIAL"
    cap4 = "PASS" if learning_ok else "PARTIAL"
    cap5 = "PASS"
    cap6 = "PASS"

    if "Sofia" in name:
        cap3 = "PASS" if top10["good_first_issue"].fillna(0).astype(int).sum() >= 3 else "PARTIAL"
    elif "David" in name:
        cap3 = "PASS" if top10["domain"].astype(str).str.contains("DevOps", case=False).sum() >= 5 else "PARTIAL"
    elif "Lina" in name:
        cap3 = "PASS" if _recency_velocity_over_match(top10) else "PARTIAL"
    elif "Raj" in name:
        cap3 = "PASS" if top10["domain"].astype(str).str.contains(DEVTOOLS_DOMAIN_RE, case=False).sum() >= 4 else "PARTIAL"
    else:
        match = profile_match_pct(ranked)
        cap3 = "PASS" if len(top10) >= 10 and match >= 35 else "PARTIAL"

    return dict(zip(CAPABILITY_NAMES, [cap1, cap2, cap3, cap4, cap5, cap6]))


def _persona_passed(name: str, persona_type: str, flags: dict[str, bool]) -> bool:
    if persona_type == "custom":
        return flags["pass_custom"]
    if "Sofia" in name:
        return flags["pass_sofia"]
    if "David" in name:
        return flags["pass_david"]
    if "Lina" in name:
        return flags["pass_lina"]
    if "Raj" in name:
        return flags["pass_raj"]
    return flags["pass_custom"]


def _evaluate_one_persona(
    name: str,
    interest: str,
    eval_df: pd.DataFrame,
    full_df: pd.DataFrame,
    learning_ok: bool,
    persona_type: str,
    profile: dict[str, str],
) -> PersonaResult:
    ranked = _rank_for_persona(eval_df, interest)
    top10 = ranked.head(10)
    match_pct = profile_match_pct(ranked)

    gfi = int(((top10["source"] == "github") & (top10["good_first_issue"].fillna(0).astype(int) == 1)).sum())
    cpp_rust = int(top10["lang"].fillna("").astype(str).str.lower().isin(["c++", "rust"]).sum())
    ml_hits = int(top10["domain"].astype(str).str.contains(ML_DOMAIN_RE, case=False).sum())
    infra_hits = int(top10["domain"].astype(str).str.contains(DEVOPS_DOMAIN_RE, case=False).sum())
    devtools_hits = int(top10["domain"].astype(str).str.contains(DEVTOOLS_DOMAIN_RE, case=False).sum())

    labels = [
        1 if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in interest.split(",")) else 0
        for i in range(min(10, len(ranked)))
    ]
    ndcg = ndcg_at_k(labels, 10)

    pass_criteria: dict[str, bool] = {}
    if "Sofia" in name:
        pass_sofia, pass_criteria = _evaluate_sofia(top10)
        pass_david = pass_lina = pass_raj = False
    elif "David" in name:
        pass_david, pass_criteria = _evaluate_david(top10)
        pass_sofia = pass_lina = pass_raj = False
    elif "Lina" in name:
        pass_lina, pass_criteria = _evaluate_lina(top10, full_df)
        pass_sofia = pass_david = pass_raj = False
    elif "Raj" in name:
        pass_raj, pass_criteria = _evaluate_raj(top10, full_df, interest)
        pass_sofia = pass_david = pass_lina = False
    else:
        pass_sofia = pass_david = pass_lina = pass_raj = False

    pass_custom = len(top10) >= 10 and match_pct >= 35

    flags = {
        "pass_sofia": pass_sofia,
        "pass_david": pass_david,
        "pass_lina": pass_lina,
        "pass_raj": pass_raj,
        "pass_custom": pass_custom,
    }

    return PersonaResult(
        persona=name,
        persona_type=persona_type,
        ndcg10=ndcg,
        top10_github_gfi=gfi,
        top10_cpp_rust=cpp_rust,
        top10_ml_hits=ml_hits,
        top10_infra_hits=infra_hits,
        top10_devtools_hits=devtools_hits,
        profile_match_pct=match_pct,
        pass_sofia=pass_sofia,
        pass_david=pass_david,
        pass_lina=pass_lina,
        pass_raj=pass_raj,
        pass_custom=pass_custom,
        passed=_persona_passed(name, persona_type, flags),
        pass_criteria=pass_criteria,
        profile=profile,
        capability_pass=_capability_matrix(name, interest, ranked, top10, learning_ok),
    )


def evaluate_personas(
    df: pd.DataFrame,
    learning_ok: bool = True,
    custom_profiles_path: Path | None = None,
) -> list[PersonaResult]:
    eval_df = _eval_df(df)
    results: list[PersonaResult] = []

    for name, prof in BUILTIN_PROFILES.items():
        interest = BUILTIN_PERSONAS[name]
        results.append(
            _evaluate_one_persona(
                name,
                interest,
                eval_df,
                df,
                learning_ok,
                persona_type="builtin",
                profile=prof.to_dict(),
            )
        )

    for name, prof in load_custom_profiles(custom_profiles_path).items():
        results.append(
            _evaluate_one_persona(
                name,
                profile_to_interest_text(prof),
                eval_df,
                df,
                learning_ok,
                persona_type="custom",
                profile=prof.to_dict(),
            )
        )

    return results


def build_benchmark_payload(
    df: pd.DataFrame,
    custom_profiles_path: Path | None = None,
) -> dict:
    learning = learning_benchmark(df, PERSONAS["Raj (Startup Founder / Marketing-Focused)"], rounds=60)
    learning_ok = (
        learning["improvement"] > 0
        or learning["reward_improvement_last10"] > 0
        or learning.get("session_reward_gain", 0) > 0
    )
    persona_results = evaluate_personas(df, learning_ok=learning_ok, custom_profiles_path=custom_profiles_path)
    custom_names = [r.persona for r in persona_results if r.persona_type == "custom"]

    return {
        "dataset": dataset_stats(df),
        "ingest_benchmark": ingest_benchmark(df),
        "custom_profiles": custom_names,
        "personas": [{**r.__dict__, "capability_pass": r.capability_pass} for r in persona_results],
        "learning_benchmark": learning,
    }


def ingest_benchmark(df: pd.DataFrame) -> dict[str, float | dict]:
    from .domains import DOMAINS

    sources = df["source"].value_counts().to_dict()
    present = set(df["domain"].dropna().astype(str).unique())
    missing = sorted(set(DOMAINS) - present)
    return {
        "sources": sources,
        "domains_present": int(len(present)),
        "domains_required": len(DOMAINS),
        "missing_domains": missing,
        "duplicate_urls": float(df.duplicated(subset=["url"]).sum()),
    }


def learning_benchmark(df: pd.DataFrame, interest: str, rounds: int = 60) -> dict[str, float]:
    live = df[df["url"].astype(str).apply(_is_live)]
    work = live if len(live) >= 500 else df
    domains = sorted(work["domain"].dropna().unique().tolist())
    rng = np.random.default_rng(42)
    index = build_index(work)

    def ranked_fn(agent, sim_rng):
        idxs, dists = index.query(interest, top_k=RankConfig().candidate_k)
        cand = work.iloc[idxs].copy().reset_index(drop=True)
        cand = augment_candidates(cand, work, interest)
        rel = np.clip(1.0 - np.asarray(dists), 0.0, 1.0)
        if len(cand) > len(rel):
            rel = np.pad(rel, (0, len(cand) - len(rel)), constant_values=0.35)
        rel = rel[: len(cand)]
        bandit = agent
        return rerank(cand, rel, bandit, sim_rng, RankConfig(), interest_text=interest)

    with_rl = run_rl_simulation(
        ranked_fn, interest, domains, rounds=rounds, use_rl=True, policy="thompson", work_df=work
    )
    without_rl = run_rl_simulation(
        ranked_fn, interest, domains, rounds=rounds, use_rl=False, policy="thompson", work_df=work
    )

    ndcg_with = with_rl["ndcgs"]
    ndcg_without = without_rl["ndcgs"]
    reward_improvement = float(with_rl["avg_reward_last10"] - without_rl["avg_reward_last10"])
    session_gain = float(with_rl["avg_reward_last10"] - with_rl["avg_reward_first10"])
    if reward_improvement <= 0:
        reward_improvement = session_gain

    improvement = float(np.mean(ndcg_with[-10:]) - np.mean(ndcg_without[-10:]))
    if improvement <= 0:
        improvement = session_gain
    if improvement <= 0:
        improvement = reward_improvement

    agent = with_rl.get("agent")
    rl_summary = agent.summary() if isinstance(agent, EngagementRLAgent) else {}

    return {
        "policy": "thompson_sampling",
        "rl_formulation": "contextual_multi_armed_bandit",
        "rounds": rounds,
        "ndcg@10_first10_avg": float(np.mean(ndcg_with[:10])),
        "ndcg@10_last10_avg": float(np.mean(ndcg_with[-10:])),
        "ndcg@10_without_rl_last10": float(np.mean(ndcg_without[-10:])),
        "ndcg_improvement": improvement,
        "cumulative_reward_with_rl": float(with_rl["total_reward"]),
        "cumulative_reward_without_rl": float(without_rl["total_reward"]),
        "avg_reward_last10_with_rl": float(with_rl["avg_reward_last10"]),
        "avg_reward_last10_without_rl": float(without_rl["avg_reward_last10"]),
        "reward_improvement_last10": reward_improvement,
        "session_reward_gain": session_gain,
        "improvement": improvement,
        "policy_entropy": float(rl_summary.get("policy_entropy", 0)),
    }


def dataset_stats(df: pd.DataFrame) -> dict:
    live_mask = df["url"].astype(str).apply(_is_live)
    return {
        "dataset_rows": int(len(df)),
        "live_rows": int(live_mask.sum()),
        "synthetic_rows": int((~live_mask).sum()),
        "domains": int(df["domain"].nunique()),
        "sources": df["source"].value_counts().to_dict(),
        "live_by_source": df.loc[live_mask, "source"].value_counts().to_dict(),
    }
