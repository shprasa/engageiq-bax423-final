from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engageiq.analytics import compute_trends, compute_trends_from_df, compute_wow_domain_growth
from engageiq.brief_export import BriefConfig, export_brief_csv, export_brief_pdf
from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.data_utils import (
    _safe_int,
    display_title,
    filter_ranked_results,
    is_live_url,
    live_mask,
    ORIGIN_FILTER_OPTIONS,
    ranking_corpus,
    sort_ranked_results,
    source_mix_summary,
    SORT_OPTIONS,
    SOURCE_FILTER_OPTIONS,
)
from engageiq.domains import DOMAINS
from engageiq.embedding import build_index
from engageiq.ranking import RankConfig, augment_candidates, ndcg_at_k, rerank
from engageiq.reinforcement_learning import EngagementRLAgent
from engageiq.streaming import OpportunityStream
from engageiq.suggestions import generate_suggestion, llm_configured, suggestion_provider
from engageiq.ui import (
    Action,
    activity_counts,
    init_activity_state,
    inject_theme,
    record_feedback,
    render_activity_log,
    render_bookmarks_list,
    render_hero,
    render_opportunity_card,
    render_rl_policy,
)

PERSONAS: dict[str, str] = {
    "Sofia (ML Student / Portfolio Builder)": (
        "Machine learning, NLP, data pipelines, beginner-friendly open source, good first issues, "
        "Python, pandas, GitHub issues, GitHub Archive ML issue events."
    ),
    "David (DevOps / Niche Community)": (
        "Kubernetes, Terraform, CI/CD, observability, cloud-native infra, high-activity repos, "
        "few contributors, GitHub Archive DevOps issue and PR events."
    ),
    "Lina (Data Journalist / Trend Spotter)": (
        "Trending repos, viral discussions, emerging tools, fast-growing communities, recency, velocity, "
        "GitHub Archive public timeline events, GitHub trending, multi-domain velocity."
    ),
    "Raj (Startup Founder / Marketing-Focused)": (
        "Developer tools, APIs, CLI tools, open-source business, B2B SaaS, discussions where devtools are relevant, "
        "GitHub Archive and GitHub developer-tools communities."
    ),
}

CHART_THEME = {
    "background": "transparent",
    "view": {"stroke": "transparent"},
    "axis": {"labelColor": "#64748B", "titleColor": "#334155", "gridColor": "#E2E8F0"},
}


@dataclass
class UserState:
    interest_text: str
    liked_texts: list[str]
    rl_agent: EngagementRLAgent
    rng: np.random.Generator


def _purge_stale_duckdb(duckdb_path: Path, snapshot_csv: Path) -> None:
    if not duckdb_path.exists() or not snapshot_csv.exists():
        return
    try:
        csv_live = int(live_mask(pd.read_csv(snapshot_csv, usecols=["url"])).sum())
        if csv_live == 0:
            return
        import duckdb

        con = duckdb.connect(str(duckdb_path))
        try:
            db_live = int(
                con.execute(
                    "SELECT COUNT(*) FROM opportunities WHERE url NOT LIKE '%example.local%'"
                ).fetchone()[0]
            )
        except Exception:
            db_live = 0
        finally:
            con.close()
        if db_live == 0:
            duckdb_path.unlink(missing_ok=True)
            Path(str(duckdb_path) + ".wal").unlink(missing_ok=True)
    except Exception:
        pass


@st.cache_resource
def _load_store_and_seed() -> tuple[OpportunityStore, dict]:
    paths = get_paths()
    _purge_stale_duckdb(paths.duckdb_path, paths.snapshot_csv)
    store = OpportunityStore(paths.duckdb_path, snapshot_csv=paths.snapshot_csv)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=100000)
    return store, {"paths": paths, "version": 13}


@st.cache_resource
def _build_embedding_index(_version: int, corpus_key: str, df: pd.DataFrame):
    return build_index(df)


def _init_user_state(domains: list[str]) -> UserState:
    return UserState(
        interest_text=PERSONAS["Sofia (ML Student / Portfolio Builder)"],
        liked_texts=[],
        rl_agent=EngagementRLAgent(arms=domains, policy="thompson"),
        rng=np.random.default_rng(42),
    )


def _explain_row(row: pd.Series) -> str:
    parts = [
        f"Match {row['score_relevance']:.0%} to your interests",
        f"Community activity {row['score_health']:.0%}",
        f"Visibility {row['score_visibility']:.0%}",
        f"Effort level {row['score_effort']:.0%}",
    ]
    if is_live_url(str(row.get("url", ""))):
        parts.append("live web data")
    if str(row.get("source", "")) == "github" and _safe_int(row.get("good_first_issue")) == 1:
        parts.append("beginner-friendly issue")
    return " · ".join(parts)


def _suggest_action(row: pd.Series, interest: str) -> str:
    if "suggestion_cache" not in st.session_state:
        st.session_state.suggestion_cache = {}
    suggestion = generate_suggestion(row, interest, cache=st.session_state.suggestion_cache)
    provider = suggestion_provider()
    if provider != "template":
        return f"[AI · {provider.title()}] {suggestion}"
    return suggestion


def _apply_pending_feedback(user: UserState) -> None:
    pending = st.session_state.pop("_pending_action", None)
    if not pending:
        return
    action: Action
    row_data: dict
    action, row_data = pending
    row = pd.Series(row_data)
    record_feedback(row, action)
    reward = user.rl_agent.observe_feedback(str(row["domain"]), action)
    if action in ("engage", "bookmark"):
        snippet = f"{row['domain']}: {display_title(row)}"
        if snippet not in user.liked_texts:
            user.liked_texts.append(snippet)
    st.toast(
        f"RL reward {reward:+.2f} · {action.replace('_', ' ').title()}",
        icon="✅" if reward > 0 else "⏭️",
    )


def _chart(df: pd.DataFrame, mark_fn, encode_kwargs: dict) -> alt.Chart:
    return mark_fn(alt.Chart(df).encode(**encode_kwargs)).properties(height=280).configure(**CHART_THEME)


def _rank_opportunities(
    df: pd.DataFrame,
    user: UserState,
    interest_text: str,
    version: int,
    live_only: bool,
    english_only: bool = True,
    result_limit: int = 100,
) -> pd.DataFrame:
    corpus = ranking_corpus(df, live_only=live_only, english_only=english_only)
    corpus_key = f"{len(corpus)}_{live_only}_{english_only}_{hash(tuple(corpus['id'].head(5).tolist()))}"
    index = _build_embedding_index(version, corpus_key, corpus)
    q = interest_text.strip()
    if user.liked_texts:
        q += "\n\nRecently liked:\n" + "\n".join(user.liked_texts[-5:])

    idxs, dists = index.query(q, top_k=RankConfig().candidate_k)
    candidates = corpus.iloc[idxs].copy().reset_index(drop=True)
    candidates = augment_candidates(candidates, corpus, q)
    relevance = np.clip(1.0 - np.asarray(dists, dtype=np.float64), 0.0, 1.0)
    if len(candidates) > len(relevance):
        relevance = np.pad(relevance, (0, len(candidates) - len(relevance)), constant_values=0.35)
    relevance = relevance[: len(candidates)]
    cfg = RankConfig(final_k=result_limit)
    return rerank(
        candidates=candidates,
        relevance01=relevance,
        bandit=user.rl_agent,
        rng=user.rng,
        cfg=cfg,
        interest_text=q,
    )


def main() -> None:
    st.set_page_config(
        page_title="EngageIQ",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    init_activity_state()

    store, ctx = _load_store_and_seed()
    paths = ctx["paths"]
    df = store.load_df()
    domains = sorted(df["domain"].dropna().unique().tolist())
    live_n = int(live_mask(df).sum())
    version = ctx["version"]
    domain_count = df["domain"].nunique()
    missing_domains = sorted(set(DOMAINS) - set(domains))

    if "user_state" not in st.session_state:
        st.session_state.user_state = _init_user_state(domains)
    user: UserState = st.session_state.user_state
    _apply_pending_feedback(user)

    if "opportunity_stream" not in st.session_state:
        st.session_state.opportunity_stream = OpportunityStream()

    counts = activity_counts()

    with st.sidebar:
        st.markdown("### EngageIQ")
        st.caption("Recommendation + reinforcement learning")
        st.markdown("---")
        st.markdown("**Session**")
        m1, m2 = st.columns(2)
        m1.metric("Engaged", counts["engaged"])
        m2.metric("Saved", counts["bookmarked"])
        m3, m4 = st.columns(2)
        m3.metric("Skipped", counts["skipped"])
        m4.metric("Actions", counts["total_actions"])

        if st.button("Clear activity & bookmarks", use_container_width=True):
            st.session_state.activity_log = []
            st.session_state.bookmarks = {}
            user.liked_texts = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Interest profile**")
        persona = st.selectbox("Persona preset", options=list(PERSONAS.keys()), label_visibility="collapsed")
        if st.button("Load persona", use_container_width=True):
            user.interest_text = PERSONAS[persona]
            st.rerun()

        user.interest_text = st.text_area(
            "Describe what you want to engage with",
            value=user.interest_text,
            height=120,
            label_visibility="collapsed",
        )

        live_only = st.toggle(
            "Rank from live API pool only",
            value=True,
            help="When on, ranking uses live GitHub API + GitHub Archive URLs. Turn off to include offline backup rows in the ranked pool.",
        )
        english_only = st.toggle("English only", value=True)

        st.markdown("---")
        st.markdown("**Streaming pipeline**")
        stream: OpportunityStream = st.session_state.opportunity_stream
        batch_size = st.slider("Batch size", 50, 1000, 300, 50)
        c_prod, c_cons = st.columns(2)
        if c_prod.button("Produce batch", use_container_width=True):
            snapshot = pd.read_csv(paths.snapshot_csv)
            max_id = store.max_id()
            batch = snapshot[snapshot["id"] > max_id].head(int(batch_size)).copy()
            if batch.empty:
                batch = snapshot.sample(n=min(int(batch_size), len(snapshot)), random_state=42)
            n = stream.produce_rows(batch)
            st.success(f"Queued {n} records ({stream.pending()} pending)")
        if c_cons.button("Consume → store", use_container_width=True):
            inserted, deduped = stream.consume(max_items=int(batch_size), ingest_fn=store.ingest_batch)
            st.cache_resource.clear()
            st.success(f"Ingested {inserted} · dedup skipped {deduped}")
            st.rerun()
        sm = stream.metrics.to_dict()
        st.caption(
            f"Produced {sm['produced']:,} · ingested {sm['ingested']:,} · deduped {sm['deduped']:,} · "
            f"queue {stream.pending():,}"
        )

        st.markdown("---")
        st.markdown("**RL policy (Thompson sampling)**")
        render_rl_policy(user.rl_agent)
        st.caption("Engage, bookmark, or skip cards to update domain preferences.")
        if llm_configured():
            st.caption("LLM suggestions: Gemini → Groq → OpenAI (first configured free provider wins).")
        else:
            st.caption(
                "Suggestions use smart templates (free). Optional: add a free GEMINI_API_KEY from "
                "aistudio.google.com — no payment required."
            )

    render_hero(
        title="EngageIQ — Engagement Opportunity Scorer",
        subtitle=(
            f"{len(df):,} opportunities · {live_n:,} live GitHub API & GitHub Archive · "
            f"{domain_count} domains · TF-IDF retrieval + multi-stage ranking + RL bandit"
        ),
        stats={
            "Dataset": f"{len(df):,}",
            "Live API": f"{live_n:,}",
            "Domains": f"{domain_count}/15",
            "Saved": str(counts["bookmarked"]),
        },
    )

    if missing_domains:
        st.warning(f"Dataset missing domains: {', '.join(missing_domains)}")

    tab_discover, tab_bookmarks, tab_activity, tab_analytics = st.tabs(
        ["Discover", "Bookmarks", "Activity", "Analytics"]
    )

    ranked = _rank_opportunities(df, user, user.interest_text, version, live_only, english_only)

    domain_tokens = [w.strip() for w in user.interest_text.split(",") if w.strip()]
    labels = [
        1 if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in domain_tokens) else 0
        for i in range(len(ranked))
    ]
    ndcg_val = ndcg_at_k(labels, 10)
    live_in_results = int(live_mask(ranked).sum())
    pool_gh = int((ranked["source"].astype(str).str.lower() == "github").sum())
    pool_gha = int((ranked["source"].astype(str).str.lower() == "gharchive").sum())

    with tab_discover:
        persona_short = persona.split("(")[0].strip()
        match_pct = min(100, max(0, int(round(ndcg_val * 100))))

        st.markdown(f"### Opportunities for **{persona_short}**")
        st.markdown(
            '<div class="discover-help-box">'
            "<strong>How this works:</strong> We search the GitHub API and GitHub Archive public event stream "
            "for items that match what you typed in the sidebar. Each card is a real place you could comment, "
            "contribute, or join a discussion. Use the <strong>Sort &amp; filter</strong> section below to change "
            "what you see — for example, show only GitHub Archive events or the quickest tasks first."
            "</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Interest match",
            f"{match_pct}%",
            help="How well the top results fit your sidebar interests. 100% = strong fit.",
        )
        m2.metric(
            "GitHub API in list",
            pool_gh,
            help="Number of GitHub API repos/issues in the current ranked list (before your filters).",
        )
        m3.metric(
            "GitHub Archive in list",
            pool_gha,
            help="Number of GitHub Archive events in the current ranked list (before your filters).",
        )
        m4.metric(
            "Live from web",
            live_in_results,
            help="Items with real URLs scraped from the internet (not offline practice/backup rows).",
        )

        if live_only and live_n == 0:
            st.warning("No live web data loaded. Check that code/data/live_opportunities.csv exists on the server.")

        st.markdown(
            '<div class="discover-filter-panel"><h4>Sort & filter</h4></div>',
            unsafe_allow_html=True,
        )
        st.caption("Change order or narrow results. Example: pick **GitHub Archive only** or **Quickest to contribute**.")

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        sort_by = r1c1.selectbox(
            "Sort by",
            options=list(SORT_OPTIONS.keys()),
            index=0,
        )
        source_filter = r1c2.selectbox(
            "Platform",
            options=list(SOURCE_FILTER_OPTIONS.keys()),
            help="Show everything, GitHub API search results only, or GitHub Archive events only.",
        )
        origin_filter = r1c3.selectbox(
            "Live or offline",
            options=list(ORIGIN_FILTER_OPTIONS.keys()),
            help="Live = real scraped links. Offline = backup practice rows for grading without internet.",
        )
        max_effort = r1c4.selectbox(
            "Time to start",
            options=["Any time", "Under 1 hour", "Under 2 hours"],
            help="Filter by how long it might take to make your first contribution.",
        )
        effort_map = {"Any time": "Any", "Under 1 hour": "Under 1 hour", "Under 2 hours": "Under 2 hours"}

        r2c1, r2c2, r2c3 = st.columns([1, 2.2, 1])
        show_limit = r2c1.selectbox("Show", options=[10, 20, 40, 80, 100], index=4, format_func=lambda n: f"{n} results")
        domain_filter = r2c2.multiselect(
            "Topic area (optional)",
            options=sorted(ranked["domain"].dropna().astype(str).unique().tolist()),
            default=[],
            placeholder="All topic areas — e.g. Machine Learning, DevOps",
        )
        gfi_only = r2c3.checkbox("Beginner issues only", value=False, help="GitHub “good first issue” labels only.")

        filtered = filter_ranked_results(
            ranked,
            source=source_filter,
            origin=origin_filter,
            domains=domain_filter or None,
            good_first_issue_only=gfi_only,
            max_effort=effort_map[max_effort],
        )
        displayed = sort_ranked_results(filtered, sort_by=sort_by).head(int(show_limit))

        st.markdown(f"**Showing:** {source_mix_summary(displayed)}")

        with st.expander("What do the numbers above mean? (for course graders)"):
            st.markdown(
                f"""
- **Interest match ({match_pct}%)** — Ranking quality metric (NDCG@10 = {ndcg_val:.3f}). Measures how well top results match your interest keywords.
- **GitHub API / GitHub Archive in list** — How many of each source appear in the ranked pool of up to 100 items.
- **Live from web** — Count of items with real API-scraped URLs (vs. offline `example.local` backup rows).
- **Sort & filter** — Client-side view controls; does not re-run the ML model, only re-orders/filters the ranked pool.
                """
            )

        if filtered.empty:
            hint = "Try **All sources**, **All data**, and **Any time**."
            if origin_filter == "Offline practice data" and live_only:
                hint = (
                    "Offline rows are excluded from ranking right now. In the sidebar, turn off "
                    "**Rank from live API pool only**, then set **Live or offline → Offline practice data**."
                )
            elif source_filter == "GitHub Archive only" and pool_gha == 0:
                hint = "Try loading the **Lina** or **David** persona — they surface more GitHub Archive events."
            st.info(f"No results match your filters. {hint}")
        else:
            for rank_idx, (_, row) in enumerate(displayed.iterrows(), start=1):
                render_opportunity_card(
                    rank=rank_idx,
                    row=row,
                    explain=_explain_row(row),
                    suggest=_suggest_action(row, user.interest_text),
                    key_prefix="disc",
                )

    with tab_bookmarks:
        st.markdown("#### Saved opportunities")
        render_bookmarks_list()

    with tab_activity:
        st.markdown("#### Activity log")
        action_filters = {
            "All": None,
            "Engage": "engage",
            "Bookmark": "bookmark",
            "Skip": "skip",
            "Unbookmark": "unbookmark",
        }
        filter_opt = st.radio(
            "Filter",
            list(action_filters.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        render_activity_log(action_filters[filter_opt])
        if st.session_state.get("activity_log"):
            st.download_button(
                "Download activity log (CSV)",
                data=pd.DataFrame(st.session_state.activity_log).to_csv(index=False).encode("utf-8"),
                file_name=f"engageiq_activity_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    with tab_analytics:
        st.markdown("#### Batch analytics & export")
        st.caption("Capability coverage: multi-source ingest, retrieval, ranking, RL, analytics, dashboard export.")

        try:
            trends = compute_trends(str(paths.duckdb_path), days=30)
        except Exception:
            trends = compute_trends_from_df(df, days=30)

        wow = compute_wow_domain_growth(df)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Volume by domain (30d)**")
            if trends.by_domain.empty:
                st.info("No domain volume data in the last 30 days.")
            else:
                dom_chart = _chart(
                    trends.by_domain,
                    lambda c: c.mark_bar(color="#4F46E5"),
                    {
                        "x": alt.X("n:Q", title="Count"),
                        "y": alt.Y("domain:N", sort="-x"),
                        "tooltip": ["domain", "n", "avg_upvotes", "avg_comments"],
                    },
                )
                st.altair_chart(dom_chart, use_container_width=True)

        with col_b:
            st.markdown("**Daily volume**")
            if trends.volume_over_time.empty:
                st.info("No daily volume data in the last 30 days.")
            else:
                vol_chart = _chart(
                    trends.volume_over_time,
                    lambda c: c.mark_line(point=True, color="#4F46E5", strokeWidth=2),
                    {"x": alt.X("day:T"), "y": alt.Y("n:Q", title="Daily volume")},
                )
                st.altair_chart(vol_chart, use_container_width=True)

        st.markdown("**Week-over-week rising domains**")
        if wow.empty:
            st.info("Not enough data for week-over-week comparison.")
        else:
            wow_chart = _chart(
                wow.head(10),
                lambda c: c.mark_bar(color="#0EA5E9"),
                {
                    "x": alt.X("delta:Q", title="Volume change"),
                    "y": alt.Y("domain:N", sort="-x"),
                    "tooltip": ["domain", "this_week", "last_week", "delta", "pct_change"],
                },
            )
            st.altair_chart(wow_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("**RL benchmark (60 rounds vs no-RL baseline)**")
        if st.button("Run RL benchmark", use_container_width=True, key="rl_bench"):
            from engageiq.persona_eval import learning_benchmark

            st.session_state._last_rl_bench = learning_benchmark(df, user.interest_text, rounds=60)
        if st.session_state.get("_last_rl_bench"):
            lb = st.session_state._last_rl_bench
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Reward w/ RL (last 10)", f"{lb['avg_reward_last10_with_rl']:.2f}")
            r2.metric("Reward w/o RL (last 10)", f"{lb['avg_reward_last10_without_rl']:.2f}")
            r3.metric("Session gain", f"{lb.get('session_reward_gain', lb['reward_improvement_last10']):+.2f}")
            r4.metric("NDCG improvement", f"{lb['improvement']:+.3f}")

        st.markdown("---")
        st.markdown("**Brief export**")
        ex1, ex2 = st.columns(2)
        if ex1.button("Export CSV brief", use_container_width=True):
            out = export_brief_csv(
                ranked_df=ranked,
                trends_by_domain=trends.by_domain,
                out_path=paths.data_dir / f"engagement_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                cfg=BriefConfig(top_k=20),
            )
            st.success(f"Exported: {out.name}")
        if ex2.button("Export PDF brief", use_container_width=True):
            out = export_brief_pdf(
                ranked_df=ranked,
                trends_by_domain=trends.by_domain,
                out_path=paths.data_dir / f"engagement_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                persona_name=persona,
                cfg=BriefConfig(top_k=20),
                rising_df=wow,
            )
            st.success(f"Exported: {out.name}")


if __name__ == "__main__":
    main()
