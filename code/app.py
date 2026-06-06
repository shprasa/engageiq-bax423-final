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
    dataset_pool_summary,
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
from engageiq.benchmark_ops import persona_benchmark_row, run_and_write_benchmarks, benchmark_summary
from engageiq.profile_store import (
    delete_custom_profile,
    load_custom_profiles,
    upsert_custom_profile,
    validate_profile,
)
from engageiq.personas import (
    BUILTIN_PROFILES,
    BUILTIN_PERSONAS,
    CUSTOM_SENTINEL,
    PROFILE_FIELD_LABELS,
    PROFILE_FIELDS,
    PROFILE_WIDGET_KEYS,
    UserProfile,
    apply_profile_to_session,
    profile_to_interest_text,
    read_profile_from_session,
)
from engageiq.ranking import RankConfig, augment_candidates, ndcg_at_k, profile_match_pct, rerank
from engageiq.reinforcement_learning import EngagementRLAgent
from engageiq.snapshot_ops import merge_live_refresh
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

PERSONAS = BUILTIN_PERSONAS

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
    return store, {"paths": paths, "version": 18}


@st.cache_resource
def _build_embedding_index(_version: int, corpus_key: str, df: pd.DataFrame):
    return build_index(df)


def _init_user_state(domains: list[str]) -> UserState:
    first = next(iter(BUILTIN_PROFILES))
    return UserState(
        interest_text=BUILTIN_PERSONAS[first],
        liked_texts=[],
        rl_agent=EngagementRLAgent(arms=domains, policy="thompson"),
        rng=np.random.default_rng(42),
    )


def _persona_catalog() -> dict[str, UserProfile]:
    return {**BUILTIN_PROFILES, **load_custom_profiles()}


def _persona_option_labels() -> list[str]:
    custom = list(load_custom_profiles().keys())
    return list(BUILTIN_PROFILES.keys()) + custom + [CUSTOM_SENTINEL]


def _persist_profile_and_benchmark(df: pd.DataFrame, paths, profile_name: str) -> dict:
    payload = run_and_write_benchmarks(df, paths)
    row = persona_benchmark_row(payload, profile_name) or {}
    st.session_state.last_benchmark_payload = payload
    st.session_state.last_profile_save_msg = (
        f"Saved **{profile_name}** to disk. "
        f"Benchmark: interest match **{row.get('profile_match_pct', '—')}%**, "
        f"{'PASS' if row.get('passed') else 'CHECK'} "
        f"({row.get('persona_type', 'custom')} profile)."
    )
    st.cache_resource.clear()
    return payload


def _on_persona_select_change() -> None:
    """Streamlit callback — preset selection must set widget state directly."""
    selected = st.session_state.persona_select
    catalog = _persona_catalog()
    if selected not in catalog:
        return
    apply_profile_to_session(st.session_state, catalog[selected])
    if "user_state" in st.session_state:
        st.session_state.user_state.interest_text = profile_to_interest_text(catalog[selected])
    st.session_state.suggestion_cache = {}


def _ensure_profile_widgets(user: UserState) -> None:
    if "persona_select" not in st.session_state:
        st.session_state.persona_select = _persona_option_labels()[0]
    if not any(st.session_state.get(PROFILE_WIDGET_KEYS[field]) for field in PROFILE_FIELDS):
        first = next(iter(BUILTIN_PROFILES))
        apply_profile_to_session(st.session_state, BUILTIN_PROFILES[first])
        user.interest_text = BUILTIN_PERSONAS[first]


def _sync_user_interest_text(user: UserState) -> str:
    profile = read_profile_from_session(st.session_state)
    user.interest_text = profile_to_interest_text(profile)
    return user.interest_text


def _profile_display_name(selected: str) -> str:
    if selected == CUSTOM_SENTINEL:
        return "Custom profile"
    return selected.split("(")[0].strip()


def _explain_row(row: pd.Series) -> str:
    match = float(row.get("score_match", row.get("score_relevance", 0)))
    parts = [
        f"Match {match:.0%} to your interests",
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
    if action in ("engage", "bookmark", "skip"):
        opp_id = int(row["id"])
        acted = st.session_state.acted_opportunity_ids
        if opp_id not in acted:
            acted.append(opp_id)
    if action in ("engage", "bookmark"):
        snippet = f"{row['domain']}: {display_title(row)}"
        if snippet not in user.liked_texts:
            user.liked_texts.append(snippet)
    st.toast(
        f"RL reward {reward:+.2f} · {action.replace('_', ' ').title()}",
        icon="✅" if reward > 0 else "⏭️",
    )


def _write_snapshot_everywhere(refreshed: pd.DataFrame, paths) -> None:
    live_only = refreshed[live_mask(refreshed)]
    snap_paths = {
        paths.snapshot_csv,
        paths.project_root / "data" / "opportunities_snapshot.csv",
        paths.code_dir / "data" / "opportunities_snapshot.csv",
    }
    live_paths = {
        paths.live_csv,
        paths.code_dir / "data" / "live_opportunities.csv",
    }
    for snap_path in snap_paths:
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        refreshed.to_csv(snap_path, index=False)
    for live_path in live_paths:
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_only.to_csv(live_path, index=False)


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
            st.session_state.acted_opportunity_ids = []
            user.liked_texts = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Profile & interests**")
        _ensure_profile_widgets(user)
        profile_options = _persona_option_labels()
        st.selectbox(
            "Profile preset",
            options=profile_options,
            key="persona_select",
            on_change=_on_persona_select_change,
            help="Built-in personas use the official course profile tables. Choose Custom to write your own.",
        )
        selected_profile = st.session_state.persona_select

        st.caption("Fill out all five fields — ranking uses the full profile, not a single interest line.")
        for field in PROFILE_FIELDS:
            st.text_area(
                PROFILE_FIELD_LABELS[field],
                height=72 if field != "goal" else 88,
                key=PROFILE_WIDGET_KEYS[field],
            )

        interest_text = _sync_user_interest_text(user)

        with st.expander("Save a custom profile"):
            custom_name = st.text_input("Profile name", placeholder="e.g. Security researcher")
            st.caption(
                "All five fields are required. Saving writes `data/custom_personas.json` "
                "and regenerates `data/benchmark_results.json` for every persona."
            )
            if st.session_state.get("last_profile_save_msg"):
                st.success(st.session_state.last_profile_save_msg)
            c_save, c_del = st.columns(2)
            if c_save.button("Save profile & run benchmarks", use_container_width=True, type="primary"):
                name = custom_name.strip()
                profile = read_profile_from_session(st.session_state)
                missing = validate_profile(profile)
                if not name:
                    st.warning("Enter a profile name first.")
                elif name in BUILTIN_PROFILES:
                    st.warning("That name is reserved for a built-in course persona.")
                elif missing:
                    st.warning(f"Complete all fields: {', '.join(missing)}")
                else:
                    try:
                        upsert_custom_profile(name, profile)
                        st.session_state.persona_select = name
                        with st.spinner("Running full benchmarks for all personas (~60s)…"):
                            _persist_profile_and_benchmark(df, paths, name)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Save failed: {exc}")
            if c_del.button("Delete custom profile", use_container_width=True):
                if selected_profile in load_custom_profiles():
                    try:
                        delete_custom_profile(selected_profile)
                        first = profile_options[0]
                        apply_profile_to_session(st.session_state, BUILTIN_PROFILES[first])
                        st.session_state.persona_select = first
                        user.interest_text = BUILTIN_PERSONAS[first]
                        st.session_state.suggestion_cache = {}
                        with st.spinner("Re-running benchmarks after delete (~60s)…"):
                            _persist_profile_and_benchmark(df, paths, first)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
                else:
                    st.info("Select a saved custom profile to delete.")

        st.markdown("---")
        st.markdown("**Data pool for ranking**")
        use_live_snapshot = st.toggle(
            "Use saved live API snapshot",
            value=True,
            help=(
                "ON: rank from opportunities saved during the last live GitHub API + GitHub Archive call "
                f"({live_n:,} rows in the bundled CSV). OFF: rank from the full offline backup dataset "
                f"({len(df):,} rows total, including synthetic practice rows for ≥10k grading)."
            ),
        )
        if not use_live_snapshot:
            st.caption(
                "Backup mode: previously live API rows are treated as saved backup opportunities, "
                "alongside synthetic practice rows (example.local)."
            )
        else:
            st.caption(
                "Snapshot mode: ranking uses real URLs saved from the last API refresh. "
                "Turn off to browse the full offline grading dataset. "
                "**Refresh live API data now** also regenerates persona benchmarks."
            )

        if st.session_state.get("last_api_refresh_msg"):
            st.success(st.session_state.last_api_refresh_msg)

        if st.button("Refresh live API data now", use_container_width=True, type="primary"):
            try:
                with st.spinner("Calling GitHub API + GitHub Archive…"):
                    refreshed, result = merge_live_refresh(paths.snapshot_csv)
                    _write_snapshot_everywhere(refreshed, paths)
                    store._reload_from_csv(paths.snapshot_csv)
                with st.spinner("Re-running persona benchmarks on updated dataset (~60s)…"):
                    updated_df = store.load_df()
                    payload = run_and_write_benchmarks(updated_df, paths)
                    summary = benchmark_summary(payload)
                    st.session_state.last_api_refresh_msg = f"{result.message} {summary}"
                    st.session_state.last_benchmark_payload = payload
                    st.cache_resource.clear()
                    st.rerun()
            except Exception as exc:
                st.error(f"Live refresh failed: {exc}")

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

    ranked = _rank_opportunities(df, user, interest_text, version, use_live_snapshot, english_only)

    domain_tokens = [w.strip() for w in interest_text.split(",") if w.strip()]
    labels = [
        1 if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in domain_tokens) else 0
        for i in range(min(10, len(ranked)))
    ]
    ndcg_val = ndcg_at_k(labels, 10)
    live_in_results = int(live_mask(ranked).sum())
    pool_gh = int((ranked["source"].astype(str).str.lower() == "github").sum())
    pool_gha = int((ranked["source"].astype(str).str.lower() == "gharchive").sum())

    with tab_discover:
        profile_short = _profile_display_name(selected_profile)
        match_pct = profile_match_pct(ranked)
        acted_ids = set(st.session_state.get("acted_opportunity_ids", []))

        st.markdown(f"### Opportunities for **{profile_short}**")
        st.caption(dataset_pool_summary(df, use_live_snapshot_pool=use_live_snapshot))
        st.markdown(
            '<div class="discover-help-box">'
            "<strong>How this works:</strong> Pick a built-in persona or fill in the five profile fields "
            "(background, interests, goal, platforms, time budget) in the sidebar. "
            "Each card is one saved opportunity from the bundled dataset. "
            "<strong>Engage / Bookmark / Skip</strong> logs one action, removes the card, updates RL ranking, "
            "and refreshes the feed. Use <strong>Refresh live API data now</strong> in the sidebar to fetch new rows "
            "(requires <code>GITHUB_TOKEN</code> in <code>code/.env</code>)."
            "</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Interest match",
            f"{match_pct}%",
            help="Average persona-aware fit of your top-10 ranked results (text similarity + profile signals like visibility for Lina or DevOps fit for David).",
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
            "In ranked pool",
            live_in_results if use_live_snapshot else len(ranked),
            help=(
                "Saved live API rows in the current ranked pool when snapshot mode is on; "
                "full backup row count when backup mode is on."
            ),
        )

        if use_live_snapshot and live_n == 0:
            st.warning(
                "No saved live API rows in the CSV. Use **Refresh live API data now** in the sidebar "
                "or turn off **Use saved live API snapshot** to rank the full offline backup."
            )

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
            "Data origin",
            options=list(ORIGIN_FILTER_OPTIONS.keys()),
            help=(
                "Saved from live API = real URLs from the last API refresh. "
                "Synthetic practice rows = example.local backup rows for offline grading."
            ),
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
        pending = filtered[~filtered["id"].astype(int).isin(acted_ids)]
        displayed = sort_ranked_results(pending, sort_by=sort_by).head(int(show_limit))

        acted_note = f" · {len(acted_ids)} acted on this session" if acted_ids else ""
        st.markdown(f"**Showing:** {source_mix_summary(displayed)}{acted_note}")

        with st.expander("What do the numbers above mean? (for course graders)"):
            st.markdown(
                f"""
- **Interest match ({match_pct}%)** — Average persona-aware fit of your top-10 results (embedding similarity plus profile signals such as visibility for trend-spotting or DevOps domain fit). Keyword NDCG@10 = {ndcg_val:.3f}.
- **GitHub API / GitHub Archive in list** — How many of each source appear in the ranked pool of up to 100 items.
- **Data pool** — Snapshot mode ranks saved live API rows; backup mode ranks the full ≥10k offline dataset (including synthetic practice rows).
- **Sort & filter** — Reorders/filters the ranked pool for display.
                """
            )

        if filtered.empty:
            hint = "Try **All sources**, **All opportunities**, and **Any time**."
            if origin_filter == "Synthetic practice rows" and use_live_snapshot:
                hint = (
                    "Synthetic rows are excluded in snapshot mode. Turn off **Use saved live API snapshot** "
                    "in the sidebar, then filter to **Synthetic practice rows**."
                )
            elif source_filter == "GitHub Archive only" and pool_gha == 0:
                hint = "Try the **Lina** or **David** profile — they surface more GitHub Archive events."
            st.info(f"No results match your filters. {hint}")
        else:
            if pending.empty and acted_ids:
                st.info(
                    f"You've acted on all visible opportunities ({len(acted_ids)} this session). "
                    "Change filters, switch profiles, or clear activity in the sidebar to see more cards."
                )
            for rank_idx, (_, row) in enumerate(displayed.iterrows(), start=1):
                render_opportunity_card(
                    rank=rank_idx,
                    row=row,
                    explain=_explain_row(row),
                    suggest=_suggest_action(row, interest_text),
                    key_prefix="disc",
                    use_live_snapshot_pool=use_live_snapshot,
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

            st.session_state._last_rl_bench = learning_benchmark(df, interest_text, rounds=60)
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
                persona_name=selected_profile,
                cfg=BriefConfig(top_k=20),
                rising_df=wow,
            )
            st.success(f"Exported: {out.name}")


if __name__ == "__main__":
    main()
