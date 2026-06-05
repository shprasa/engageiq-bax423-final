from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engageiq.analytics import compute_trends, compute_trends_from_df, compute_wow_domain_growth
from engageiq.reinforcement_learning import EngagementRLAgent
from engageiq.brief_export import BriefConfig, export_brief_csv, export_brief_pdf
from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.data_utils import _safe_int, display_title, is_live_url, live_mask, ranking_corpus
from engageiq.embedding import build_index
from engageiq.ranking import RankConfig, augment_candidates, ndcg_at_k, rerank
from engageiq.secrets import openai_configured
from engageiq.sketches import BloomFilter, CountMinSketch, HyperLogLog
from engageiq.streaming import OpportunityStream, try_kafka_publish
from engageiq.suggestions import generate_suggestion
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
        "Python, pandas, GitHub issues, Hacker News ML threads."
    ),
    "David (DevOps / Niche Community)": (
        "Kubernetes, Terraform, CI/CD, observability, cloud-native infra, high-activity repos, "
        "few contributors, Hacker News infra threads."
    ),
    "Lina (Data Journalist / Trend Spotter)": (
        "Trending repos, viral discussions, emerging tools, fast-growing communities, recency, velocity, "
        "Hacker News, GitHub trending, multi-domain velocity."
    ),
    "Raj (Startup Founder / Marketing-Focused)": (
        "Developer tools, APIs, CLI tools, open-source business, B2B SaaS, discussions where devtools are relevant, "
        "Hacker News and GitHub developer-tools communities."
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
    return store, {"paths": paths, "version": 8}


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
        f"Relevance {row['score_relevance']:.2f}",
        f"Health {row['score_health']:.2f}",
        f"Visibility {row['score_visibility']:.2f}",
        f"Effort {row['score_effort']:.2f}",
    ]
    if is_live_url(str(row.get("url", ""))):
        parts.append("live API boost")
    if str(row.get("source", "")) == "github" and _safe_int(row.get("good_first_issue")) == 1:
        parts.append("good-first-issue boost")
    return " · ".join(parts)


def _suggest_action(row: pd.Series, interest: str) -> str:
    if "suggestion_cache" not in st.session_state:
        st.session_state.suggestion_cache = {}
    suggestion = generate_suggestion(row, interest, cache=st.session_state.suggestion_cache)
    if openai_configured():
        return f"[AI] {suggestion}"
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
    return rerank(
        candidates=candidates,
        relevance01=relevance,
        bandit=user.rl_agent,
        rng=user.rng,
        cfg=RankConfig(),
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

    if "user_state" not in st.session_state:
        st.session_state.user_state = _init_user_state(domains)
    user: UserState = st.session_state.user_state
    _apply_pending_feedback(user)

    if "sketches" not in st.session_state or st.session_state.get("sketch_rows") != len(df):
        st.session_state.sketches = {
            "bloom": BloomFilter(capacity=max(25000, len(df)), fp_rate=0.01),
            "cms_domain": CountMinSketch(width=4096, depth=5),
            "cms_source": CountMinSketch(width=512, depth=5),
            "hll_authors": HyperLogLog(p=12),
        }
        for _, r in df.iterrows():
            st.session_state.sketches["bloom"].add(str(int(r["id"])))
            st.session_state.sketches["cms_domain"].add(str(r["domain"]))
            st.session_state.sketches["cms_source"].add(str(r["source"]))
            st.session_state.sketches["hll_authors"].add(str(r["author"]))
        st.session_state.sketch_rows = len(df)

    if "opportunity_stream" not in st.session_state:
        st.session_state.opportunity_stream = OpportunityStream(bloom=st.session_state.sketches["bloom"])

    counts = activity_counts()

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">EngageIQ</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-tagline">Smart engagement opportunity scorer</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**Your session**")
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
        st.markdown("**Test persona**")
        persona = st.selectbox("Profile preset", options=list(PERSONAS.keys()), label_visibility="collapsed")
        if st.button("Load persona", use_container_width=True):
            user.interest_text = PERSONAS[persona]
            st.rerun()

        user.interest_text = st.text_area(
            "Interest profile",
            value=user.interest_text,
            height=120,
            help="Used for embedding retrieval and ranking personalization.",
        )

        live_only = st.toggle(
            "Show live API opportunities only",
            value=True,
            help="When on, rankings use real GitHub/HN URLs. Turn off to include offline backup records.",
        )
        english_only = st.toggle(
            "English only",
            value=True,
            help="Hide opportunities whose title and description appear to be non-English.",
        )

        st.markdown("**Reinforcement learning**")
        render_rl_policy(user.rl_agent)

        st.markdown("---")
        st.markdown("**Streaming pipeline**")
        stream: OpportunityStream = st.session_state.opportunity_stream
        use_kafka = st.checkbox("Try Kafka publish (localhost:9092)", value=False)
        batch_size = st.slider("Stream batch size", 50, 2000, 500, 50)
        c_prod, c_cons = st.columns(2)
        if c_prod.button("1 · Produce batch", use_container_width=True):
            snapshot = pd.read_csv(paths.snapshot_csv)
            max_id = store.max_id()
            batch = snapshot[snapshot["id"] > max_id].head(int(batch_size)).copy()
            if batch.empty:
                batch = snapshot.sample(n=min(int(batch_size), len(snapshot)), random_state=42)
            n = stream.produce_rows(batch)
            if use_kafka:
                sent = try_kafka_publish(batch.to_dict(orient="records"))
                stream.metrics.kafka_sent += sent
                if sent:
                    st.caption(f"Kafka: published {sent} records to engageiq.opportunities")
            st.success(f"Queued {n} records ({stream.pending()} pending)")
        if c_cons.button("2 · Consume (dedup → store)", use_container_width=True):
            def _ingest(batch: pd.DataFrame) -> int:
                return store.ingest_batch(batch)

            def _sketch_hook(batch: pd.DataFrame) -> None:
                for _, r in batch.iterrows():
                    st.session_state.sketches["cms_domain"].add(str(r["domain"]))
                    st.session_state.sketches["cms_source"].add(str(r["source"]))
                    st.session_state.sketches["hll_authors"].add(str(r["author"]))

            inserted, deduped = stream.consume(
                max_items=int(batch_size),
                ingest_fn=_ingest,
                sketch_hooks=[_sketch_hook],
            )
            st.cache_resource.clear()
            st.success(f"Ingested {inserted} · dedup skipped {deduped}")
            st.rerun()

        sm = stream.metrics.to_dict()
        st.caption(
            f"Stream: produced {sm['produced']:,} · ingested {sm['ingested']:,} · "
            f"deduped {sm['deduped']:,} · queue {stream.pending():,}"
        )

        st.markdown("**Sketch metrics**")
        st.caption(
            f"Ingested: {len(df):,} · Live: {live_n:,} · "
            f"Unique authors (HLL): {int(st.session_state.sketches['hll_authors'].count()):,}"
        )

    render_hero(
        title="EngageIQ — Engagement Opportunity Scorer",
        subtitle=(
            f"{len(df):,} opportunities in store · {live_n:,} from live GitHub & Hacker News APIs · "
            "Kafka-ready streaming · Bloom dedup · Embeddings + ANN · RL bandit"
        ),
        stats={
            "Dataset": f"{len(df):,}",
            "Live API": f"{live_n:,}",
            "Domains": str(df["domain"].nunique()),
            "Saved": str(counts["bookmarked"]),
        },
    )

    tab_discover, tab_bookmarks, tab_activity, tab_rl, tab_analytics = st.tabs(
        ["🔍 Discover", "★ Bookmarks", "📋 Activity Log", "🧠 RL Policy", "📈 Analytics"]
    )

    ranked = _rank_opportunities(df, user, user.interest_text, version, live_only, english_only)

    domain_tokens = [w.strip() for w in user.interest_text.split(",") if w.strip()]
    labels = [
        1 if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in domain_tokens) else 0
        for i in range(len(ranked))
    ]
    ndcg_val = ndcg_at_k(labels, 10)
    live_in_results = int(live_mask(ranked).sum())

    with tab_discover:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        c1.markdown(f"**Personalized opportunities** for `{persona.split('(')[0].strip()}`")
        c2.metric("NDCG@10", f"{ndcg_val:.3f}")
        c3.metric("Live in top-20", live_in_results)
        showing = []
        if live_only:
            showing.append("Live only")
        else:
            showing.append("All data")
        if english_only:
            showing.append("English")
        c4.metric("Showing", " · ".join(showing))

        if live_only and live_n == 0:
            st.warning("No live API rows loaded. Data file missing on server — check code/data/live_opportunities.csv.")
        elif live_only:
            st.caption(
                "Real GitHub repos and Hacker News threads. Summaries are plain text; "
                "toggle off English only in the sidebar to include other languages."
            )

        for i, row in ranked.iterrows():
            render_opportunity_card(
                rank=i + 1,
                row=row,
                explain=_explain_row(row),
                suggest=_suggest_action(row, user.interest_text),
                key_prefix="disc",
            )

    with tab_bookmarks:
        st.markdown("#### Saved opportunities")
        st.caption("Bookmarks and engagements persist for your session.")
        render_bookmarks_list()

    with tab_activity:
        st.markdown("#### Engagement activity log")
        filter_opt = st.radio(
            "Filter",
            ["All", "Engage", "Bookmark", "Skip", "Unbookmark"],
            horizontal=True,
            label_visibility="collapsed",
        )
        render_activity_log(None if filter_opt == "All" else filter_opt)
        if st.session_state.get("activity_log"):
            st.download_button(
                "Download activity log (CSV)",
                data=pd.DataFrame(st.session_state.activity_log).to_csv(index=False).encode("utf-8"),
                file_name=f"engageiq_activity_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    with tab_rl:
        st.markdown("#### Reinforcement learning from feedback")
        st.caption(
            "EngageIQ uses a **contextual multi-armed bandit** (Thompson sampling) to learn which domains "
            "you prefer. Actions: engage (+1.0), bookmark (+0.85), skip (0.0)."
        )
        render_rl_policy(user.rl_agent)

        if user.rl_agent.reward_history:
            hist = pd.DataFrame(user.rl_agent.reward_history)
            st.markdown("**Reward history (this session)**")
            cum_chart = (
                alt.Chart(hist)
                .mark_line(point=True, color="#10B981")
                .encode(x="round:Q", y="cumulative_reward:Q", tooltip=["round", "reward", "arm"])
                .properties(height=240)
            )
            st.altair_chart(cum_chart, use_container_width=True)

        from engageiq.persona_eval import learning_benchmark

        st.markdown("**60-round benchmark (RL vs random domain exploration)**")
        if st.button("Run RL benchmark", use_container_width=True, key="rl_bench_tab"):
            lb = learning_benchmark(df, user.interest_text, rounds=60)
            st.session_state._last_rl_bench = lb
        if st.session_state.get("_last_rl_bench"):
            lb = st.session_state._last_rl_bench
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Reward w/ RL (last 10)", f"{lb['avg_reward_last10_with_rl']:.2f}")
            r2.metric("Reward w/o RL (last 10)", f"{lb['avg_reward_last10_without_rl']:.2f}")
            r3.metric("Session gain", f"{lb.get('session_reward_gain', lb['reward_improvement_last10']):+.2f}")
            r4.metric("NDCG improvement", f"{lb['improvement']:+.3f}")

    with tab_analytics:
        try:
            trends = compute_trends(str(paths.duckdb_path), days=30)
        except Exception:
            trends = compute_trends_from_df(df, days=30)

        wow = compute_wow_domain_growth(df)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Opportunity volume by domain (30d)**")
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
            st.markdown("**Daily ingestion volume**")
            vol_chart = _chart(
                trends.volume_over_time,
                lambda c: c.mark_line(point=True, color="#4F46E5", strokeWidth=2),
                {"x": alt.X("day:T"), "y": alt.Y("n:Q", title="Daily volume")},
            )
            st.altair_chart(vol_chart, use_container_width=True)

        st.markdown("**Week-over-week rising domains**")
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
        st.markdown("**Export weekly brief**")
        ex1, ex2, ex3 = st.columns(3)
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

        st.markdown("**RL simulation benchmark (60 rounds · Thompson sampling vs no-RL)**")
        if ex3.button("Run RL simulation", use_container_width=True):
            from engageiq.persona_eval import learning_benchmark

            lb = learning_benchmark(df, user.interest_text, rounds=60)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("NDCG w/ RL (last 10)", f"{lb['ndcg@10_last10_avg']:.3f}")
            r2.metric("NDCG w/o RL (last 10)", f"{lb['ndcg@10_without_rl_last10']:.3f}")
            r3.metric("Reward Δ (last 10)", f"{lb['reward_improvement_last10']:+.3f}")
            r4.metric("Cumulative reward", f"{lb['cumulative_reward_with_rl']:.0f}")
            st.caption(
                f"Policy: {lb['policy']} · formulation: {lb['rl_formulation']} · "
                f"entropy={lb['policy_entropy']:.2f}"
            )


if __name__ == "__main__":
    main()
