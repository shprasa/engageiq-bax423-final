from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engageiq.analytics import compute_trends, compute_trends_from_df, compute_wow_domain_growth
from engageiq.bandit import BetaBandit
from engageiq.brief_export import BriefConfig, export_brief_csv, export_brief_pdf
from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.embedding import build_index
from engageiq.ranking import RankConfig, ndcg_at_k, rerank
from engageiq.sketches import BloomFilter, CountMinSketch, HyperLogLog
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
)

PERSONAS: dict[str, str] = {
    "Sofia (ML Student / Portfolio Builder)": (
        "Machine learning, NLP, data pipelines, beginner-friendly open source, good first issues, "
        "Python, pandas, GitHub issues, Reddit ML discussion."
    ),
    "David (DevOps / Niche Community)": (
        "Kubernetes, Terraform, CI/CD, observability, cloud-native infra, high-activity repos, "
        "few contributors, Reddit r/devops and r/kubernetes."
    ),
    "Lina (Data Journalist / Trend Spotter)": (
        "Trending repos, viral discussions, emerging tools, fast-growing communities, recency, velocity, "
        "Hacker News, GitHub trending, Reddit multi-domain."
    ),
    "Raj (Startup Founder / Marketing-Focused)": (
        "Developer tools, APIs, CLI tools, open-source business, B2B SaaS, discussions where devtools are relevant, "
        "Reddit r/programming r/SideProject r/startups."
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
    bandit: BetaBandit
    rng: np.random.Generator


@st.cache_resource
def _load_store_and_seed() -> tuple[OpportunityStore, dict]:
    paths = get_paths()
    store = OpportunityStore(paths.duckdb_path, snapshot_csv=paths.snapshot_csv)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=100000)
    return store, {"paths": paths, "version": 4}


@st.cache_resource
def _build_embedding_index(df: pd.DataFrame):
    return build_index(df)


def _init_user_state(domains: list[str]) -> UserState:
    return UserState(
        interest_text=PERSONAS["Sofia (ML Student / Portfolio Builder)"],
        liked_texts=[],
        bandit=BetaBandit(arms=domains),
        rng=np.random.default_rng(42),
    )


def _explain_row(row: pd.Series) -> str:
    parts = [
        f"Relevance {row['score_relevance']:.2f}",
        f"Health {row['score_health']:.2f}",
        f"Visibility {row['score_visibility']:.2f}",
        f"Effort {row['score_effort']:.2f}",
    ]
    if str(row.get("source", "")) == "github" and int(row.get("good_first_issue") or 0) == 1:
        parts.append("good-first-issue boost")
    return " · ".join(parts)


def _suggest_action(row: pd.Series) -> str:
    src = str(row.get("source", ""))
    dom = str(row.get("domain", ""))
    if src == "github":
        if int(row.get("good_first_issue") or 0) == 1:
            return f"Pick one good-first-issue in {dom}, confirm scope in a comment, then open a small PR."
        return f"Find a docs or test gap in {dom}, propose the fix in an issue, then submit a PR."
    if src == "reddit":
        return f"Write a 5–8 sentence comment in {dom} with a concrete tip, one link, and a question to invite replies."
    return f"Post a short HN-style response in {dom} summarizing trade-offs and one practical takeaway."


def _apply_pending_feedback(user: UserState) -> None:
    pending = st.session_state.pop("_pending_action", None)
    if not pending:
        return
    action: Action
    row_data: dict
    action, row_data = pending
    row = pd.Series(row_data)
    record_feedback(row, action)
    if action in ("engage", "bookmark"):
        user.bandit.update(str(row["domain"]), 1)
        snippet = f"{row['domain']}: {row['title']}"
        if snippet not in user.liked_texts:
            user.liked_texts.append(snippet)
    elif action == "skip":
        user.bandit.update(str(row["domain"]), 0)
    st.toast(f"Recorded: {action.replace('_', ' ').title()}", icon="✅" if action != "skip" else "⏭️")


def _chart(df: pd.DataFrame, mark_fn, encode_kwargs: dict) -> alt.Chart:
    return mark_fn(alt.Chart(df).encode(**encode_kwargs)).properties(height=280).configure(**CHART_THEME)


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
    live_n = int((~df["url"].astype(str).str.contains("example.local", na=False)).sum())

    if "user_state" not in st.session_state:
        st.session_state.user_state = _init_user_state(domains)
    user: UserState = st.session_state.user_state
    _apply_pending_feedback(user)

    if "sketches" not in st.session_state:
        st.session_state.sketches = {
            "bloom": BloomFilter(capacity=20000, fp_rate=0.01),
            "cms_domain": CountMinSketch(width=4096, depth=5),
            "cms_source": CountMinSketch(width=512, depth=5),
            "hll_authors": HyperLogLog(p=12),
        }
        for _, r in df.iterrows():
            st.session_state.sketches["bloom"].add(str(int(r["id"])))
            st.session_state.sketches["cms_domain"].add(str(r["domain"]))
            st.session_state.sketches["cms_source"].add(str(r["source"]))
            st.session_state.sketches["hll_authors"].add(str(r["author"]))

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

        if st.button("Clear activity log", use_container_width=True):
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

        st.markdown("---")
        st.markdown("**Pipeline controls**")
        batch_size = st.slider("Ingest batch size", 50, 2000, 500, 50)
        if st.button("Ingest streaming batch", use_container_width=True):
            snapshot = pd.read_csv(paths.snapshot_csv)
            max_id = store.max_id()
            batch = snapshot[snapshot["id"] > max_id].head(int(batch_size)).copy()
            if batch.empty:
                st.info("Snapshot fully ingested.")
            else:
                inserted = store.ingest_batch(batch)
                for _, r in batch.iterrows():
                    key = str(int(r["id"]))
                    if key in st.session_state.sketches["bloom"]:
                        continue
                    st.session_state.sketches["bloom"].add(key)
                    st.session_state.sketches["cms_domain"].add(str(r["domain"]))
                    st.session_state.sketches["cms_source"].add(str(r["source"]))
                    st.session_state.sketches["hll_authors"].add(str(r["author"]))
                st.success(f"Ingested {inserted} records")
                st.rerun()

        st.markdown("**Sketch metrics**")
        st.caption(f"Ingested: {len(df):,} · Unique authors (HLL): {int(st.session_state.sketches['hll_authors'].count()):,}")

    render_hero(
        title="EngageIQ — Engagement Opportunity Scorer",
        subtitle=(
            f"{len(df):,} opportunities ({live_n:,} live GitHub/HN + synthetic backup) · "
            "Streaming sketches · Embeddings + ANN · Multi-stage ranking · Adaptive learning"
        ),
        stats={
            "Dataset": f"{len(df):,}",
            "Live API": f"{live_n:,}",
            "Domains": str(df["domain"].nunique()),
            "Saved": str(counts["bookmarked"]),
        },
    )

    tab_discover, tab_bookmarks, tab_activity, tab_analytics = st.tabs(
        ["🔍 Discover", "★ Bookmarks", "📋 Activity Log", "📈 Analytics"]
    )

    index = _build_embedding_index(df)
    q = user.interest_text.strip()
    if user.liked_texts:
        q += "\n\nRecently liked:\n" + "\n".join(user.liked_texts[-5:])

    idxs, dists = index.query(q, top_k=RankConfig().candidate_k)
    candidates = df.iloc[idxs].copy().reset_index(drop=True)
    relevance = np.clip(1.0 - np.asarray(dists, dtype=np.float64), 0.0, 1.0)
    ranked = rerank(
        candidates=candidates,
        relevance01=relevance,
        bandit=user.bandit,
        rng=user.rng,
        cfg=RankConfig(),
        interest_text=q,
    )

    domain_tokens = [w.strip() for w in user.interest_text.split(",") if w.strip()]
    labels = [
        1 if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in domain_tokens) else 0
        for i in range(len(ranked))
    ]
    ndcg_val = ndcg_at_k(labels, 10)

    with tab_discover:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"**Personalized top opportunities** for `{persona.split('(')[0].strip()}`")
        c2.metric("NDCG@10", f"{ndcg_val:.3f}")
        c3.metric("Results", len(ranked))

        for i, row in ranked.iterrows():
            render_opportunity_card(
                rank=i + 1,
                row=row,
                explain=_explain_row(row),
                suggest=_suggest_action(row),
                key_prefix="disc",
            )

    with tab_bookmarks:
        st.markdown("#### Saved opportunities")
        st.caption("Bookmarks persist for your session. Engaging an item also saves it here.")
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
                use_container_width=False,
            )

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

        st.markdown("**Adaptive learning simulation (60 rounds)**")
        if ex3.button("Run simulation", use_container_width=True):
            sim_user = _init_user_state(domains)
            sim_user.interest_text = user.interest_text
            ndcgs: list[float] = []
            for _ in range(60):
                s_idxs, s_dists = index.query(sim_user.interest_text, top_k=RankConfig().candidate_k)
                cand = df.iloc[s_idxs].copy().reset_index(drop=True)
                rel = np.clip(1.0 - np.asarray(s_dists), 0.0, 1.0)
                r = rerank(cand, rel, sim_user.bandit, sim_user.rng, RankConfig(), interest_text=sim_user.interest_text)
                probs = [0.55 if rr["domain"].lower() in sim_user.interest_text.lower() else 0.15 for _, rr in r.iterrows()]
                chosen = int(sim_user.rng.integers(0, len(r)))
                reward = 1 if sim_user.rng.random() < probs[chosen] else 0
                sim_user.bandit.update(str(r.loc[chosen, "domain"]), reward)
                sim_labels = [1 if r.loc[i, "domain"].lower() in sim_user.interest_text.lower() else 0 for i in range(len(r))]
                ndcgs.append(ndcg_at_k(sim_labels, 10))

            r1, r2, r3 = st.columns(3)
            r1.metric("NDCG first 10", f"{float(np.mean(ndcgs[:10])):.3f}")
            r2.metric("NDCG last 10", f"{float(np.mean(ndcgs[-10:])):.3f}")
            r3.metric("Improvement", f"{float(np.mean(ndcgs[-10:]) - np.mean(ndcgs[:10])):+.3f}")


if __name__ == "__main__":
    main()
