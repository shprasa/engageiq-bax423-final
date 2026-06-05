from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engageiq.analytics import compute_trends, compute_trends_from_df
from engageiq.bandit import BetaBandit
from engageiq.brief_export import BriefConfig, export_brief_csv, export_brief_pdf
from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.embedding import build_index
from engageiq.ranking import RankConfig, ndcg_at_k, rerank
from engageiq.sketches import BloomFilter, CountMinSketch, HyperLogLog


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
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=10000)
    return store, {"paths": paths, "version": 3}


@st.cache_resource
def _build_embedding_index(df: pd.DataFrame):
    return build_index(df)


def _init_user_state(domains: list[str]) -> UserState:
    rng = np.random.default_rng(42)
    return UserState(
        interest_text=PERSONAS["Sofia (ML Student / Portfolio Builder)"],
        liked_texts=[],
        bandit=BetaBandit(arms=domains),
        rng=rng,
    )


def _explain_row(row: pd.Series) -> str:
    parts = [
        f"Relevance={row['score_relevance']:.2f}",
        f"Health={row['score_health']:.2f}",
        f"Visibility={row['score_visibility']:.2f}",
        f"Effort={row['score_effort']:.2f}",
    ]
    if str(row.get("source", "")) == "github" and int(row.get("good_first_issue") or 0) == 1:
        parts.append("Good-first-issue boost")
    return " · ".join(parts)


def _suggest_action(row: pd.Series) -> str:
    src = str(row.get("source", ""))
    dom = str(row.get("domain", ""))
    if src == "github":
        if int(row.get("good_first_issue") or 0) == 1:
            return f"Suggested action: pick 1 'good first issue' in **{dom}**, comment to confirm scope, then open a small PR."
        return f"Suggested action: identify a docs/test gap in **{dom}**, open an issue proposing the fix, then submit a PR."
    if src == "reddit":
        return f"Suggested action: write a 5–8 sentence comment in **{dom}** with (a) a concrete tip, (b) one link, (c) a question to invite replies."
    return f"Suggested action: post a short HN-style response in **{dom}** summarizing trade-offs + one practical takeaway."


def main() -> None:
    st.set_page_config(page_title="EngageIQ Prototype", layout="wide")

    store, ctx = _load_store_and_seed()
    paths = ctx["paths"]

    st.title("EngageIQ — Engagement Opportunity Scorer (Prototype)")
    st.caption(
        "Offline-first prototype: streaming ingest (simulated), sketching (Bloom/CMS/HLL), embeddings+ANN retrieval, "
        "multi-stage ranking, online learning from feedback, and batch trend analytics."
    )

    # Load current ingested dataset
    df = store.load_df()
    domains = sorted(df["domain"].dropna().unique().tolist())

    if "user_state" not in st.session_state:
        st.session_state.user_state = _init_user_state(domains)
    user: UserState = st.session_state.user_state

    # sketches live in session (represents streaming pipeline state)
    if "sketches" not in st.session_state:
        st.session_state.sketches = {
            "bloom": BloomFilter(capacity=20000, fp_rate=0.01),
            "cms_domain": CountMinSketch(width=4096, depth=5),
            "cms_source": CountMinSketch(width=512, depth=5),
            "hll_authors": HyperLogLog(p=12),
        }

        # seed sketches with current ingested data
        for _, r in df.iterrows():
            st.session_state.sketches["bloom"].add(str(int(r["id"])))
            st.session_state.sketches["cms_domain"].add(str(r["domain"]))
            st.session_state.sketches["cms_source"].add(str(r["source"]))
            st.session_state.sketches["hll_authors"].add(str(r["author"]))

    colL, colR = st.columns([0.45, 0.55], gap="large")

    with colL:
        st.subheader("Profile")
        persona = st.selectbox("Choose a test persona", options=list(PERSONAS.keys()))
        if st.button("Load persona profile"):
            user.interest_text = PERSONAS[persona]
            user.liked_texts = []

        user.interest_text = st.text_area(
            "Interest profile (used for embedding retrieval)",
            value=user.interest_text,
            height=140,
        )

        st.subheader("Streaming ingest (simulated)")
        batch_size = st.slider("Batch size", min_value=50, max_value=2000, value=500, step=50)
        if st.button("Ingest next streaming batch"):
            snapshot = pd.read_csv(paths.snapshot_csv)
            max_id = store.max_id()
            batch = snapshot[snapshot["id"] > max_id].head(int(batch_size)).copy()
            if batch.empty:
                st.info("No more records to ingest (already at end of snapshot).")
            else:
                inserted = store.ingest_batch(batch)
                # update sketches
                for _, r in batch.iterrows():
                    key = str(int(r["id"]))
                    if key in st.session_state.sketches["bloom"]:
                        continue
                    st.session_state.sketches["bloom"].add(key)
                    st.session_state.sketches["cms_domain"].add(str(r["domain"]))
                    st.session_state.sketches["cms_source"].add(str(r["source"]))
                    st.session_state.sketches["hll_authors"].add(str(r["author"]))
                st.success(f"Ingested {inserted} new items.")
                st.rerun()

        st.subheader("Sketch metrics (from streaming pipeline state)")
        st.write(
            {
                "ingested_records": int(store.count()),
                "approx_unique_authors_hll": int(st.session_state.sketches["hll_authors"].count()),
            }
        )
        top_domain_counts = [
            (d, st.session_state.sketches["cms_domain"].estimate(d)) for d in domains
        ]
        top_domain_counts = sorted(top_domain_counts, key=lambda x: x[1], reverse=True)[:8]
        st.dataframe(pd.DataFrame(top_domain_counts, columns=["domain", "cms_count_estimate"]), use_container_width=True)

    with colR:
        st.subheader("Ranked opportunities")

        # Re-load df after ingest
        df = store.load_df()

        # Build / reuse embedding index
        index = _build_embedding_index(df)
        q = user.interest_text.strip()
        if user.liked_texts:
            q += "\n\nRecently liked:\n" + "\n".join(user.liked_texts[-5:])

        idxs, dists = index.query(q, top_k=RankConfig().candidate_k)
        candidates = df.iloc[idxs].copy().reset_index(drop=True)

        # Convert cosine distance to (0..1) relevance
        relevance = 1.0 - np.asarray(dists, dtype=np.float64)
        relevance = np.clip(relevance, 0.0, 1.0)

        ranked = rerank(
            candidates=candidates,
            relevance01=relevance,
            bandit=user.bandit,
            rng=user.rng,
            cfg=RankConfig(),
            interest_text=q,
        )

        # Lightweight offline metric: treat domain match as "relevance label"
        domain_interest = set(
            [w.strip() for w in user.interest_text.split(",") if w.strip()]
        )
        labels = [
            1
            if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in domain_interest)
            else 0
            for i in range(len(ranked))
        ]
        st.caption(f"Reported metric (proxy): NDCG@10 = **{ndcg_at_k(labels, 10):.3f}**")

        for i, row in ranked.iterrows():
            with st.container(border=True):
                st.markdown(f"**{i+1}. {row['title']}**")
                st.write(row["url"])
                st.write(f"Source: {row['source']} · Domain: {row['domain']} · Community: {row['community']}")
                st.write(_explain_row(row))
                st.write(_suggest_action(row))

                fb_cols = st.columns([0.18, 0.18, 0.18, 0.46])
                if fb_cols[0].button("Engage", key=f"engage_{int(row['id'])}"):
                    user.bandit.update(str(row["domain"]), 1)
                    user.liked_texts.append(f"{row['domain']}: {row['title']} {row['text']}")
                    st.success("Feedback recorded: Engage")
                    st.rerun()
                if fb_cols[1].button("Bookmark", key=f"bookmark_{int(row['id'])}"):
                    user.bandit.update(str(row["domain"]), 1)
                    user.liked_texts.append(f"{row['domain']}: {row['title']} {row['text']}")
                    st.success("Feedback recorded: Bookmark")
                    st.rerun()
                if fb_cols[2].button("Skip", key=f"skip_{int(row['id'])}"):
                    user.bandit.update(str(row["domain"]), 0)
                    st.info("Feedback recorded: Skip")
                    st.rerun()

        st.subheader("Batch analytics & trend detection")
        try:
            trends = compute_trends(str(paths.duckdb_path), days=30)
        except Exception:
            trends = compute_trends_from_df(df, days=30)

        dom_chart = (
            alt.Chart(trends.by_domain)
            .mark_bar()
            .encode(
                x=alt.X("n:Q", title="Opportunities (last 30d)"),
                y=alt.Y("domain:N", sort="-x"),
                tooltip=["domain", "n", "avg_upvotes", "avg_comments"],
            )
        )
        st.altair_chart(dom_chart, use_container_width=True)

        vol_chart = (
            alt.Chart(trends.volume_over_time)
            .mark_line(point=True)
            .encode(x=alt.X("day:T"), y=alt.Y("n:Q", title="Daily volume"))
        )
        st.altair_chart(vol_chart, use_container_width=True)

        st.subheader("Export weekly engagement brief")
        c1, c2 = st.columns(2)
        if c1.button("Export brief (CSV)"):
            out = export_brief_csv(
                ranked_df=ranked,
                trends_by_domain=trends.by_domain,
                out_path=paths.data_dir / f"engagement_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                cfg=BriefConfig(top_k=20),
            )
            st.success(f"Exported: {out}")
        if c2.button("Export brief (PDF)"):
            out = export_brief_pdf(
                ranked_df=ranked,
                trends_by_domain=trends.by_domain,
                out_path=paths.data_dir / f"engagement_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                persona_name=persona,
                cfg=BriefConfig(top_k=20),
            )
            st.success(f"Exported: {out}")

        st.subheader("Adaptive learning benchmark (simulated)")
        st.caption("Runs 60 simulated feedback rounds and reports improvement in proxy NDCG@10.")
        if st.button("Run simulation"):
            sim_user = _init_user_state(domains)
            sim_user.interest_text = user.interest_text
            ndcgs = []

            for _ in range(60):
                idxs, dists = index.query(sim_user.interest_text, top_k=RankConfig().candidate_k)
                cand = df.iloc[idxs].copy().reset_index(drop=True)
                rel = np.clip(1.0 - np.asarray(dists), 0.0, 1.0)
                r = rerank(cand, rel, sim_user.bandit, sim_user.rng, RankConfig(), interest_text=sim_user.interest_text)

                # Simulate reward: if domain tokens appear in profile, engage probability higher
                probs = []
                for _, rr in r.iterrows():
                    p = 0.15
                    if rr["domain"].lower() in sim_user.interest_text.lower():
                        p = 0.55
                    probs.append(p)
                probs = np.asarray(probs)
                chosen = int(sim_user.rng.integers(0, len(r)))
                reward = 1 if sim_user.rng.random() < probs[chosen] else 0
                sim_user.bandit.update(str(r.loc[chosen, "domain"]), reward)

                labels = [
                    1 if r.loc[i, "domain"].lower() in sim_user.interest_text.lower() else 0
                    for i in range(len(r))
                ]
                ndcgs.append(ndcg_at_k(labels, 10))

            st.write(
                {
                    "ndcg@10_first10_avg": float(np.mean(ndcgs[:10])),
                    "ndcg@10_last10_avg": float(np.mean(ndcgs[-10:])),
                    "improvement": float(np.mean(ndcgs[-10:]) - np.mean(ndcgs[:10])),
                }
            )


if __name__ == "__main__":
    main()

