from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import altair as alt
import pandas as pd
import streamlit as st

from engageiq.data_utils import describe_opportunity, escape_html, format_created, is_live_url, opportunity_meta

Action = Literal["engage", "bookmark", "skip", "unbookmark"]

SOURCE_COLORS = {
    "github": ("#238636", "GitHub"),
    "hackernews": ("#FF6600", "Hacker News"),
    "reddit": ("#FF4500", "Reddit"),
}

BRAND = {
    "primary": "#4F46E5",
    "primary_dark": "#3730A3",
    "accent": "#0EA5E9",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#64748B",
    "surface": "#FFFFFF",
    "bg": "#F1F5F9",
    "border": "#E2E8F0",
}


def inject_theme() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.block-container {{
    padding-top: 1.25rem;
    max-width: 1400px;
}}

.hero-banner {{
    background: linear-gradient(135deg, {BRAND["primary"]} 0%, {BRAND["primary_dark"]} 55%, #1E1B4B 100%);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    color: white;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.25);
}}

.hero-banner h1 {{
    color: white !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.35rem 0 !important;
    letter-spacing: -0.02em;
}}

.hero-sub {{
    color: rgba(255,255,255,0.88);
    font-size: 0.92rem;
    line-height: 1.5;
    margin: 0;
}}

.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.65rem;
    margin: 1rem 0 0.25rem 0;
}}

.stat-card {{
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    padding: 0.65rem 0.75rem;
    backdrop-filter: blur(4px);
}}

.stat-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.85;
}}

.stat-value {{
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.15rem;
}}

.section-card {{
    background: {BRAND["surface"]};
    border: 1px solid {BRAND["border"]};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}}

.section-title {{
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {BRAND["muted"]};
    margin-bottom: 0.65rem;
}}

.opp-card {{
    background: {BRAND["surface"]};
    border: 1px solid {BRAND["border"]};
    border-left: 4px solid {BRAND["primary"]};
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    transition: box-shadow 0.15s ease;
}}

.opp-card:hover {{
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.1);
}}

.opp-card.bookmarked {{
    border-left-color: {BRAND["warning"]};
    background: linear-gradient(90deg, #FFFBEB 0%, #FFFFFF 12%);
}}

.opp-rank {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: {BRAND["primary"]};
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.5rem;
}}

.badge {{
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
}}

.badge-source {{ color: white; }}
.badge-domain {{
    background: #EEF2FF;
    color: {BRAND["primary_dark"]};
}}
.badge-bookmark {{
    background: #FEF3C7;
    color: #92400E;
}}
.badge-engage {{
    background: #D1FAE5;
    color: #065F46;
}}
.badge-live {{
    background: #DCFCE7;
    color: #166534;
}}
.badge-backup {{
    background: #F1F5F9;
    color: #475569;
}}

.opp-desc {{
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.55;
    margin: 0.5rem 0;
    padding: 0.55rem 0.7rem;
    background: #F8FAFC;
    border-radius: 8px;
    border-left: 3px solid {BRAND["accent"]};
}}

.opp-meta {{
    font-size: 0.78rem;
    color: {BRAND["muted"]};
    margin: 0.35rem 0 0.5rem 0;
}}

.opp-link {{
    font-size: 0.82rem;
    color: {BRAND["primary"]};
    text-decoration: none;
    word-break: break-all;
}}

.score-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.55rem 0;
}}

.score-pill {{
    background: #F8FAFC;
    border: 1px solid {BRAND["border"]};
    border-radius: 8px;
    padding: 0.25rem 0.55rem;
    font-size: 0.75rem;
    color: #334155;
}}

.score-pill strong {{
    color: {BRAND["primary"]};
}}

.suggest-box {{
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
    font-size: 0.85rem;
    color: #0C4A6E;
    margin-top: 0.5rem;
}}

.activity-row {{
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid {BRAND["border"]};
}}

.activity-time {{
    font-size: 0.72rem;
    color: {BRAND["muted"]};
    min-width: 110px;
}}

.activity-body {{
    flex: 1;
}}

.activity-title {{
    font-weight: 600;
    color: #0F172A;
    font-size: 0.9rem;
}}

.empty-state {{
    text-align: center;
    padding: 2.5rem 1rem;
    color: {BRAND["muted"]};
    background: #F8FAFC;
    border-radius: 12px;
    border: 1px dashed {BRAND["border"]};
}}

.sidebar-brand {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {BRAND["primary"]};
    margin-bottom: 0.15rem;
}}

.sidebar-tagline {{
    font-size: 0.78rem;
    color: {BRAND["muted"]};
    margin-bottom: 1rem;
}}

div[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FAFBFF 0%, #F1F5F9 100%);
    border-right: 1px solid {BRAND["border"]};
}}

div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid {BRAND["border"]};
    border-radius: 10px;
    padding: 0.5rem 0.65rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.35rem;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, stats: dict[str, str | int]) -> None:
    stat_html = "".join(
        f'<div class="stat-card"><div class="stat-label">{k}</div><div class="stat-value">{v}</div></div>'
        for k, v in stats.items()
    )
    st.markdown(
        f"""
<div class="hero-banner">
  <h1>{title}</h1>
  <p class="hero-sub">{subtitle}</p>
  <div class="stat-grid">{stat_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def source_badge(source: str) -> str:
    color, label = SOURCE_COLORS.get(str(source).lower(), (BRAND["muted"], str(source)))
    return f'<span class="badge badge-source" style="background:{color};">{label}</span>'


def action_badge(action: str) -> str:
    cls = {"engage": "badge-engage", "bookmark": "badge-bookmark", "skip": "badge-skip", "unbookmark": "badge-skip"}.get(
        action, "badge-domain"
    )
    label = action.replace("_", " ").title()
    return f'<span class="badge {cls}">{label}</span>'


def score_pills(row: pd.Series) -> str:
    pills = [
        ("Relevance", row.get("score_relevance", 0)),
        ("Health", row.get("score_health", 0)),
        ("Visibility", row.get("score_visibility", 0)),
        ("Effort", row.get("score_effort", 0)),
    ]
    html = "".join(
        f'<span class="score-pill"><strong>{name}</strong> {float(val):.2f}</span>' for name, val in pills
    )
    if str(row.get("source", "")) == "github" and int(row.get("good_first_issue") or 0) == 1:
        html += '<span class="score-pill" style="background:#ECFDF5;border-color:#A7F3D0;"><strong>GFI</strong> ✓</span>'
    return f'<div class="score-row">{html}</div>'


@dataclass
class ActivityEntry:
    opp_id: int
    action: Action
    title: str
    url: str
    domain: str
    source: str
    score_final: float
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "opp_id": self.opp_id,
            "action": self.action,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "source": self.source,
            "score_final": self.score_final,
            "timestamp": self.timestamp,
        }


@dataclass
class BookmarkEntry:
    opp_id: int
    title: str
    url: str
    domain: str
    source: str
    score_final: float
    description: str = ""
    bookmarked_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "opp_id": self.opp_id,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "source": self.source,
            "score_final": self.score_final,
            "description": self.description,
            "bookmarked_at": self.bookmarked_at,
        }


def init_activity_state() -> None:
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks: dict[int, dict] = {}
    if "activity_log" not in st.session_state:
        st.session_state.activity_log: list[dict] = []


def log_activity(entry: ActivityEntry) -> None:
    st.session_state.activity_log.insert(0, entry.to_dict())


def record_feedback(row: pd.Series, action: Action) -> None:
    opp_id = int(row["id"])
    desc = describe_opportunity(row)
    entry = ActivityEntry(
        opp_id=opp_id,
        action=action,
        title=str(row["title"]),
        url=str(row["url"]),
        domain=str(row["domain"]),
        source=str(row["source"]),
        score_final=float(row.get("score_final", 0)),
    )
    log_activity(entry)

    bm = BookmarkEntry(
        opp_id=opp_id,
        title=str(row["title"]),
        url=str(row["url"]),
        domain=str(row["domain"]),
        source=str(row["source"]),
        score_final=float(row.get("score_final", 0)),
        description=desc,
    )
    if action == "bookmark":
        st.session_state.bookmarks[opp_id] = bm.to_dict()
    elif action == "unbookmark":
        st.session_state.bookmarks.pop(opp_id, None)
    elif action == "engage":
        st.session_state.bookmarks[opp_id] = bm.to_dict()


def is_bookmarked(opp_id: int) -> bool:
    return int(opp_id) in st.session_state.get("bookmarks", {})


def activity_counts() -> dict[str, int]:
    log = st.session_state.get("activity_log", [])
    return {
        "engaged": sum(1 for e in log if e["action"] == "engage"),
        "bookmarked": len(st.session_state.get("bookmarks", {})),
        "skipped": sum(1 for e in log if e["action"] == "skip"),
        "total_actions": len(log),
    }


def origin_badge(url: str) -> str:
    if is_live_url(url):
        return '<span class="badge badge-live">● Live API</span>'
    return '<span class="badge badge-backup">Offline backup</span>'


def render_opportunity_card(
    rank: int,
    row: pd.Series,
    explain: str,
    suggest: str,
    key_prefix: str,
    show_actions: bool = True,
) -> None:
    bookmarked = is_bookmarked(int(row["id"]))
    card_cls = "opp-card bookmarked" if bookmarked else "opp-card"
    bookmark_badge = '<span class="badge badge-bookmark">★ Saved</span>' if bookmarked else ""

    title = escape_html(row["title"])
    url = str(row["url"])
    url_safe = escape_html(url)
    domain = escape_html(row["domain"])
    description = escape_html(describe_opportunity(row))
    meta = escape_html(opportunity_meta(row))
    explain_safe = escape_html(explain)
    suggest_safe = escape_html(suggest)

    st.markdown(
        f"""
<div class="{card_cls}">
  <div style="display:flex;align-items:flex-start;gap:0.5rem;">
    <span class="opp-rank">{rank}</span>
    <div style="flex:1;">
      <div style="font-weight:700;font-size:1.05rem;color:#0F172A;margin-bottom:0.35rem;">{title}</div>
      <div style="margin-bottom:0.35rem;">
        {source_badge(str(row["source"]))}
        {origin_badge(url)}
        <span class="badge badge-domain">{domain}</span>
        {bookmark_badge}
      </div>
      <div class="opp-meta">{meta}</div>
      <div class="opp-desc">{description}</div>
      <a class="opp-link" href="{url_safe}" target="_blank" rel="noopener">{url_safe}</a>
      {score_pills(row)}
      <div style="font-size:0.8rem;color:#475569;margin-top:0.45rem;"><strong>Why ranked here:</strong> {explain_safe}</div>
      <div class="suggest-box"><strong>Suggested action:</strong> {suggest_safe}</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if show_actions:
        fb = st.columns([0.14, 0.14, 0.14, 0.18, 0.40])
        row_dict = row.to_dict()
        if is_live_url(url) and fb[0].link_button("Open ↗", url, use_container_width=True):
            pass
        if fb[1].button("✓ Engage", key=f"{key_prefix}_engage_{int(row['id'])}", use_container_width=True):
            st.session_state._pending_action = ("engage", row_dict)
            st.rerun()
        if bookmarked:
            if fb[2].button("★ Saved", key=f"{key_prefix}_unbm_{int(row['id'])}", use_container_width=True):
                st.session_state._pending_action = ("unbookmark", row_dict)
                st.rerun()
        elif fb[2].button("☆ Bookmark", key=f"{key_prefix}_bm_{int(row['id'])}", use_container_width=True):
            st.session_state._pending_action = ("bookmark", row_dict)
            st.rerun()
        if fb[3].button("✕ Skip", key=f"{key_prefix}_skip_{int(row['id'])}", use_container_width=True):
            st.session_state._pending_action = ("skip", row_dict)
            st.rerun()


def render_rl_policy(agent) -> None:
    """Show learned domain Q-values from the RL agent."""
    from engageiq.reinforcement_learning import EngagementRLAgent

    if not isinstance(agent, EngagementRLAgent) or agent.rounds == 0:
        st.caption("RL policy: interact with opportunities (engage/bookmark/skip) to train the agent.")
        return

    summary = agent.summary()
    st.markdown("**Reinforcement learning policy**")
    c1, c2, c3 = st.columns(3)
    c1.metric("RL rounds", summary["rounds"])
    c2.metric("Total reward", f"{summary['total_reward']:.2f}")
    c3.metric("Avg reward", f"{summary['avg_reward']:.2f}")

    top = agent.top_domains(8)
    if top:
        df = pd.DataFrame(top, columns=["domain", "q_value"])
        chart = (
            alt.Chart(df)
            .mark_bar(color="#4F46E5")
            .encode(
                x=alt.X("q_value:Q", title="Learned Q-value (success rate)"),
                y=alt.Y("domain:N", sort="-x"),
                tooltip=["domain", "q_value"],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)
    st.caption(f"Policy: {summary['policy']} · entropy={summary['policy_entropy']:.2f}")


def render_activity_log(filter_action: str | None = None) -> None:
    log = st.session_state.get("activity_log", [])
    if filter_action and filter_action != "All":
        log = [e for e in log if e["action"] == filter_action.lower()]

    if not log:
        st.markdown(
            '<div class="empty-state"><div style="font-size:2rem;margin-bottom:0.5rem;">📋</div>'
            "No activity yet. Engage, bookmark, or skip opportunities to build your log.</div>",
            unsafe_allow_html=True,
        )
        return

    for e in log[:100]:
        st.markdown(
            f"""
<div class="activity-row">
  <div class="activity-time">{e["timestamp"]}</div>
  <div class="activity-body">
    <div>{action_badge(e["action"])} {source_badge(e["source"])} <span class="badge badge-domain">{e["domain"]}</span></div>
    <div class="activity-title">{e["title"]}</div>
    <div style="font-size:0.78rem;color:{BRAND["muted"]};">{e["url"]}</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_bookmarks_list() -> None:
    bookmarks = list(st.session_state.get("bookmarks", {}).values())
    if not bookmarks:
        st.markdown(
            '<div class="empty-state"><div style="font-size:2rem;margin-bottom:0.5rem;">★</div>'
            "No bookmarks yet. Save opportunities you want to revisit later.</div>",
            unsafe_allow_html=True,
        )
        return

    bookmarks = sorted(bookmarks, key=lambda b: b.get("bookmarked_at", ""), reverse=True)
    for i, b in enumerate(bookmarks, start=1):
        title = escape_html(b["title"])
        url = str(b["url"])
        desc = escape_html(b.get("description") or "")
        st.markdown(
            f"""
<div class="opp-card bookmarked">
  <div>
    <div style="font-weight:700;color:#0F172A;">{i}. {title}</div>
    <div style="margin:0.35rem 0;">{source_badge(b["source"])} {origin_badge(url)} <span class="badge badge-domain">{escape_html(b["domain"])}</span></div>
    {f'<div class="opp-desc">{desc}</div>' if desc else ''}
    <a class="opp-link" href="{escape_html(url)}" target="_blank" rel="noopener">{escape_html(url)}</a>
    <div style="font-size:0.75rem;color:{BRAND["muted"]};margin-top:0.35rem;">Saved {b.get("bookmarked_at", "—")} · Score {float(b.get("score_final", 0)):.2f}</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        bc1, bc2 = st.columns([0.25, 0.75])
        if is_live_url(url):
            bc1.link_button("Open ↗", url, key=f"open_bm_{b['opp_id']}", use_container_width=True)
        if bc1.button("Remove", key=f"rm_bm_{b['opp_id']}", use_container_width=True):
            st.session_state._pending_action = (
                "unbookmark",
                {
                    "id": b["opp_id"],
                    "title": b["title"],
                    "url": b["url"],
                    "domain": b["domain"],
                    "source": b["source"],
                    "score_final": b.get("score_final", 0),
                },
            )
            st.rerun()
