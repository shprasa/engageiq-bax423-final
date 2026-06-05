from __future__ import annotations

import html
import re
from datetime import datetime

import pandas as pd


def is_live_url(url: str) -> bool:
    u = str(url or "").lower()
    return "example.local" not in u and u.startswith("http")


def live_mask(df: pd.DataFrame) -> pd.Series:
    return df["url"].astype(str).apply(is_live_url)


def sort_live_first(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_is_live"] = live_mask(out).astype(int)
    out = out.sort_values(["_is_live", "id"], ascending=[False, True]).drop(columns=["_is_live"])
    return out.reset_index(drop=True)


def ranking_corpus(df: pd.DataFrame, live_only: bool = True) -> pd.DataFrame:
    if live_only:
        live = df[live_mask(df)]
        if len(live) >= 100:
            return live.reset_index(drop=True)
    return df.reset_index(drop=True)


def excerpt(text: str, max_len: int = 220) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw or raw.lower().startswith("looking for insights on"):
        return ""
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1].rsplit(" ", 1)[0] + "…"


def format_created(value) -> str:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return "—"
        return ts.strftime("%b %d, %Y")
    except Exception:
        return "—"


def opportunity_meta(row: pd.Series) -> str:
    parts: list[str] = []
    author = str(row.get("author") or "").strip()
    community = str(row.get("community") or "").strip()
    if author:
        parts.append(f"by {author}")
    if community and community != author:
        parts.append(community)
    parts.append(format_created(row.get("created_at")))

    src = str(row.get("source", "")).lower()
    if src == "github":
        stars = row.get("stars")
        forks = row.get("forks")
        issues = row.get("issues_open")
        if pd.notna(stars) and str(stars) != "":
            parts.append(f"★ {int(float(stars)):,}")
        if pd.notna(forks) and str(forks) != "":
            parts.append(f"⑂ {int(float(forks)):,}")
        if pd.notna(issues) and str(issues) != "":
            parts.append(f"issues {int(float(issues)):,}")
        lang = str(row.get("lang") or "").strip()
        if lang:
            parts.append(lang)
    else:
        up = row.get("upvotes")
        com = row.get("comments")
        if pd.notna(up):
            parts.append(f"↑ {int(float(up)):,}")
        if pd.notna(com):
            parts.append(f"💬 {int(float(com)):,}")

    return " · ".join(parts)


def describe_opportunity(row: pd.Series) -> str:
    body = excerpt(str(row.get("text") or ""))
    if body:
        return body
    src = str(row.get("source", "")).lower()
    title = str(row.get("title") or "")
    domain = str(row.get("domain") or "")
    if src == "hackernews":
        return f"Hacker News discussion in {domain}: {title}"
    if src == "github":
        return f"Open-source repository in {domain} — explore issues and contribution opportunities."
    if src == "reddit":
        return f"Community thread in {domain} — join the discussion with a substantive comment."
    return f"Engagement opportunity in {domain}."


def escape_html(text: str) -> str:
    return html.escape(str(text or ""), quote=True)
