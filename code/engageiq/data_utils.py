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


def excerpt(text: str, max_len: int = 320) -> str:
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


def _is_github_issue(row: pd.Series) -> bool:
    url = str(row.get("url") or "").lower()
    if "/issues/" in url:
        return True
    return int(row.get("good_first_issue") or 0) == 1 and str(row.get("source", "")).lower() == "github"


def opportunity_type_label(row: pd.Series) -> str:
    src = str(row.get("source", "")).lower()
    if src == "github":
        if _is_github_issue(row):
            return "GitHub Issue · Good First Issue" if int(row.get("good_first_issue") or 0) == 1 else "GitHub Issue"
        return "GitHub Repository"
    if src == "hackernews":
        url = str(row.get("url") or "")
        if "news.ycombinator.com/item" in url:
            return "Hacker News Discussion"
        return "Hacker News Story"
    if src == "reddit":
        return "Reddit Thread"
    return "Engagement Opportunity"


def display_title(row: pd.Series) -> str:
    """Primary headline — real title from the source API."""
    src = str(row.get("source", "")).lower()
    title = str(row.get("title") or "").strip()
    community = str(row.get("community") or "").strip()
    text = str(row.get("text") or "").strip()

    if src == "github":
        if _is_github_issue(row):
            return title or "Untitled GitHub issue"
        # Repository: prefer human-readable description as headline when title is just owner/repo
        if community and (title == community or "/" in title and len(title) < 80):
            if text and text.lower() != title.lower() and not text.lower().startswith("looking for insights"):
                return text
            return community
        return title or community or "GitHub repository"

    if src == "hackernews":
        return title or "Hacker News story"

    if src == "reddit":
        return title or "Reddit discussion"

    if title.lower().startswith(f"{row.get('domain', '')} - opportunity".lower()):
        return text or title
    return title or "Opportunity"


def display_subtitle(row: pd.Series) -> str:
    """Secondary context line under the headline."""
    src = str(row.get("source", "")).lower()
    title = str(row.get("title") or "").strip()
    community = str(row.get("community") or "").strip()
    author = str(row.get("author") or "").strip()

    if src == "github":
        if _is_github_issue(row):
            parts = [f"Repository: {community}" if community else ""]
            if author:
                parts.append(f"Opened by {author}")
            return " · ".join(p for p in parts if p)
        parts = []
        if community and community != display_title(row):
            parts.append(community)
        if author:
            parts.append(f"maintainer {author}")
        lang = str(row.get("lang") or "").strip()
        if lang:
            parts.append(lang)
        return " · ".join(parts) if parts else "Open-source project on GitHub"

    if src == "hackernews":
        pts = int(float(row.get("upvotes") or 0))
        com = int(float(row.get("comments") or 0))
        parts = [f"{pts:,} points", f"{com:,} comments"]
        if author:
            parts.append(f"by {author}")
        return " · ".join(parts)

    if src == "reddit":
        parts = []
        if community:
            parts.append(f"r/{community}" if not community.startswith("r/") else community)
        if author:
            parts.append(f"u/{author}")
        return " · ".join(parts) if parts else "Community discussion on Reddit"

    return str(row.get("domain") or "")


def display_summary(row: pd.Series) -> str:
    """Longer body text to help the user decide."""
    src = str(row.get("source", "")).lower()
    title = str(row.get("title") or "").strip()
    text = excerpt(str(row.get("text") or ""), max_len=400)
    headline = display_title(row)

    if src == "github":
        if _is_github_issue(row):
            return text or f"Issue in {row.get('community', 'repository')}: {title}"
        if text and text.lower() != headline.lower():
            return text
        stars = row.get("stars")
        issues = row.get("issues_open")
        bits = [f"Repository {row.get('community', title)}."]
        if pd.notna(stars) and str(stars) != "":
            bits.append(f"{int(float(stars)):,} stars.")
        if pd.notna(issues) and str(issues) != "":
            bits.append(f"{int(float(issues)):,} open issues.")
        bits.append("Browse issues and README to find a contribution entry point.")
        return " ".join(bits)

    if src == "hackernews":
        if text and text.lower() != title.lower():
            return text
        url = str(row.get("url") or "")
        if "news.ycombinator.com/item" in url:
            return f"Active Hacker News thread with {int(float(row.get('comments') or 0)):,} comments — join the technical discussion."
        return f"Trending link shared on Hacker News ({int(float(row.get('upvotes') or 0)):,} points). Read the article, then add a substantive comment on the discussion thread."

    if src == "reddit":
        if text:
            return text
        return f"Discussion thread in {row.get('domain', 'tech community')} — read comments and add value with a specific tip or question."

    return text or f"Opportunity in {row.get('domain', 'this domain')}."


def decision_facts(row: pd.Series) -> list[tuple[str, str]]:
    """Key-value facts for the decision panel."""
    facts: list[tuple[str, str]] = [
        ("Source", opportunity_type_label(row)),
        ("Domain", str(row.get("domain") or "—")),
        ("Posted", format_created(row.get("created_at"))),
    ]
    src = str(row.get("source", "")).lower()

    if src == "github":
        if pd.notna(row.get("stars")) and str(row.get("stars")) != "":
            facts.append(("Stars", f"{int(float(row['stars'])):,}"))
        if pd.notna(row.get("forks")) and str(row.get("forks")) != "":
            facts.append(("Forks", f"{int(float(row['forks'])):,}"))
        if pd.notna(row.get("issues_open")) and str(row.get("issues_open")) != "":
            facts.append(("Open issues", f"{int(float(row['issues_open'])):,}"))
        lang = str(row.get("lang") or "").strip()
        if lang:
            facts.append(("Language", lang))
        if int(row.get("good_first_issue") or 0) == 1:
            facts.append(("Contribution", "Good first issue"))
    elif src == "hackernews":
        facts.append(("Points", f"{int(float(row.get('upvotes') or 0)):,}"))
        facts.append(("Comments", f"{int(float(row.get('comments') or 0)):,}"))
        author = str(row.get("author") or "").strip()
        if author:
            facts.append(("Author", author))
    elif src == "reddit":
        facts.append(("Upvotes", f"{int(float(row.get('upvotes') or 0)):,}"))
        facts.append(("Comments", f"{int(float(row.get('comments') or 0)):,}"))

    author = str(row.get("author") or "").strip()
    if author and not any(k == "Author" for k, _ in facts):
        facts.append(("Author", author))

    community = str(row.get("community") or "").strip()
    if community and src == "github":
        facts.append(("Repo", community[:40]))

    return facts


def estimated_engagement_time(row: pd.Series) -> str:
    effort = float(row.get("score_effort") or 0.5)
    if int(row.get("good_first_issue") or 0) == 1:
        return "< 1 hour (good first issue)"
    if effort < 0.35:
        return "< 1 hour"
    if effort < 0.65:
        return "1–2 hours"
    return "2+ hours (deeper contribution)"


def opportunity_meta(row: pd.Series) -> str:
    return " · ".join(f"{k}: {v}" for k, v in decision_facts(row)[:6])


def describe_opportunity(row: pd.Series) -> str:
    return display_summary(row)


def escape_html(text: str) -> str:
    return html.escape(str(text or ""), quote=True)
