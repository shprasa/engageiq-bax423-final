from __future__ import annotations

import html
import re
from datetime import datetime

import pandas as pd


def is_live_url(url: str) -> bool:
    u = str(url or "").lower()
    return "example.local" not in u and u.startswith("http")


def is_offline_url(url: str) -> bool:
    return not is_live_url(url)


def live_mask(df: pd.DataFrame) -> pd.Series:
    return df["url"].astype(str).apply(is_live_url)


def sort_live_first(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_is_live"] = live_mask(out).astype(int)
    out = out.sort_values(["_is_live", "id"], ascending=[False, True]).drop(columns=["_is_live"])
    return out.reset_index(drop=True)


def ranking_corpus(df: pd.DataFrame, live_only: bool = True, english_only: bool = True) -> pd.DataFrame:
    out = df
    if live_only:
        live = out[live_mask(out)]
        if len(live) >= 100:
            out = live.reset_index(drop=True)
        else:
            out = out.reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    if english_only:
        eng = out[english_mask(out)]
        if len(eng) >= 100:
            out = eng.reset_index(drop=True)
    return out


def excerpt(text: str, max_len: int = 320) -> str:
    raw = strip_markdown(str(text or ""))
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw or raw.lower().startswith("looking for insights on"):
        return ""
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1].rsplit(" ", 1)[0] + "…"


def strip_markdown(text: str) -> str:
    """Flatten markdown so Streamlit does not render issue bodies as giant headings."""
    raw = str(text or "")
    raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s#{1,6}\s+", " · ", raw)
    raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
    raw = re.sub(r"\*([^*]+)\*", r"\1", raw)
    raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
    raw = re.sub(r"`([^`]+)`", r"\1", raw)
    raw = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", raw)
    raw = re.sub(r"^>\s+", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^[-*]\s+", "", raw, flags=re.MULTILINE)
    return raw.strip()


_NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF"
    r"\u0600-\u06FF\u0900-\u097F\uAC00-\uD7AF\u0E00-\u0E7F]"
)


def is_likely_english(text: str, min_ratio: float = 0.82) -> bool:
    sample = strip_markdown(str(text or ""))[:500].strip()
    if not sample:
        return True
    if len(_NON_LATIN.findall(sample)) >= 2:
        return False
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if ord(c) < 128)
    return (latin / len(letters)) >= min_ratio


def row_is_english(row: pd.Series) -> bool:
    title = str(row.get("title") or "")
    body = str(row.get("text") or "")
    return is_likely_english(f"{title}\n{body}")


def english_mask(df: pd.DataFrame) -> pd.Series:
    return df.apply(row_is_english, axis=1)


def _clean_lang(value) -> str:
    lang = str(value or "").strip()
    if not lang or lang.lower() in ("nan", "none", "null"):
        return ""
    return lang


def format_created(value, *, compact: bool = False) -> str:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return "—"
        if compact:
            return ts.strftime("%b %d, '%y")
        return ts.strftime("%b %d, %Y")
    except Exception:
        return "—"


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _has_value(value) -> bool:
    try:
        if value is None or pd.isna(value):
            return False
    except (TypeError, ValueError):
        return bool(str(value).strip())
    return str(value).strip() != ""


def _is_github_issue(row: pd.Series) -> bool:
    url = str(row.get("url") or "").lower()
    if "/issues/" in url:
        return True
    return _safe_int(row.get("good_first_issue")) == 1 and str(row.get("source", "")).lower() == "github"


def opportunity_type_label(row: pd.Series) -> str:
    src = str(row.get("source", "")).lower()
    if src == "github":
        if _is_github_issue(row):
            return "GitHub Issue · Good First Issue" if _safe_int(row.get("good_first_issue")) == 1 else "GitHub Issue"
        return "GitHub Repository"
    if src == "gharchive":
        url = str(row.get("url") or "")
        if "/issues/" in url:
            return "GitHub Archive · Issue Event"
        if "/pull/" in url:
            return "GitHub Archive · Pull Request Event"
        return "GitHub Archive · Event"
    return "Engagement Opportunity"


def short_source_label(row: pd.Series) -> str:
    """Compact source label for card fact chips."""
    src = str(row.get("source", "")).lower()
    if src == "github":
        if _is_github_issue(row):
            return "GitHub GFI" if _safe_int(row.get("good_first_issue")) == 1 else "GitHub issue"
        return "GitHub repo"
    if src == "gharchive":
        url = str(row.get("url") or "")
        if "/issues/" in url:
            return "GH Archive issue"
        if "/pull/" in url:
            return "GH Archive PR"
        return "GH Archive event"
    return "Opportunity"


def _truncate_text(text: str, max_len: int = 18) -> str:
    s = str(text or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return "—"
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


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

    if src == "gharchive":
        return title or "GitHub Archive event"

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
        lang = _clean_lang(row.get("lang"))
        if lang:
            parts.append(lang)
        return " · ".join(parts) if parts else "Open-source project on GitHub"

    if src == "gharchive":
        com = _safe_int(row.get("comments"))
        parts = [f"{com:,} comments"]
        if author:
            parts.append(f"by {author}")
        if community:
            parts.append(community)
        return " · ".join(parts)

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
        if _has_value(stars):
            bits.append(f"{int(float(stars)):,} stars.")
        if _has_value(issues):
            bits.append(f"{int(float(issues)):,} open issues.")
        bits.append("Browse issues and README to find a contribution entry point.")
        return " ".join(bits)

    if src == "gharchive":
        if text and text.lower() != title.lower():
            return text
        return (
            f"Public GitHub Archive event on {row.get('community', 'a repository')} — "
            f"review the issue/PR and add a comment or small contribution."
        )

    return text or f"Opportunity in {row.get('domain', 'this domain')}."


def decision_facts(row: pd.Series) -> list[tuple[str, str]]:
    """Key-value facts for the decision panel (compact values for card layout)."""
    facts: list[tuple[str, str]] = [
        ("Source", short_source_label(row)),
        ("Domain", _truncate_text(str(row.get("domain") or "—"), 22)),
        ("Posted", format_created(row.get("created_at"), compact=True)),
    ]
    src = str(row.get("source", "")).lower()

    if src == "github":
        if _has_value(row.get("stars")):
            facts.append(("Stars", f"{int(float(row['stars'])):,}"))
        if _has_value(row.get("forks")):
            facts.append(("Forks", f"{int(float(row['forks'])):,}"))
        if _has_value(row.get("issues_open")):
            facts.append(("Open issues", f"{int(float(row['issues_open'])):,}"))
        lang = _clean_lang(row.get("lang"))
        if lang:
            facts.append(("Language", _truncate_text(lang, 16)))
        is_gfi = _safe_int(row.get("good_first_issue")) == 1
        if is_gfi and not _is_github_issue(row):
            facts.append(("GFI", "Yes"))
    elif src == "gharchive":
        if _has_value(row.get("comments")):
            facts.append(("Comments", f"{_safe_int(row.get('comments')):,}"))
        if community := str(row.get("community") or "").strip():
            facts.append(("Repo", _truncate_text(community, 22)))

    author = str(row.get("author") or "").strip()
    if author and not any(k == "Author" for k, _ in facts):
        facts.append(("Author", author))

    community = str(row.get("community") or "").strip()
    if community and src == "github":
        facts.append(("Repo", community[:40]))

    return facts


def estimated_engagement_time(row: pd.Series) -> str:
    effort = _safe_float(row.get("score_effort"), 0.5)
    if _safe_int(row.get("good_first_issue")) == 1:
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


def escape_markdown(text: str) -> str:
    """Escape characters that Streamlit markdown treats as formatting."""
    s = str(text or "")
    for ch in ("\\", "*", "_", "`", "#", "[", "]"):
        s = s.replace(ch, f"\\{ch}")
    return s


SORT_OPTIONS: dict[str, tuple[str, bool]] = {
    "Best match to your interests": ("score_final", False),
    "Quickest to contribute": ("_engagement_rank", True),
    "Most visible / popular": ("score_visibility", False),
    "Most active community": ("score_health", False),
    "Most recent": ("created_at", False),
}

SOURCE_FILTER_OPTIONS: dict[str, str | None] = {
    "All sources": None,
    "GitHub API only": "github",
    "GitHub Archive only": "gharchive",
}

ORIGIN_FILTER_OPTIONS: dict[str, str | None] = {
    "All opportunities": None,
    "Live API URLs only": "live",
}


def card_origin_label(url: str, *, use_live_snapshot_pool: bool) -> str:
    """Human-readable data-pool label for opportunity cards."""
    if is_live_url(url):
        return "Live API snapshot" if use_live_snapshot_pool else "Offline snapshot · live API"
    return "Invalid URL"


def dataset_pool_summary(df: pd.DataFrame, *, use_live_snapshot_pool: bool) -> str:
    live_n = int(live_mask(df).sum())
    if use_live_snapshot_pool:
        return f"Ranking pool: {live_n:,} live API rows from the bundled snapshot"
    return f"Ranking pool: offline bundled snapshot ({live_n:,} saved rows, no live API calls)"


def _engagement_rank_series(df: pd.DataFrame) -> pd.Series:
    """Lower rank = quicker to engage (GFI first, then low effort score)."""
    gfi = df["good_first_issue"].fillna(0).astype(float)
    effort = df["score_effort"].fillna(0.5).astype(float)
    return gfi * -1000 + effort


def filter_ranked_results(
    df: pd.DataFrame,
    *,
    source: str = "All",
    origin: str = "All",
    domains: list[str] | None = None,
    good_first_issue_only: bool = False,
    max_effort: str = "Any",
) -> pd.DataFrame:
    out = df.copy()
    src_val = SOURCE_FILTER_OPTIONS.get(source, source)
    if src_val:
        out = out[out["source"].astype(str).str.lower() == src_val.lower()]
    if origin == "Live from web" or origin == "Live API" or origin in ("Saved from live API", "Live API URLs only"):
        out = out[live_mask(out)]
    elif origin in ("Offline practice data", "Offline backup", "Synthetic practice rows"):
        out = out[~live_mask(out)]
    if domains:
        dom_set = {d.strip() for d in domains if d.strip()}
        if dom_set:
            out = out[out["domain"].astype(str).isin(dom_set)]
    if good_first_issue_only:
        out = out[out["good_first_issue"].fillna(0).astype(int) == 1]
    if max_effort == "Under 1 hour":
        gfi = out["good_first_issue"].fillna(0).astype(int) == 1
        low_effort = out["score_effort"].fillna(1.0).astype(float) < 0.35
        out = out[gfi | low_effort]
    elif max_effort == "Under 2 hours":
        gfi = out["good_first_issue"].fillna(0).astype(int) == 1
        med_effort = out["score_effort"].fillna(1.0).astype(float) < 0.65
        out = out[gfi | med_effort]
    return out.reset_index(drop=True)


def source_mix_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results"
    gh = int((df["source"].astype(str).str.lower() == "github").sum())
    gha = int((df["source"].astype(str).str.lower() == "gharchive").sum())
    live_n = int(live_mask(df).sum())
    offline_n = len(df) - live_n
    return f"{len(df)} items ({gh} GitHub API, {gha} GitHub Archive, {live_n} live, {offline_n} offline)"


def sort_ranked_results(df: pd.DataFrame, sort_by: str = "Best match to your interests") -> pd.DataFrame:
    if df.empty:
        return df
    spec = SORT_OPTIONS.get(sort_by, SORT_OPTIONS["Best match to your interests"])
    col, ascending = spec
    out = df.copy()
    if col == "_engagement_rank":
        out["_engagement_rank"] = _engagement_rank_series(out)
    if col == "created_at":
        out["_sort_ts"] = pd.to_datetime(out["created_at"], errors="coerce")
        out = out.sort_values("_sort_ts", ascending=ascending, na_position="last").drop(
            columns=["_sort_ts"], errors="ignore"
        )
    else:
        if col not in out.columns:
            col = "score_final"
            ascending = False
        out = out.sort_values(col, ascending=ascending, na_position="last")
    return out.drop(columns=["_engagement_rank"], errors="ignore").reset_index(drop=True)
