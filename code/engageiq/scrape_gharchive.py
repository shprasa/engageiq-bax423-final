from __future__ import annotations

import gzip
import io
import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from engageiq.domains import DOMAIN_QUERIES, DOMAINS
from engageiq.http_client import get as http_get

GHARCHIVE_BASE = "https://data.gharchive.org"

# Event types that represent real engagement opportunities.
ENGAGEMENT_TYPES = frozenset(
    {
        "IssuesEvent",
        "PullRequestEvent",
        "IssueCommentEvent",
        "PullRequestReviewCommentEvent",
    }
)


def _match_domain(text: str) -> str | None:
    text = text.lower()
    for domain in DOMAINS:
        for kw in DOMAIN_QUERIES[domain]["gharchive"]:
            if kw.lower() in text:
                return domain
    return None


def _label_has_gfi(labels) -> int:
    if not labels:
        return 0
    for lab in labels:
        name = str(lab.get("name") if isinstance(lab, dict) else lab).lower()
        if "good first issue" in name or name == "good-first-issue":
            return 1
    return 0


def _event_to_row(event: dict, row_id: int, fallback_idx: int) -> dict | None:
    if event.get("type") not in ENGAGEMENT_TYPES:
        return None
    if event.get("public") is False:
        return None

    repo = event.get("repo") or {}
    repo_name = str(repo.get("name") or "")
    actor = str((event.get("actor") or {}).get("login") or "")
    created_at = str(event.get("created_at") or "")[:19]
    payload = event.get("payload") or {}
    action = str(payload.get("action") or "")
    etype = event["type"]

    title = ""
    text = ""
    url = f"https://github.com/{repo_name}" if repo_name else ""
    comments = 0
    gfi = 0

    if etype == "IssuesEvent" and action in ("opened", "reopened"):
        issue = payload.get("issue") or {}
        title = str(issue.get("title") or f"Issue on {repo_name}")
        body = str(issue.get("body") or "")[:1500]
        text = f"{title}\n\n{body}".strip()
        url = str(issue.get("html_url") or url)
        comments = int(issue.get("comments") or 0)
        gfi = _label_has_gfi(issue.get("labels"))
    elif etype == "PullRequestEvent" and action in ("opened", "reopened"):
        pr = payload.get("pull_request") or {}
        title = str(pr.get("title") or f"Pull request on {repo_name}")
        body = str(pr.get("body") or "")[:1500]
        text = f"{title}\n\n{body}".strip()
        url = str(pr.get("html_url") or url)
        comments = int(pr.get("comments") or 0)
    elif etype == "IssueCommentEvent":
        issue = payload.get("issue") or {}
        comment = payload.get("comment") or {}
        title = str(issue.get("title") or f"Comment thread on {repo_name}")
        comment_body = str(comment.get("body") or "")[:800]
        text = f"{title}\n\nLatest comment:\n{comment_body}".strip()
        url = str(issue.get("html_url") or url)
        comments = int(issue.get("comments") or 0)
        gfi = _label_has_gfi(issue.get("labels"))
    elif etype == "PullRequestReviewCommentEvent":
        pr = payload.get("pull_request") or {}
        comment = payload.get("comment") or {}
        title = str(pr.get("title") or f"PR review on {repo_name}")
        comment_body = str(comment.get("body") or "")[:800]
        text = f"{title}\n\nReview comment:\n{comment_body}".strip()
        url = str(pr.get("html_url") or url)
        comments = int(pr.get("comments") or 0)
    else:
        return None

    if not title or not url.startswith("http"):
        return None

    match_text = f"{repo_name} {title} {text}"
    domain = _match_domain(match_text) or DOMAINS[fallback_idx % len(DOMAINS)]

    return {
        "id": row_id,
        "source": "gharchive",
        "domain": domain,
        "title": title[:500],
        "text": text[:2000],
        "url": url,
        "community": repo_name,
        "created_at": created_at,
        "upvotes": 0,
        "comments": comments,
        "author": actor,
        "lang": "",
        "stars": "",
        "forks": "",
        "issues_open": "",
        "good_first_issue": gfi,
    }


def scrape_gharchive(hours_back: int = 6, max_events: int = 2500, archive_lag_hours: int = 2) -> pd.DataFrame:
    """Download recent GH Archive hourly files and extract engagement events."""
    end = datetime.now(timezone.utc) - timedelta(hours=archive_lag_hours)
    rows: list[dict] = []
    row_id = 3_000_000
    seen_urls: set[str] = set()

    print(f"  GH Archive: scanning last {hours_back} hour(s) (max {max_events} events)...", flush=True)

    for hour_offset in range(hours_back):
        if len(rows) >= max_events:
            break
        dt = end - timedelta(hours=hour_offset)
        url = f"{GHARCHIVE_BASE}/{dt.year:04d}-{dt.month:02d}-{dt.day:02d}-{dt.hour}.json.gz"
        print(f"  GH Archive: downloading {url} ...", flush=True)
        try:
            resp = http_get(url, timeout=180)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  GH Archive: skip hour ({exc})", flush=True)
            continue

        try:
            with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
                for line_no, raw_line in enumerate(gz, start=1):
                    if len(rows) >= max_events:
                        break
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    row = _event_to_row(event, row_id, len(rows))
                    if not row:
                        continue
                    if row["url"] in seen_urls:
                        continue
                    seen_urls.add(row["url"])
                    rows.append(row)
                    row_id += 1
                    if len(rows) % 250 == 0:
                        print(f"  GH Archive: {len(rows)} events collected", flush=True)
        except Exception as exc:
            print(f"  GH Archive: parse error for {url}: {exc}", flush=True)
            continue
        time.sleep(0.5)

    print(f"  GH Archive: collected {len(rows)} live events", flush=True)
    return pd.DataFrame(rows)
