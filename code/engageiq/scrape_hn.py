from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from engageiq.domains import DOMAIN_QUERIES, DOMAINS
from engageiq.http_client import get as http_get

HN_API = "https://hacker-news.firebaseio.com/v0"


def _get_item(item_id: int) -> dict | None:
    r = http_get(f"{HN_API}/item/{item_id}.json", timeout=20)
    r.raise_for_status()
    return r.json()


def _match_domain(title: str, url: str) -> str | None:
    text = f"{title} {url}".lower()
    for domain in DOMAINS:
        for kw in DOMAIN_QUERIES[domain]["hn"]:
            if kw.lower() in text:
                return domain
    return None


def scrape_hackernews(max_stories: int = 3500) -> pd.DataFrame:
    print(f"  HN: fetching story id lists...", flush=True)
    story_ids: list[int] = []
    for endpoint in ("topstories", "newstories", "beststories", "askstories", "showstories"):
        r = http_get(f"{HN_API}/{endpoint}.json", timeout=20)
        r.raise_for_status()
        story_ids.extend(r.json())

    # dedupe while preserving order
    seen: set[int] = set()
    unique_ids: list[int] = []
    for sid in story_ids:
        if sid not in seen:
            seen.add(sid)
            unique_ids.append(sid)
    ids = unique_ids[:max_stories]
    print(f"  HN: downloading {len(ids)} stories (one HTTP request each)...", flush=True)

    rows: list[dict] = []
    row_id = 2_000_000

    for i, story_id in enumerate(ids):
        try:
            item = _get_item(story_id)
        except Exception:
            continue
        if not item or item.get("type") != "story":
            continue

        title = item.get("title") or ""
        url = item.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
        body = (item.get("text") or "").strip()
        # For Ask/Show HN posts, include self-text; for link posts use title + comment count context
        if body:
            text = f"{title}\n\n{body[:1500]}"
        else:
            text = title
        domain = _match_domain(title, url) or DOMAINS[i % len(DOMAINS)]

        ts = item.get("time")
        created = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            if ts
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        )

        rows.append(
            {
                "id": row_id,
                "source": "hackernews",
                "domain": domain,
                "title": title,
                "text": text,
                "url": url,
                "community": "news.ycombinator.com",
                "created_at": created,
                "upvotes": int(item.get("score") or 0),
                "comments": int(item.get("descendants") or 0),
                "author": item.get("by") or "",
                "lang": "",
                "stars": "",
                "forks": "",
                "issues_open": "",
                "good_first_issue": "",
            }
        )
        row_id += 1

        if (i + 1) % 100 == 0:
            print(f"  HN: {i + 1}/{len(ids)} stories fetched", flush=True)
            time.sleep(0.3)

    return pd.DataFrame(rows)
