from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from engageiq.domains import DOMAIN_QUERIES, DOMAINS
from engageiq.secrets import get_reddit_config, reddit_configured


def scrape_reddit(per_sub_limit: int = 100) -> pd.DataFrame:
    if not reddit_configured():
        return pd.DataFrame()

    import praw

    cfg = get_reddit_config()
    reddit = praw.Reddit(
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
        user_agent=cfg["user_agent"],
        username=cfg["username"],
        password=cfg["password"],
    )

    rows: list[dict] = []
    row_id = 3_000_000

    for domain in DOMAINS:
        for sub_name in DOMAIN_QUERIES[domain]["reddit"]:
            sub = reddit.subreddit(sub_name)
            try:
                posts = list(sub.hot(limit=per_sub_limit))
            except Exception:
                time.sleep(2)
                continue

            for post in posts:
                created = datetime.fromtimestamp(
                    float(post.created_utc), tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S")
                rows.append(
                    {
                        "id": row_id,
                        "source": "reddit",
                        "domain": domain,
                        "title": post.title or "",
                        "text": (post.selftext or post.title or "")[:2000],
                        "url": f"https://reddit.com{post.permalink}",
                        "community": f"r/{sub_name}",
                        "created_at": created,
                        "upvotes": int(post.score or 0),
                        "comments": int(post.num_comments or 0),
                        "author": str(post.author) if post.author else "",
                        "lang": "",
                        "stars": "",
                        "forks": "",
                        "issues_open": "",
                        "good_first_issue": "",
                    }
                )
                row_id += 1
            time.sleep(1)

    return pd.DataFrame(rows)
