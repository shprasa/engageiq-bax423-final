from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from engageiq.domains import DOMAIN_QUERIES, DOMAINS
from engageiq.http_client import get as http_get
from engageiq.secrets import require_github_token

GITHUB_API = "https://api.github.com"


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _search_repos(token: str, query: str, per_page: int = 30) -> list[dict]:
    url = f"{GITHUB_API}/search/repositories"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
    r = http_get(url, headers=_headers(token), params=params, timeout=30)
    if r.status_code == 403:
        reset = r.headers.get("X-RateLimit-Reset")
        if reset:
            wait = max(1, int(reset) - int(time.time()) + 1)
            time.sleep(min(wait, 60))
            r = http_get(url, headers=_headers(token), params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def _search_issues(token: str, query: str, per_page: int = 20) -> list[dict]:
    url = f"{GITHUB_API}/search/issues"
    params = {"q": query, "sort": "updated", "order": "desc", "per_page": per_page}
    r = http_get(url, headers=_headers(token), params=params, timeout=30)
    if r.status_code == 403:
        time.sleep(10)
        r = http_get(url, headers=_headers(token), params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def scrape_github(per_domain: int = 250) -> pd.DataFrame:
    token = require_github_token()
    rows: list[dict] = []
    row_id = 1_000_000
    print(f"  GitHub: up to {per_domain} items x {len(DOMAINS)} domains (~2-4 sec per search)", flush=True)

    for di, domain in enumerate(DOMAINS, start=1):
        print(f"  GitHub: domain {di}/{len(DOMAINS)} — {domain}", flush=True)
        keywords = DOMAIN_QUERIES[domain]["github"]
        collected = 0
        for kw in keywords:
            if collected >= per_domain:
                break
            try:
                repos = _search_repos(token, f"{kw} stars:>50", per_page=30)
            except (requests.HTTPError, requests.RequestException):
                time.sleep(12)
                continue

            for repo in repos:
                if collected >= per_domain:
                    break
                labels = [lbl.get("name", "").lower() for lbl in repo.get("labels", [])]
                gfi = 1 if "good first issue" in labels else 0
                created = repo.get("created_at") or datetime.now(timezone.utc).isoformat()
                rows.append(
                    {
                        "id": row_id,
                        "source": "github",
                        "domain": domain,
                        "title": repo.get("full_name", repo.get("name", "")),
                        "text": (repo.get("description") or "")[:2000],
                        "url": repo.get("html_url", ""),
                        "community": repo.get("full_name", ""),
                        "created_at": created.replace("Z", ""),
                        "upvotes": int(repo.get("stargazers_count") or 0),
                        "comments": int(repo.get("open_issues_count") or 0),
                        "author": (repo.get("owner") or {}).get("login", ""),
                        "lang": repo.get("language") or "",
                        "stars": int(repo.get("stargazers_count") or 0),
                        "forks": int(repo.get("forks_count") or 0),
                        "issues_open": int(repo.get("open_issues_count") or 0),
                        "good_first_issue": gfi,
                    }
                )
                row_id += 1
                collected += 1
            time.sleep(2)

            # Also pull a few good-first-issue items per domain
            try:
                issues = _search_issues(
                    token,
                    f'label:"good first issue" {kw} state:open',
                    per_page=10,
                )
            except (requests.HTTPError, requests.RequestException):
                issues = []

            for issue in issues[:5]:
                if collected >= per_domain:
                    break
                repo_name = (issue.get("repository_url") or "").split("/repos/")[-1]
                rows.append(
                    {
                        "id": row_id,
                        "source": "github",
                        "domain": domain,
                        "title": issue.get("title", ""),
                        "text": (issue.get("body") or "")[:2000],
                        "url": issue.get("html_url", ""),
                        "community": repo_name,
                        "created_at": (issue.get("created_at") or "").replace("Z", ""),
                        "upvotes": int(issue.get("comments") or 0),
                        "comments": int(issue.get("comments") or 0),
                        "author": (issue.get("user") or {}).get("login", ""),
                        "lang": "",
                        "stars": "",
                        "forks": "",
                        "issues_open": "",
                        "good_first_issue": 1,
                    }
                )
                row_id += 1
                collected += 1
            time.sleep(2)

    return pd.DataFrame(rows)
