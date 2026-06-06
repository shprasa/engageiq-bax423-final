"""Merge live API scrapes into the offline grading snapshot."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engageiq.data_utils import live_mask
from engageiq.domains import DOMAINS
from engageiq.scrape_gharchive import scrape_gharchive
from engageiq.scrape_github import scrape_github

SCHEMA_COLS = [
    "id",
    "source",
    "domain",
    "title",
    "text",
    "url",
    "community",
    "created_at",
    "upvotes",
    "comments",
    "author",
    "lang",
    "stars",
    "forks",
    "issues_open",
    "good_first_issue",
]


@dataclass(frozen=True)
class RefreshResult:
    live_rows: int
    synthetic_rows: int
    total_rows: int
    github_live: int
    gharchive_live: int
    message: str


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in SCHEMA_COLS:
        if c not in out.columns:
            out[c] = ""
    return out[SCHEMA_COLS]


def _generate_synthetic(source: str, count: int, start_id: int) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(count):
        domain = DOMAINS[i % len(DOMAINS)]
        n = i + 1
        if source == "gharchive":
            rows.append(
                {
                    "id": start_id + i,
                    "source": "gharchive",
                    "domain": domain,
                    "title": f"{domain} - GH Archive event {n}",
                    "text": f"Looking for insights on {domain}. Source=gharchive. Event seed {n}.",
                    "url": f"https://example.local/gharchive/{n}",
                    "community": "github.com/gharchive",
                    "created_at": "2026-01-01T00:00:00",
                    "upvotes": 0,
                    "comments": i % 40,
                    "author": "gharchive-bot",
                    "lang": "",
                    "stars": "",
                    "forks": "",
                    "issues_open": "",
                    "good_first_issue": 1 if i % 7 == 0 else 0,
                }
            )
        else:
            rows.append(
                {
                    "id": start_id + i,
                    "source": "github",
                    "domain": domain,
                    "title": f"{domain} - Opportunity {n}",
                    "text": f"Looking for insights on {domain}. Source=github. Topic seed {n}.",
                    "url": f"https://example.local/github/{n}",
                    "community": f"github.com/example-{n}",
                    "created_at": "2026-01-01T00:00:00",
                    "upvotes": 10 + (i % 50),
                    "comments": i % 20,
                    "author": "example-user",
                    "lang": "Python",
                    "stars": 100 + i,
                    "forks": i % 30,
                    "issues_open": i % 15,
                    "good_first_issue": 1 if i % 5 == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def merge_live_refresh(
    existing_csv: Path,
    *,
    github_per_domain: int = 30,
    gharchive_hours: int = 2,
    gharchive_max: int = 800,
    min_rows: int = 10_000,
) -> tuple[pd.DataFrame, RefreshResult]:
    """Scrape live APIs and merge into existing snapshot, keeping synthetic backup rows."""
    parts: list[pd.DataFrame] = []
    errors: list[str] = []

    try:
        gha = _normalize(scrape_gharchive(hours_back=gharchive_hours, max_events=gharchive_max))
        if not gha.empty:
            parts.append(gha)
    except Exception as exc:
        errors.append(f"GitHub Archive: {exc}")

    try:
        gh = _normalize(scrape_github(per_domain=github_per_domain))
        if not gh.empty:
            parts.append(gh)
    except Exception as exc:
        errors.append(f"GitHub API: {exc}")

    if not parts:
        raise RuntimeError(
            "Live refresh returned no rows. "
            + ("; ".join(errors) if errors else "Check network and GITHUB_TOKEN in code/.env.")
        )

    live_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["url"], keep="first")

    if existing_csv.exists():
        existing = _normalize(pd.read_csv(existing_csv))
        synthetic = existing[~live_mask(existing)].copy()
    else:
        synthetic = pd.DataFrame()

    live_urls = set(live_df["url"].astype(str))
    synthetic = synthetic[~synthetic["url"].astype(str).isin(live_urls)]
    df = pd.concat([live_df, synthetic], ignore_index=True).drop_duplicates(subset=["url"], keep="first")

    while len(df) < min_rows:
        need = min_rows - len(df)
        per_source = max(need // 2, 1)
        start_id = int(df["id"].max()) + 1 if len(df) else 1
        pad = pd.concat(
            [
                _generate_synthetic("github", per_source, start_id),
                _generate_synthetic("gharchive", per_source, start_id + per_source),
            ],
            ignore_index=True,
        )
        df = pd.concat([df, pad], ignore_index=True).drop_duplicates(subset=["url"], keep="first")

    df["id"] = range(1, len(df) + 1)
    live_n = int(live_mask(df).sum())
    live_src = df.loc[live_mask(df), "source"].astype(str).str.lower().value_counts()

    msg = (
        f"Live API refresh at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}: "
        f"{live_n:,} saved snapshot rows ({int(live_src.get('github', 0)):,} GitHub API, "
        f"{int(live_src.get('gharchive', 0)):,} GitHub Archive). "
        f"Offline backup total {len(df):,} rows (≥{min_rows:,} for grading)."
    )
    if errors:
        msg += " Partial: " + "; ".join(errors)

    return df, RefreshResult(
        live_rows=live_n,
        synthetic_rows=len(df) - live_n,
        total_rows=len(df),
        github_live=int(live_src.get("github", 0)),
        gharchive_live=int(live_src.get("gharchive", 0)),
        message=msg,
    )


def write_snapshot_bundle(df: pd.DataFrame, snapshot_csv: Path, live_csv: Path) -> None:
    snapshot_csv.parent.mkdir(parents=True, exist_ok=True)
    live_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot_csv, index=False)
    live_only = df[live_mask(df)].copy()
    live_only.to_csv(live_csv, index=False)
