"""Merge live API scrapes into the offline grading snapshot (live data only)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engageiq.data_utils import live_mask
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
    out = out[SCHEMA_COLS]
    return out[live_mask(out)].drop_duplicates(subset=["url"], keep="first")


def merge_live_refresh(
    existing_csv: Path,
    *,
    github_per_domain: int = 80,
    gharchive_hours: int = 72,
    gharchive_max: int = 0,
    min_rows: int = 10_000,
) -> tuple[pd.DataFrame, RefreshResult]:
    """Scrape live APIs and merge into existing snapshot (live rows only)."""
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

    if existing_csv.exists():
        existing = _normalize(pd.read_csv(existing_csv))
        if not existing.empty:
            parts.insert(0, existing)

    if not parts:
        raise RuntimeError(
            "Live refresh returned no rows. "
            + ("; ".join(errors) if errors else "Check network and GITHUB_TOKEN in code/.env.")
        )

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["url"], keep="first")

    if len(df) < min_rows:
        try:
            extra = _normalize(scrape_gharchive(hours_back=max(gharchive_hours * 2, 336), max_events=gharchive_max))
            if not extra.empty:
                df = pd.concat([df, extra], ignore_index=True).drop_duplicates(subset=["url"], keep="first")
        except Exception as exc:
            errors.append(f"Extended GH Archive: {exc}")

    df = df[live_mask(df)].reset_index(drop=True)
    if len(df) < min_rows:
        raise RuntimeError(
            f"Live refresh produced {len(df):,} rows (need ≥{min_rows:,}). Run scripts/build_snapshot.py for a full rebuild."
        )

    df["id"] = range(1, len(df) + 1)
    live_n = len(df)
    live_src = df["source"].astype(str).str.lower().value_counts()

    msg = (
        f"Live API refresh at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}: "
        f"{live_n:,} live snapshot rows ({int(live_src.get('github', 0)):,} GitHub API, "
        f"{int(live_src.get('gharchive', 0)):,} GitHub Archive). "
        f"All rows are real API-sourced URLs (no synthetic padding)."
    )
    if errors:
        msg += " Partial: " + "; ".join(errors)

    return df, RefreshResult(
        live_rows=live_n,
        synthetic_rows=0,
        total_rows=live_n,
        github_live=int(live_src.get("github", 0)),
        gharchive_live=int(live_src.get("gharchive", 0)),
        message=msg,
    )


def write_snapshot_bundle(df: pd.DataFrame, snapshot_csv: Path, live_csv: Path) -> None:
    snapshot_csv.parent.mkdir(parents=True, exist_ok=True)
    live_csv.parent.mkdir(parents=True, exist_ok=True)
    live_only = df[live_mask(df)].copy()
    live_only.to_csv(snapshot_csv, index=False)
    live_only.to_csv(live_csv, index=False)


def write_snapshot_everywhere(df: pd.DataFrame, paths) -> None:
    """Keep project data/ and code/data/ in sync (Streamlit Cloud reads code/data/)."""
    live_only = df[live_mask(df)].copy()
    snap_paths = {
        paths.snapshot_csv,
        paths.project_root / "data" / "opportunities_snapshot.csv",
        paths.code_dir / "data" / "opportunities_snapshot.csv",
    }
    live_paths = {
        paths.live_csv,
        paths.project_root / "data" / "live_opportunities.csv",
        paths.code_dir / "data" / "live_opportunities.csv",
    }
    for snap_path in snap_paths:
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(snap_path, index=False)
    for live_path in live_paths:
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_only.to_csv(live_path, index=False)
