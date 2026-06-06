"""Replace Hacker News rows with GitHub Archive in snapshot CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.domains import DOMAINS
from engageiq.scrape_gharchive import scrape_gharchive

SOURCES = ("github", "gharchive")


def _transform_offline_hn(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["source"].astype(str).eq("hackernews")
    if not mask.any():
        return df
    out = df.copy()
    out.loc[mask, "source"] = "gharchive"
    out.loc[mask, "url"] = out.loc[mask, "url"].astype(str).str.replace(
        "example.local/hackernews/", "example.local/gharchive/", regex=False
    )
    out.loc[mask, "community"] = out.loc[mask, "community"].astype(str).str.replace(
        "news.ycombinator.com", "github.com/gharchive", regex=False
    )
    out.loc[mask, "text"] = (
        out.loc[mask, "text"]
        .astype(str)
        .str.replace("Source=hackernews", "Source=gharchive", regex=False)
    )
    out.loc[mask, "title"] = out.loc[mask, "title"].astype(str).str.replace(
        " - Opportunity ", " - GH Archive event ", regex=False
    )
    return out


def _generate_synthetic_gharchive(existing: pd.DataFrame, target: int) -> pd.DataFrame:
    need = max(0, target - int((existing["source"] == "gharchive").sum()))
    if need == 0:
        return pd.DataFrame()
    rows: list[dict] = []
    start_id = int(existing["id"].max()) + 1 if len(existing) else 1
    for i in range(need):
        domain = DOMAINS[i % len(DOMAINS)]
        n = i + 1
        rows.append(
            {
                "id": start_id + i,
                "source": "gharchive",
                "domain": domain,
                "title": f"{domain} - GH Archive event {n}",
                "text": f"Looking for insights on {domain}. Source=gharchive. Event seed {173651 + n}.",
                "url": f"https://example.local/gharchive/{n}",
                "community": f"github.com/archive-{domain.replace('/', '-')}",
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
    return pd.DataFrame(rows)


def migrate_snapshot(
    snapshot_path: Path,
    *,
    hours_back: int = 6,
    max_live: int = 2500,
    min_rows: int = 10000,
    skip_live_scrape: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(snapshot_path)
    df = df[df["source"].isin(["github", "hackernews", "gharchive"])].copy()

    # Drop live HN; keep GitHub + offline rows (transform HN offline -> gharchive).
    live_hn = df["source"].eq("hackernews") & ~df["url"].astype(str).str.contains("example.local")
    df = df[~live_hn].copy()
    df = _transform_offline_hn(df)

    live_gha = pd.DataFrame()
    if not skip_live_scrape:
        try:
            live_gha = scrape_gharchive(hours_back=hours_back, max_events=max_live)
            if not live_gha.empty:
                live_gha = live_gha.drop_duplicates(subset=["url"], keep="first")
        except Exception as exc:
            print(f"WARNING: live GH Archive scrape failed: {exc}")

    if not live_gha.empty:
        df = pd.concat([live_gha, df], ignore_index=True)

    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    gharchive_target = max(int((df["source"] == "github").sum()), 4000)
    extra = _generate_synthetic_gharchive(df, gharchive_target)
    if not extra.empty:
        df = pd.concat([df, extra], ignore_index=True)

    while len(df) < min_rows:
        extra = _generate_synthetic_gharchive(df, len(df) + (min_rows - len(df)))
        if extra.empty:
            break
        df = pd.concat([df, extra], ignore_index=True)

    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    df = df[df["source"].isin(SOURCES)]

    print("Source mix:", df["source"].value_counts().to_dict())
    print("Domains:", df["domain"].nunique())
    print("Rows:", len(df))
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate dataset from Hacker News to GitHub Archive")
    parser.add_argument("--hours-back", type=int, default=6)
    parser.add_argument("--max-live", type=int, default=2500)
    parser.add_argument("--skip-live-scrape", action="store_true")
    args = parser.parse_args()

    paths = get_paths()
    targets = [
        paths.snapshot_csv,
        paths.code_dir / "data" / "opportunities_snapshot.csv",
        paths.project_root / "data" / "opportunities_snapshot.csv",
    ]
    seen: set[Path] = set()
    for path in targets:
        path = path.resolve()
        if path in seen or not path.parent.exists():
            continue
        seen.add(path)
        print(f"Migrating {path} ...")
        migrate_snapshot(
            path,
            hours_back=args.hours_back,
            max_live=args.max_live,
            skip_live_scrape=args.skip_live_scrape,
        )

    # Refresh live-only CSV from migrated snapshot.
    snap = paths.snapshot_csv
    if snap.exists():
        df = pd.read_csv(snap)
        live = df[~df["url"].astype(str).str.contains("example.local", na=False)]
        live_path = paths.code_dir / "data" / "live_opportunities.csv"
        live.to_csv(live_path, index=False)
        print(f"Wrote {len(live)} live rows -> {live_path}")


if __name__ == "__main__":
    main()
