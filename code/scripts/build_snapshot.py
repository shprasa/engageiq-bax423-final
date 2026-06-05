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
from engageiq.scrape_github import scrape_github
from engageiq.scrape_gharchive import scrape_gharchive

SOURCES = ("github", "gharchive")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


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


def build_snapshot(
    out_csv: Path,
    synthetic_csv: Path | None,
    github_per_domain: int = 100,
    gharchive_hours: int = 6,
    gharchive_max: int = 2500,
    min_rows: int = 10000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    print("Scraping GitHub Archive (no auth)...")
    gha_df = _normalize(scrape_gharchive(hours_back=gharchive_hours, max_events=gharchive_max))
    gha_df["data_origin"] = "live"
    print(f"  GH Archive rows: {len(gha_df)}")
    if not gha_df.empty:
        parts.append(gha_df)

    print("Scraping GitHub API (GITHUB_TOKEN required)...")
    try:
        gh_df = _normalize(scrape_github(per_domain=github_per_domain))
        gh_df["data_origin"] = "live"
        print(f"  GitHub rows: {len(gh_df)}")
        parts.append(gh_df)
    except Exception as exc:
        print(f"  GitHub scrape skipped: {exc}")

    live_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not live_df.empty:
        live_df = live_df.drop_duplicates(subset=["url"], keep="first")

    synthetic_df = pd.DataFrame()
    if synthetic_csv and synthetic_csv.exists():
        synthetic_df = _normalize(pd.read_csv(synthetic_csv))
        synthetic_df = synthetic_df[synthetic_df["source"].isin(SOURCES)]
        synthetic_df["data_origin"] = "synthetic"
        print(f"Synthetic backup rows: {len(synthetic_df)}")

    if live_df.empty:
        df = synthetic_df
    elif synthetic_df.empty:
        df = live_df
    else:
        live_urls = set(live_df["url"].astype(str))
        synthetic_df = synthetic_df[~synthetic_df["url"].astype(str).isin(live_urls)]
        df = pd.concat([live_df, synthetic_df], ignore_index=True)

    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    # Pad with synthetic rows per source to reach min_rows.
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

    if "data_origin" in df.columns:
        print("Origin mix:", df["data_origin"].value_counts().to_dict())
    print("Domain count:", df["domain"].nunique())
    print("Source mix:", df["source"].value_counts().to_dict())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["data_origin"], errors="ignore").to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EngageIQ offline snapshot from live APIs")
    parser.add_argument("--out", type=str, default="", help="Output CSV path")
    parser.add_argument("--synthetic", type=str, default="", help="Synthetic CSV to merge")
    parser.add_argument("--github-per-domain", type=int, default=100)
    parser.add_argument("--gharchive-hours", type=int, default=6)
    parser.add_argument("--gharchive-max", type=int, default=2500)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick scrape: 2 gharchive hours, github-per-domain=30",
    )
    args = parser.parse_args()

    if args.fast:
        args.gharchive_hours = 2
        args.gharchive_max = 800
        args.github_per_domain = 30
        print("FAST MODE: smaller live sample + synthetic padding to 10k", flush=True)

    paths = get_paths()
    out = Path(args.out) if args.out else paths.snapshot_csv
    synthetic = Path(args.synthetic) if args.synthetic else paths.data_dir / "opportunities_snapshot_synthetic.csv"
    build_snapshot(
        out_csv=out,
        synthetic_csv=synthetic if synthetic.exists() else None,
        github_per_domain=args.github_per_domain,
        gharchive_hours=args.gharchive_hours,
        gharchive_max=args.gharchive_max,
    )


if __name__ == "__main__":
    main()
