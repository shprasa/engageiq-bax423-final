from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.scrape_github import scrape_github
from engageiq.scrape_hn import scrape_hackernews
from engageiq.scrape_reddit import scrape_reddit


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


def build_snapshot(
    out_csv: Path,
    synthetic_csv: Path | None,
    github_per_domain: int = 100,
    hn_max: int = 2000,
    min_rows: int = 10000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    print("Scraping Hacker News (no auth)...")
    hn_df = _normalize(scrape_hackernews(max_stories=hn_max))
    hn_df["data_origin"] = "live"
    print(f"  HN rows: {len(hn_df)}")
    parts.append(hn_df)

    print("Scraping GitHub (GITHUB_TOKEN required)...")
    gh_df = _normalize(scrape_github(per_domain=github_per_domain))
    gh_df["data_origin"] = "live"
    print(f"  GitHub rows: {len(gh_df)}")
    parts.append(gh_df)

    print("Scraping Reddit (optional)...")
    rd_df = _normalize(scrape_reddit(per_sub_limit=50))
    if not rd_df.empty:
        rd_df["data_origin"] = "live"
        print(f"  Reddit rows: {len(rd_df)}")
        parts.append(rd_df)
    else:
        print("  Reddit rows: 0 (skipped — no creds)")

    live_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not live_df.empty:
        live_df = live_df.drop_duplicates(subset=["url"], keep="first")

    synthetic_df = pd.DataFrame()
    if synthetic_csv and synthetic_csv.exists():
        synthetic_df = _normalize(pd.read_csv(synthetic_csv))
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
    df["id"] = range(1, len(df) + 1)

    if "data_origin" in df.columns:
        print("Origin mix:", df["data_origin"].value_counts().to_dict())
    print("Domain count:", df["domain"].nunique())
    print("Source mix:", df["source"].value_counts().to_dict())

    if len(df) < min_rows:
        print(f"WARNING: {len(df)} rows (target {min_rows}).")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["data_origin"], errors="ignore").to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EngageIQ offline snapshot from live APIs")
    parser.add_argument("--out", type=str, default="", help="Output CSV path")
    parser.add_argument("--synthetic", type=str, default="", help="Synthetic CSV to merge")
    parser.add_argument("--github-per-domain", type=int, default=100)
    parser.add_argument("--hn-max", type=int, default=2000)
    args = parser.parse_args()

    paths = get_paths()
    out = Path(args.out) if args.out else paths.snapshot_csv
    synthetic = Path(args.synthetic) if args.synthetic else paths.data_dir / "opportunities_snapshot_synthetic.csv"
    build_snapshot(
        out_csv=out,
        synthetic_csv=synthetic,
        github_per_domain=args.github_per_domain,
        hn_max=args.hn_max,
    )


if __name__ == "__main__":
    main()
