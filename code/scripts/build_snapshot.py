from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.data_utils import live_mask
from engageiq.scrape_github import scrape_github
from engageiq.scrape_gharchive import scrape_gharchive
from engageiq.snapshot_ops import write_snapshot_everywhere

SOURCES = ("github", "gharchive")

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


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in SCHEMA_COLS:
        if c not in df.columns:
            df[c] = ""
    out = df[SCHEMA_COLS].copy()
    for num_col in ("upvotes", "comments", "stars", "forks", "issues_open", "good_first_issue"):
        out[num_col] = pd.to_numeric(out[num_col], errors="coerce").fillna(0)
    return out[live_mask(out)].drop_duplicates(subset=["url"], keep="first")


def _merge_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    live_parts = [_normalize(p) for p in parts if p is not None and not p.empty]
    if not live_parts:
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.concat(live_parts, ignore_index=True)
    return df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)


def build_snapshot(
    out_csv: Path,
    *,
    github_per_domain: int = 400,
    gharchive_hours: int = 168,
    gharchive_max: int = 0,
    min_rows: int = 10_000,
) -> pd.DataFrame:
    """Build an offline snapshot from live API scrapes only (no synthetic padding)."""
    parts: list[pd.DataFrame] = []

    if out_csv.exists():
        existing = _normalize(pd.read_csv(out_csv))
        if not existing.empty:
            parts.append(existing)
            print(f"Loaded {len(existing):,} existing live rows from {out_csv.name}")

    df = _merge_parts(parts)

    if len(df) < min_rows:
        need = min_rows - len(df)
        print(f"Scraping GitHub Archive until >= {min_rows:,} total rows (need ~{need:,} more)...")
        gha_df = scrape_gharchive(
            hours_back=max(gharchive_hours, 168),
            max_events=gharchive_max,
            target_rows=max(need + 500, min_rows),
        )
        print(f"  GH Archive live rows: {len(gha_df):,}")
        if not gha_df.empty:
            parts.append(gha_df)
        df = _merge_parts(parts)
        print(f"Combined live total after GH Archive: {len(df):,} rows")

    if len(df) < min_rows:
        print("Scraping GitHub API (GITHUB_TOKEN required)...")
        try:
            gh_df = scrape_github(per_domain=github_per_domain)
            print(f"  GitHub API live rows: {len(gh_df):,}")
            if not gh_df.empty:
                parts.append(gh_df)
        except Exception as exc:
            print(f"  GitHub scrape failed: {exc}")
        df = _merge_parts(parts)
        print(f"Combined live total after GitHub API: {len(df):,} rows")

    if len(df) < min_rows:
        print("Extending GH Archive to 720 hours...")
        gha_df = scrape_gharchive(hours_back=720, max_events=gharchive_max, target_rows=min_rows)
        if not gha_df.empty:
            parts.append(gha_df)
        df = _merge_parts(parts)

    if len(df) < min_rows:
        raise RuntimeError(
            f"Collected only {len(df):,} live rows (need >= {min_rows:,}). "
            "Increase --gharchive-hours, --github-per-domain, or check GITHUB_TOKEN/network."
        )

    github_n = int((df["source"].astype(str).str.lower() == "github").sum())
    if github_n < 3000:
        print(f"Adding GitHub API rows (currently {github_n:,}; target >= 3,000)...")
        try:
            gh_df = scrape_github(per_domain=github_per_domain)
            if not gh_df.empty:
                parts.append(gh_df)
                df = _merge_parts(parts)
                github_n = int((df["source"].astype(str).str.lower() == "github").sum())
                print(f"Combined live total after GitHub API top-up: {len(df):,} rows ({github_n:,} GitHub API)")
        except Exception as exc:
            print(f"  GitHub top-up skipped: {exc}")

    df["id"] = range(1, len(df) + 1)
    print("Domain count:", df["domain"].nunique())
    print("Source mix:", df["source"].value_counts().to_dict())
    print("Live rows only:", int(live_mask(df).sum()), f"/ {len(df)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    paths = get_paths()
    write_snapshot_everywhere(df, paths)
    print(f"Wrote {len(df):,} live rows -> {out_csv} (+ synced code/data/ for Streamlit Cloud)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EngageIQ offline snapshot (live API data only)")
    parser.add_argument("--out", type=str, default="", help="Output CSV path")
    parser.add_argument("--github-per-domain", type=int, default=400)
    parser.add_argument("--gharchive-hours", type=int, default=168, help="Max GH Archive history window in hours")
    parser.add_argument(
        "--gharchive-max",
        type=int,
        default=0,
        help="Max GH Archive events (0 = no limit)",
    )
    parser.add_argument("--min-rows", type=int, default=10_000)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick scrape for dev (still live-only; may fail min_rows check)",
    )
    args = parser.parse_args()

    if args.fast:
        args.gharchive_hours = 48
        args.gharchive_max = 5000
        args.github_per_domain = 80
        args.min_rows = 500
        print("FAST MODE: smaller live scrape only", flush=True)

    paths = get_paths()
    out = Path(args.out) if args.out else paths.snapshot_csv
    df = build_snapshot(
        out_csv=out,
        github_per_domain=args.github_per_domain,
        gharchive_hours=args.gharchive_hours,
        gharchive_max=args.gharchive_max,
        min_rows=args.min_rows,
    )

    from engageiq.benchmark_ops import benchmark_summary, run_and_write_benchmarks

    print("Running persona benchmarks on refreshed dataset…", flush=True)
    payload = run_and_write_benchmarks(df, paths)
    print(benchmark_summary(payload), flush=True)


if __name__ == "__main__":
    main()
