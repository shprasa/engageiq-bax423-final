from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb
import pandas as pd


@dataclass(frozen=True)
class TrendSummary:
    by_domain: pd.DataFrame
    by_source: pd.DataFrame
    volume_over_time: pd.DataFrame


def compute_trends_from_df(df: pd.DataFrame, days: int = 30) -> TrendSummary:
    since = datetime.now() - timedelta(days=int(days))
    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work = work[work["created_at"] >= since]

    by_domain = (
        work.groupby("domain", as_index=False)
        .agg(n=("id", "count"), avg_upvotes=("upvotes", "mean"), avg_comments=("comments", "mean"))
        .sort_values("n", ascending=False)
    )
    by_source = (
        work.groupby("source", as_index=False)
        .agg(n=("id", "count"), avg_upvotes=("upvotes", "mean"), avg_comments=("comments", "mean"))
        .sort_values("n", ascending=False)
    )
    work["day"] = work["created_at"].dt.floor("D")
    volume_over_time = work.groupby("day", as_index=False).agg(n=("id", "count")).sort_values("day")
    return TrendSummary(by_domain=by_domain, by_source=by_source, volume_over_time=volume_over_time)


def compute_wow_domain_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Week-over-week domain volume change for trend analytics (Lina persona)."""
    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    now = pd.Timestamp.now()
    this_week = work[work["created_at"] >= (now - pd.Timedelta(days=7))]
    last_week = work[(work["created_at"] >= (now - pd.Timedelta(days=14))) & (work["created_at"] < (now - pd.Timedelta(days=7)))]
    cur = this_week.groupby("domain").size().rename("this_week")
    prev = last_week.groupby("domain").size().rename("last_week")
    wow = pd.concat([cur, prev], axis=1).fillna(0).reset_index()
    wow["delta"] = wow["this_week"] - wow["last_week"]
    wow["pct_change"] = wow["delta"] / (wow["last_week"] + 1)
    return wow.sort_values("delta", ascending=False)


def compute_trends(duckdb_path: str, days: int = 30) -> TrendSummary:
    since = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%dT%H:%M:%S")
    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        by_domain = con.execute(
            """
            SELECT domain,
                   COUNT(*) AS n,
                   AVG(COALESCE(upvotes,0)) AS avg_upvotes,
                   AVG(COALESCE(comments,0)) AS avg_comments
            FROM opportunities
            WHERE created_at >= ?
            GROUP BY domain
            ORDER BY n DESC
            """,
            [since],
        ).df()

        by_source = con.execute(
            """
            SELECT source,
                   COUNT(*) AS n,
                   AVG(COALESCE(upvotes,0)) AS avg_upvotes,
                   AVG(COALESCE(comments,0)) AS avg_comments
            FROM opportunities
            WHERE created_at >= ?
            GROUP BY source
            ORDER BY n DESC
            """,
            [since],
        ).df()

        volume_over_time = con.execute(
            """
            SELECT date_trunc('day', created_at) AS day,
                   COUNT(*) AS n
            FROM opportunities
            WHERE created_at >= ?
            GROUP BY 1
            ORDER BY 1
            """,
            [since],
        ).df()
    finally:
        con.close()

    return TrendSummary(by_domain=by_domain, by_source=by_source, volume_over_time=volume_over_time)

