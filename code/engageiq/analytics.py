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

