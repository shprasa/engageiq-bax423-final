from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd


Source = Literal["github", "reddit", "hackernews"]


@dataclass(frozen=True)
class Opportunity:
    id: int
    source: Source
    domain: str
    title: str
    text: str
    url: str
    community: str
    created_at: datetime
    upvotes: int
    comments: int
    author: str
    lang: str | None
    stars: int | None
    forks: int | None
    issues_open: int | None
    good_first_issue: int | None


class OpportunityStore:
    def __init__(self, duckdb_path: Path):
        self.duckdb_path = duckdb_path

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.duckdb_path))
        con.execute("PRAGMA threads=4;")
        return con

    def ensure_loaded_from_snapshot(self, snapshot_csv: Path, initial_ingest: int = 5000) -> None:
        if not snapshot_csv.exists():
            raise FileNotFoundError(f"Missing snapshot CSV at {snapshot_csv}")

        con = self.connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS opportunities (
                    id BIGINT,
                    source VARCHAR,
                    domain VARCHAR,
                    title VARCHAR,
                    text VARCHAR,
                    url VARCHAR,
                    community VARCHAR,
                    created_at TIMESTAMP,
                    upvotes BIGINT,
                    comments BIGINT,
                    author VARCHAR,
                    lang VARCHAR,
                    stars BIGINT,
                    forks BIGINT,
                    issues_open BIGINT,
                    good_first_issue BIGINT
                );
                """
            )

            n_existing = int(con.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])
            if n_existing == 0:
                con.execute(
                    """
                    INSERT INTO opportunities
                    SELECT * FROM read_csv_auto(?, header=true)
                    LIMIT ?;
                    """,
                    [str(snapshot_csv), int(initial_ingest)],
                )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_id ON opportunities(id);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_domain ON opportunities(domain);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_created ON opportunities(created_at);"
            )
        finally:
            con.close()

    def count(self) -> int:
        con = self.connect()
        try:
            return int(con.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])
        finally:
            con.close()

    def max_id(self) -> int:
        con = self.connect()
        try:
            v = con.execute("SELECT COALESCE(MAX(id), 0) FROM opportunities").fetchone()[0]
            return int(v or 0)
        finally:
            con.close()

    def load_df(self, limit: int | None = None) -> pd.DataFrame:
        con = self.connect()
        try:
            q = "SELECT * FROM opportunities"
            if limit is not None:
                q += f" LIMIT {int(limit)}"
            df = con.execute(q).df()
        finally:
            con.close()

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=False)
        return df

    def ingest_batch(self, batch_df: pd.DataFrame) -> int:
        if batch_df.empty:
            return 0

        con = self.connect()
        try:
            con.register("batch_df", batch_df)
            con.execute(
                """
                INSERT INTO opportunities
                SELECT * FROM batch_df
                WHERE id NOT IN (SELECT id FROM opportunities);
                """
            )
            inserted = int(con.execute("SELECT COUNT(*) FROM batch_df").fetchone()[0])
        finally:
            con.close()
        return inserted

