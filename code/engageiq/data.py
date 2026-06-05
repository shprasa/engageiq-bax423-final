from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from .data_utils import is_live_url, live_mask, sort_live_first


Source = Literal["github", "hackernews"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS opportunities (
    id BIGINT, source VARCHAR, domain VARCHAR, title VARCHAR,
    text VARCHAR, url VARCHAR, community VARCHAR,
    created_at TIMESTAMP, upvotes BIGINT, comments BIGINT,
    author VARCHAR, lang VARCHAR, stars BIGINT, forks BIGINT,
    issues_open BIGINT, good_first_issue BIGINT
);
"""


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
    def __init__(self, duckdb_path: Path, snapshot_csv: Path | None = None):
        self.duckdb_path = duckdb_path
        self.snapshot_csv = snapshot_csv
        self._memory_df: pd.DataFrame | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        try:
            self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(str(self.duckdb_path))
        except Exception:
            con = duckdb.connect(":memory:")
        con.execute("PRAGMA threads=4;")
        return con

    def _read_snapshot(self, snapshot_csv: Path) -> pd.DataFrame:
        df = pd.read_csv(snapshot_csv)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = sort_live_first(df)
        if int(live_mask(df).sum()) == 0 and self.snapshot_csv:
            live_path = snapshot_csv.parent / "live_opportunities.csv"
            if live_path.exists() and live_path != snapshot_csv:
                live_df = pd.read_csv(live_path)
                live_df["created_at"] = pd.to_datetime(live_df["created_at"], errors="coerce")
                df = sort_live_first(pd.concat([live_df, df], ignore_index=True).drop_duplicates(subset=["url"]))
        return df

    def _db_live_count(self, con: duckdb.DuckDBPyConnection) -> int:
        try:
            return int(
                con.execute(
                    "SELECT COUNT(*) FROM opportunities WHERE url NOT LIKE '%example.local%'"
                ).fetchone()[0]
            )
        except Exception:
            return 0

    def _needs_reload(self, snapshot_csv: Path, n_existing: int) -> bool:
        if n_existing == 0:
            return True
        csv_df = self._read_snapshot(snapshot_csv)
        csv_live = int(live_mask(csv_df).sum())
        csv_total = len(csv_df)
        if csv_live > 0:
            try:
                con = self.connect()
                try:
                    db_live = self._db_live_count(con)
                finally:
                    con.close()
                if db_live == 0:
                    return True
            except Exception:
                return True
        if n_existing < min(csv_total, csv_total - 500):
            return True
        return False

    def _reload_from_csv(self, snapshot_csv: Path) -> int:
        df = self._read_snapshot(snapshot_csv)
        try:
            con = self.connect()
            try:
                con.execute("DROP TABLE IF EXISTS opportunities")
                con.execute(SCHEMA)
                con.register("snap_df", df)
                con.execute("INSERT INTO opportunities SELECT * FROM snap_df")
                return len(df)
            finally:
                con.close()
        except Exception:
            self._memory_df = df
            return len(df)

    def ensure_loaded_from_snapshot(self, snapshot_csv: Path, initial_ingest: int = 100000) -> None:
        self.snapshot_csv = snapshot_csv
        if not snapshot_csv.exists():
            alt = snapshot_csv.parent / "live_opportunities.csv"
            if alt.exists():
                snapshot_csv = alt
                self.snapshot_csv = alt
            else:
                raise FileNotFoundError(f"Missing snapshot CSV at {snapshot_csv}")

        try:
            con = self.connect()
            try:
                con.execute(SCHEMA)
                n_existing = int(con.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])
            finally:
                con.close()

            if self._needs_reload(snapshot_csv, n_existing):
                self._reload_from_csv(snapshot_csv)
            elif n_existing == 0:
                df = self._read_snapshot(snapshot_csv)
                if len(df) > initial_ingest:
                    df = df.head(int(initial_ingest))
                try:
                    con = self.connect()
                    try:
                        con.register("snap_df", df)
                        con.execute("INSERT INTO opportunities SELECT * FROM snap_df")
                    finally:
                        con.close()
                except Exception:
                    self._memory_df = df
        except Exception:
            self._memory_df = self._read_snapshot(snapshot_csv)

    def count(self) -> int:
        try:
            con = self.connect()
            try:
                return int(con.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])
            finally:
                con.close()
        except Exception:
            return len(self._load_csv_fallback())

    def max_id(self) -> int:
        try:
            con = self.connect()
            try:
                v = con.execute("SELECT COALESCE(MAX(id), 0) FROM opportunities").fetchone()[0]
                return int(v or 0)
            finally:
                con.close()
        except Exception:
            df = self._load_csv_fallback()
            return int(df["id"].max()) if not df.empty else 0

    def _load_csv_fallback(self) -> pd.DataFrame:
        if self._memory_df is not None:
            return self._memory_df
        if self.snapshot_csv and self.snapshot_csv.exists():
            self._memory_df = self._read_snapshot(self.snapshot_csv)
            return self._memory_df
        raise FileNotFoundError(f"No data at {self.snapshot_csv}")

    def load_df(self, limit: int | None = None, live_only: bool = False) -> pd.DataFrame:
        try:
            con = self.connect()
            try:
                q = "SELECT * FROM opportunities"
                if live_only:
                    q += " WHERE url NOT LIKE '%example.local%'"
                q += " ORDER BY CASE WHEN url LIKE '%example.local%' THEN 1 ELSE 0 END, id"
                if limit is not None:
                    q += f" LIMIT {int(limit)}"
                df = con.execute(q).df()
            finally:
                con.close()
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=False)
            return df
        except Exception:
            df = self._load_csv_fallback().copy()
            if live_only:
                df = df[live_mask(df)]
            if limit is not None:
                df = df.head(int(limit))
            return df

    def ingest_batch(self, batch_df: pd.DataFrame) -> int:
        if batch_df.empty:
            return 0
        try:
            con = self.connect()
            try:
                con.register("batch_df", batch_df)
                before = int(con.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])
                con.execute(
                    """
                    INSERT INTO opportunities
                    SELECT * FROM batch_df
                    WHERE id NOT IN (SELECT id FROM opportunities)
                      AND url NOT IN (SELECT url FROM opportunities);
                    """
                )
                after = int(con.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])
                return after - before
            finally:
                con.close()
        except Exception:
            base = self._load_csv_fallback()
            merged = pd.concat([base, batch_df], ignore_index=True).drop_duplicates(subset=["url"])
            self._memory_df = sort_live_first(merged)
            return len(batch_df)
