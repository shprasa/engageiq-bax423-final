"""Minimal batch streaming queue with URL dedup (capability 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class StreamMetrics:
    produced: int = 0
    ingested: int = 0
    deduped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"produced": self.produced, "ingested": self.ingested, "deduped": self.deduped}


class OpportunityStream:
    """Simulated streaming pipeline: produce batches → dedup by URL → ingest."""

    def __init__(self) -> None:
        self._queue: list[dict] = []
        self._seen_urls: set[str] = set()
        self.metrics = StreamMetrics()

    def pending(self) -> int:
        return len(self._queue)

    def produce_rows(self, batch_df: pd.DataFrame) -> int:
        added = 0
        for row in batch_df.to_dict(orient="records"):
            url = str(row.get("url") or "")
            if not url or url in self._seen_urls:
                self.metrics.deduped += 1
                continue
            self._seen_urls.add(url)
            self._queue.append(row)
            added += 1
        self.metrics.produced += added
        return added

    def consume(
        self,
        max_items: int,
        ingest_fn: Callable[[pd.DataFrame], int],
    ) -> tuple[int, int]:
        if not self._queue:
            return 0, 0
        take = self._queue[: int(max_items)]
        self._queue = self._queue[int(max_items) :]
        batch = pd.DataFrame(take)
        inserted = int(ingest_fn(batch))
        self.metrics.ingested += inserted
        skipped = len(batch) - inserted
        self.metrics.deduped += skipped
        return inserted, skipped
