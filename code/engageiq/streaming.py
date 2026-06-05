"""Streaming ingest: in-app queue + optional Kafka (localhost) with Bloom dedup."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .sketches import BloomFilter


@dataclass
class StreamMetrics:
    produced: int = 0
    deduped: int = 0
    ingested: int = 0
    kafka_sent: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "produced": self.produced,
            "deduped": self.deduped,
            "ingested": self.ingested,
            "kafka_sent": self.kafka_sent,
        }


@dataclass
class OpportunityStream:
    """Simulates a real-time stream (Kafka-compatible) with deduplication."""

    bloom: BloomFilter
    url_bloom: BloomFilter | None = None
    queue: deque[dict[str, Any]] = field(default_factory=deque)
    metrics: StreamMetrics = field(default_factory=StreamMetrics)

    def __post_init__(self) -> None:
        if self.url_bloom is None:
            self.url_bloom = BloomFilter(capacity=max(50000, getattr(self.bloom, "m", 25000)), fp_rate=0.01)

    def produce_rows(self, df: pd.DataFrame) -> int:
        n = 0
        for _, row in df.iterrows():
            self.queue.append(row.to_dict())
            n += 1
        self.metrics.produced += n
        return n

    def _is_duplicate(self, row: dict[str, Any]) -> bool:
        rid = str(row.get("id", ""))
        url = str(row.get("url", ""))
        if rid and rid in self.bloom:
            return True
        if url and url in self.url_bloom:
            return True
        return False

    def _mark_seen(self, row: dict[str, Any]) -> None:
        rid = str(row.get("id", ""))
        url = str(row.get("url", ""))
        if rid:
            self.bloom.add(rid)
        if url:
            self.url_bloom.add(url)

    def consume(
        self,
        max_items: int,
        ingest_fn,
        sketch_hooks: list | None = None,
    ) -> tuple[int, int]:
        """Consume up to max_items from queue; dedup; ingest via callback."""
        buf: list[dict[str, Any]] = []
        deduped = 0
        while self.queue and len(buf) < max_items:
            row = self.queue.popleft()
            if self._is_duplicate(row):
                deduped += 1
                continue
            buf.append(row)
            self._mark_seen(row)

        inserted = 0
        if buf:
            batch = pd.DataFrame(buf)
            inserted = int(ingest_fn(batch))
            if sketch_hooks:
                for hook in sketch_hooks:
                    hook(batch)

        self.metrics.deduped += deduped
        self.metrics.ingested += inserted
        return inserted, deduped

    def pending(self) -> int:
        return len(self.queue)


def try_kafka_publish(rows: list[dict[str, Any]], bootstrap: str = "localhost:9092") -> int:
    """Optional: publish to Kafka if broker is running. Returns count sent or 0."""
    if not rows:
        return 0
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            request_timeout_ms=3000,
            api_version_auto_timeout_ms=3000,
        )
        topic = "engageiq.opportunities"
        for row in rows:
            producer.send(topic, row)
        producer.flush(timeout=5)
        producer.close()
        return len(rows)
    except Exception:
        return 0
