from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer

from engageiq.config import get_paths
from engageiq.data import OpportunityStore


def main() -> None:
    paths = get_paths()
    store = OpportunityStore(paths.duckdb_path)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=0)

    consumer = KafkaConsumer(
        "engageiq.opportunities",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )

    buf: list[dict] = []
    for msg in consumer:
        buf.append(msg.value)
        if len(buf) >= 500:
            batch = pd.DataFrame(buf)
            store.ingest_batch(batch)
            print("ingested", len(buf), "total", store.count())
            buf = []


if __name__ == "__main__":
    main()

