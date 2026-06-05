from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


def main() -> None:
    snapshot = Path(__file__).resolve().parents[2] / "data" / "opportunities_snapshot.csv"
    df = pd.read_csv(snapshot).sort_values("created_at")

    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    topic = "engageiq.opportunities"
    for _, r in df.iterrows():
        producer.send(topic, r.to_dict())
        time.sleep(0.01)

    producer.flush()
    print("Done producing.")


if __name__ == "__main__":
    main()

