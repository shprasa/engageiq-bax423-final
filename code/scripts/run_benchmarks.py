from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.persona_eval import PERSONAS, evaluate_personas, learning_benchmark


def main() -> None:
    paths = get_paths()
    store = OpportunityStore(paths.duckdb_path)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=10000)
    df = store.load_df()

    persona_results = evaluate_personas(df)
    learning = learning_benchmark(df, PERSONAS["Sofia (ML Student / Portfolio Builder)"], rounds=60)

    out = paths.project_root / "data" / "benchmark_results.json"
    payload = {
        "personas": [r.__dict__ for r in persona_results],
        "learning_benchmark": learning,
        "dataset_rows": len(df),
        "domains": int(df["domain"].nunique()),
        "sources": df["source"].value_counts().to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
