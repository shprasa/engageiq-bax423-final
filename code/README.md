# EngageIQ (Prototype) — BAX-423 Final Project

This folder contains a runnable prototype for **EngageIQ — Smart Engagement Opportunity Scorer**.

## Quick start (graders — single command)

```bash
cd code
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app will:
- Load the offline snapshot from `../data/opportunities_snapshot.csv`
- Build a local DuckDB store at `../data/engageiq.duckdb`
- Provide ranked “engagement opportunities” with explanations, feedback learning, trend analytics, and a downloadable weekly brief (CSV; PDF optional).

## What’s implemented (mapped to the brief)

- **Multi-source ingestion + streaming**: batch ingest from offline snapshot + simulated streaming batches.
- **Dedup + sketches**: Bloom filter (dedup), Count-Min Sketch (trend counts), HyperLogLog (approx unique authors).
- **Embeddings + similarity retrieval**: TF‑IDF + SVD embeddings (SentenceTransformers optional) + ANN via `NearestNeighbors`.
- **Engagement scoring + multi-stage ranking**: candidate generation → scoring → reranking, metric reported (NDCG@K).
- **Reinforcement learning**: contextual multi-armed bandit (`engageiq/reinforcement_learning.py`) — state=user profile, actions=domain arms, rewards from engage (+1.0) / bookmark (+0.85) / skip (0). Thompson sampling policy updates ranking; **RL Policy** tab shows Q-values and cumulative reward curve.
- **Batch analytics**: DuckDB queries for trends, volumes, and distributions.
- **Dashboard + brief export**: Streamlit UI + CSV export (and optional PDF).

## Files

- `app.py`: Streamlit UI
- `engageiq/`: pipeline modules (data store, sketches, embedding, ranking, learning, analytics, export)
- `scripts/`: small utilities (optional)

