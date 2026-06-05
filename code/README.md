# EngageIQ (Prototype) — BAX-423 Final Project

This folder contains a runnable prototype for **EngageIQ — Smart Engagement Opportunity Scorer**.

## Quick start

1) Create a virtualenv and install deps:

```bash
py -m pip install -r requirements.txt
```

2) **GitHub API (required for live scrape):**

```bash
copy .env.example .env
```

Edit `.env` and set `GITHUB_TOKEN=...` (create at https://github.com/settings/tokens).

**Never paste tokens in chat or commit `.env` to GitHub.**

3) Build live dataset (optional; replaces synthetic snapshot):

```bash
cd EngageIQ_Final\code
py scripts\build_snapshot.py
```

4) Run the app:

```bash
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
- **Adaptive learning**: per-domain Thompson sampling + personalization vector updated from feedback; includes 50+ simulated rounds.
- **Batch analytics**: DuckDB queries for trends, volumes, and distributions.
- **Dashboard + brief export**: Streamlit UI + CSV export (and optional PDF).

## Files

- `app.py`: Streamlit UI
- `engageiq/`: pipeline modules (data store, sketches, embedding, ranking, learning, analytics, export)
- `scripts/`: small utilities (optional)

