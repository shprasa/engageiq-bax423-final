## EngageIQ — Technical Brief (template)

Use this template to produce `brief.pdf` (≤4 pages) as required.

### 1) Live URL + Repo
- Live deployment URL: \<fill\>
- Source repo: \<fill\>

### 2) Architecture
- **Ingestion**: offline snapshot `data/opportunities_snapshot.csv` + simulated streaming batches in the UI.
- **Storage**: DuckDB (`data/engageiq.duckdb`) with `opportunities` table.
- **Streaming sketches (BAX-423 technique #1)**: Bloom filter for dedup, Count-Min Sketch for trend counts, HyperLogLog for approx uniques.
- **Embedding retrieval (BAX-423 technique #2)**: TF‑IDF + SVD embeddings + ANN search (NearestNeighbors cosine).
- **Ranking**: multi-stage candidate generation → scoring → rerank with component scores and “Why this?”.
- **Adaptive learning**: Thompson-sampling bandit over domains + benchmarked improvement over 50+ rounds.
- **Analytics**: DuckDB batch queries for trends and time-series volume; charts in Streamlit.

### 3) Pipeline diagram (high level)
Snapshot/Streaming → Dedup (Bloom) → Store (DuckDB) → Embed+ANN → Score+Rank → Feedback → Bandit Update → Dashboard + Brief Export

### 4) Test personas (pass/fail table)
Fill a table covering each of the 6 core capabilities for:
- Sofia (ML student)
- David (DevOps)
- Lina (data journalist)
- Raj (startup founder)

### 5) Benchmark results
- Retrieval/ranking metric: NDCG@10 (reported in app; proxy label).
- Learning benchmark: first-10 vs last-10 NDCG@10 (simulated 60 rounds).
- Technique impact: compare without bandit vs with bandit (include your numbers).

### 6) Limitations + next steps
- Replace synthetic snapshot with real GitHub/Reddit/HN ingestion.
- Upgrade embeddings (SentenceTransformers) and ANN (FAISS).
- Replace proxy labels with real relevance judgments or click logs.

