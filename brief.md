## EngageIQ — Technical Brief (template)

### 1) Live URL + Repo
- Live deployment URL: https://engageiq-bax423-final.streamlit.app/
- Source repo: https://github.com/shprasa/engageiq-bax423-final

### 2) Architecture
- **Ingestion**: GitHub + Hacker News snapshot → URL dedup → DuckDB.
- **Recommendation (technique #1)**: TF‑IDF + SVD embeddings + ANN search → multi-stage ranking + NDCG@10.
- **Reinforcement learning (technique #2)**: Thompson-sampling bandit over domains; rewards from engage/bookmark/skip.
- **Analytics**: DuckDB batch queries for trends; charts in Streamlit.
- **Export**: CSV/PDF weekly brief from ranked results + trends.

### 3) Pipeline diagram (high level)
Snapshot → Dedup → DuckDB → Embed+ANN → Score+Rank → Feedback → Bandit Update → Dashboard + Brief Export

### 4) Six core capabilities
1. Multi-source ingest + dedup  
2. Embeddings + ANN retrieval  
3. Scoring + multi-stage ranking  
4. Adaptive learning / RL (50+ rounds)  
5. Batch analytics + trends  
6. Dashboard + brief export  

### 5) Test personas
Sofia, David, Lina, Raj — pass/fail table in `brief.pdf`.

### 6) Benchmark results
- NDCG@10 per persona
- 60-round RL vs no-RL baseline

### 7) Limitations
- Synthetic backup rows pad dataset to 10k for offline grading.
- TF-IDF/SVD instead of SentenceTransformers/FAISS for deploy reliability.
