# EngageIQ — BAX-423 Final Project

**Student:** Shivneel Prasad  
**Course:** BAX-423 Big Data · UC Davis GSM · Spring 2026

Smart engagement opportunity scorer across GitHub API and GitHub Archive.

## Live demo

- **App:** https://engageiq-bax423-final.streamlit.app
- **Repo:** https://github.com/shprasa/engageiq-bax423-final

## Quick start (graders)

```bash
cd code
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Offline dataset: `data/opportunities_snapshot.csv` (≥10,000 records, **15 domains**).

Optional live rebuild (requires `code/.env` with `GITHUB_TOKEN`):

```bash
cd code
py scripts/build_snapshot.py
```

## Submission contents

| Path | Description |
|------|-------------|
| `code/` | Application source + `requirements.txt` |
| `data/` | Offline snapshot CSV + benchmark results |
| `brief.docx` | Technical brief (≤4 pages) |
| `prompts.md` | Key development prompts (v3) |

## BAX-423 techniques (2 implemented)

1. **Recommendation system** — TF-IDF/SVD embeddings, ANN retrieval, multi-stage ranking, NDCG@10
2. **Reinforcement learning** — Thompson-sampling contextual bandit from engage/bookmark/skip feedback

## Six core capabilities

1. Multi-source ingest + dedup (GitHub API + GitHub Archive → DuckDB)
2. Embeddings + ANN retrieval
3. Multi-stage ranking + NDCG@10
4. Adaptive learning / RL (50+ rounds)
5. Batch analytics + trends
6. Dashboard + brief export

## Scripts

```bash
py scripts/run_benchmarks.py      # persona + RL metrics → data/benchmark_results.json
py scripts/user_test_loop.py      # rubric validation
py scripts/package_submission.py  # Canvas ZIP
```
