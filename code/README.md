# EngageIQ — Code

Run from this directory:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Architecture

- **Data:** `data/opportunities_snapshot.csv` (10k rows, 15 domains) loaded into DuckDB with URL dedup
- **Recommendation:** TF-IDF/SVD → cosine NearestNeighbors → multi-stage rerank
- **RL:** Thompson-sampling bandit over domains; feedback from engage/bookmark/skip
- **UI:** Discover, Bookmarks, Activity, Analytics tabs

## Modules

| Module | Role |
|--------|------|
| `embedding.py` | TF-IDF/SVD index + ANN query |
| `ranking.py` | Candidate generation, scoring, NDCG@10 |
| `reinforcement_learning.py` | Contextual bandit + 60-round benchmark |
| `analytics.py` | DuckDB trend queries |
| `data.py` | DuckDB store + multi-source ingest |
| `domains.py` | 15 required technical domains |
| `ui.py` | Streamlit cards and session state |

## Optional scripts

- `build_snapshot.py` — refresh live GitHub/HN data (needs API tokens in `.env`)
- `run_benchmarks.py` — regenerate `data/benchmark_results.json`
- `user_test_loop.py` — automated rubric checks
