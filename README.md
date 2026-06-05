# EngageIQ — BAX-423 Final Project

**Student:** Shivneel Prasad  
**Course:** BAX-423 Big Data · UC Davis GSM · Spring 2026

Smart engagement opportunity scorer across GitHub and Hacker News.

## Live demo

- **App:** https://engageiq-bax423-final.streamlit.app
- **Repo:** https://github.com/shprasa/engageiq-bax423-final

## Quick start (graders)

```bash
cd code
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Offline dataset: `data/opportunities_snapshot.csv` (≥10,000 records, 15 domains).

Optional live rebuild (requires `code/.env` with `GITHUB_TOKEN`):

```bash
cd code
py scripts/build_snapshot.py
```

## Submission contents

| Path | Description |
|------|-------------|
| `code/` | Application source + `requirements.txt` |
| `data/` | Offline snapshot CSV + DuckDB (generated at runtime) |
| `brief.pdf` | Technical brief (≤4 pages) |
| `prompts.md` | AI prompt iteration log |

## BAX-423 techniques

1. **Sketching** — Bloom filter, Count-Min Sketch, HyperLogLog  
2. **Embeddings + ANN** — TF-IDF/SVD + cosine NearestNeighbors  
3. **Adaptive learning** — Thompson sampling bandit from user feedback  
4. **Batch analytics** — DuckDB trend queries  
5. **Streaming (optional)** — Kafka scripts in `code/infra/kafka/`

## Scripts

```bash
py scripts/build_snapshot.py      # live + synthetic merge
py scripts/run_benchmarks.py      # persona + learning metrics → data/benchmark_results.json
py scripts/generate_brief.py      # brief.pdf
```

See `DEPLOY.md` for Streamlit Cloud setup.
