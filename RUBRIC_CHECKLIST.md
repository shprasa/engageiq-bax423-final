# EngageIQ — Rubric Compliance Checklist

**Student:** Shivneel Prasad · **Live URL:** https://engageiq-bax423-final.streamlit.app/

## Submission package (`Prasad_Shivneel_BAX423_Final.zip`)

| Item | Required | Status |
|------|----------|--------|
| `code/` + `requirements.txt` + README | Yes | ✅ |
| Single-command run | Yes | ✅ `py -m streamlit run app.py` |
| `data/` snapshot ≥10,000 | Yes | ✅ 11,964 rows |
| `brief.pdf` ≤4 pages | Yes | ✅ |
| `prompts.md` | Yes | ✅ |
| Live public URL | Yes | ✅ |

## 6 core capabilities (missing any = max 60/100)

| # | Capability | Status | Evidence |
|---|------------|--------|----------|
| 1 | Multi-source ingest + streaming + dedup | ✅ | GitHub+HN scrapers, simulated ingest, Bloom filter |
| 2 | Embeddings + ANN retrieval | ✅ | TF-IDF/SVD + NearestNeighbors |
| 3 | Multi-stage ranking + metric | ✅ | 4-component scorer + NDCG@10 |
| 4 | Adaptive learning 50+ rounds | ✅ | Thompson bandit + 60-round simulation |
| 5 | Batch analytics + trends | ✅ | DuckDB/pandas + WoW chart |
| 6 | Dashboard + brief export | ✅ | Streamlit + CSV/PDF + suggested actions |

## Rubric dimensions (100 pts)

| Dimension | Pts | Status |
|-----------|-----|--------|
| Data Pipeline | 15 | ✅ Streaming, dedup, 11,964 records, 2 live API sources |
| Matching & Ranking | 20 | ✅ Embeddings, multi-stage rank, NDCG@10 |
| Adaptive Learning & Techniques | 15 | ✅ 3 techniques benchmarked in brief |
| Hosting & Deployment | 20 | ✅ Live Streamlit, explain feature works |
| Brief & Demo | 30 | ✅ brief.pdf + DEMO.md; demo on you Saturday |

## Data sources

| Source | Required | Status |
|--------|----------|--------|
| GitHub API | ≥2 of 4 | ✅ 928 live rows |
| Hacker News API | | ✅ 1,036 live rows |
| Reddit (PRAW) | Optional | ⏭️ Skipped |
| GH Archive | Optional | ⏭️ Not required |
| 15 domains | Yes | ✅ |
| ≥10,000 records | Yes | ✅ |

## Personas (brief.pdf table)

| Persona | Overall | Notes |
|---------|---------|-------|
| Sofia (ML) | PASS | GFI + ML focus |
| David (DevOps) | PASS | Infra top-10 |
| Lina (Trend) | PASS | Visibility/recency mode |
| Raj (DevTools) | PASS | DevTools top-10 |

## Before Canvas upload

- [ ] Open live URL and verify ranked cards load
- [ ] Upload `Prasad_Shivneel_BAX423_Final.zip` from Desktop
- [ ] Attend Saturday 1:1 demo with DEMO.md
