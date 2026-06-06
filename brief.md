# EngageIQ — Technical Brief

**Shivneel Prasad · BAX-423 Big Data · Spring 2026**

- **Live app:** https://engageiq-bax423-final.streamlit.app/
- **Repository:** https://github.com/shprasa/engageiq-bax423-final

---

## What EngageIQ is

EngageIQ is a **smart engagement opportunity scorer** — a prototype analytics platform that helps people decide *where to invest limited time* in open-source and developer communities. Instead of manually scanning GitHub repos, issues, pull requests, and public timeline events, a user describes their interests (or selects a persona), and EngageIQ retrieves, scores, and ranks the best opportunities to contribute, comment, or build visibility.

The system combines a **recommendation pipeline** (embedding-based retrieval + multi-signal reranking) with an **adaptive learning layer** (reinforcement-learning bandit that learns domain preferences from Engage / Bookmark / Skip feedback). Results appear in a Streamlit dashboard with plain-language explanations, suggested next actions, trend analytics, and exportable engagement briefs.

## Problem & motivation

Developer engagement is high-volume and fragmented. A student building an ML portfolio, a DevOps engineer seeking niche infra communities, a data journalist tracking emerging tools, and a startup founder scouting devtools conversations all face the same challenge: thousands of potential threads across GitHub, with no unified way to compare relevance, community health, visibility potential, and effort required.

EngageIQ treats each repo, issue, PR, or archive event as an **engagement opportunity** — a structured record with title, domain, activity signals, URL, and estimated time-to-contribute — then ranks them against a personal interest profile.

## Who it is for

Built-in personas use **five structured fields** from the course PDF (background, interests, goal, platforms, time budget):

| Persona | Goal | What EngageIQ surfaces |
|---------|------|------------------------|
| **Sofia** | Build ML portfolio via beginner-friendly OSS | Good-first issues, Python/ML repos, low-effort entry points |
| **David** | Find high-signal DevOps/K8s communities | Niche infra repos, CI/CD threads, stand-out contributor opportunities |
| **Lina** | Spot trends before they go mainstream | Recent, high-visibility archive events and fast-moving domains |
| **Raj** | Engage where devtools/B2B conversations happen | API, CLI, and developer-tools opportunities with business relevance |

Custom profiles can be saved to `data/custom_personas.json`.

## How users interact with the app

- **Discover** — load a persona or custom profile; view ranked cards (up to 100); sort and filter by source, effort, domain, live/offline; Engage / Bookmark / Skip to train the system
- **Bookmarks** — saved opportunities for later
- **Activity** — feedback log with CSV export
- **Analytics** — domain volume charts, daily trends, week-over-week rising domains, RL benchmark, PDF/CSV brief export
- **Sidebar** — streaming pipeline controls; optional live API refresh (re-runs persona benchmarks)

## Data sources

1. **GitHub Search API** (`scrape_github.py`) — repos, good-first issues, stars/forks/language metadata
2. **GitHub Archive** (`scrape_gharchive.py`) — hourly public timeline events (issues, PRs, comments)

Combined offline snapshot: **10,944 rows · 15 domains · 2,908 live · 8,036 synthetic backup** (example.local URLs for offline grading). GH Archive issue/PR threads proxy Reddit/blog discussion items in persona validation.

## System architecture

1. **Ingest & store** — scrapers → CSV snapshot → streaming queue → URL dedup → DuckDB
2. **Embed & retrieve** — TF-IDF/SVD → cosine NearestNeighbors → top-200 candidates
3. **Augment & rerank** — persona-aware injection → composite scoring (relevance, health, visibility, effort, recency) → top-100; DevOps niche-repo boost for David
4. **Learn** — Thompson-sampling bandit over 15 domains from user feedback
5. **Analyze** — DuckDB batch SQL for domain distributions and WoW trends
6. **Serve** — Streamlit dashboard with cards, charts, and export

## BAX-423 techniques

1. **Recommendation system** — TF-IDF/SVD embeddings, ANN retrieval, multi-stage reranking, NDCG@10 (shown as Interest match %)
2. **Reinforcement learning** — contextual bandit; +0.925 reward improvement over 60-round baseline (engage +1.0, bookmark +0.85, skip 0.0)

## Persona validation

Exact PDF pass criteria are automated in `persona_eval.py` (not simplified proxies). All four personas PASS; all 24 persona×capability cells PASS. See `brief.pdf` sections 13–14 and `data/benchmark_results.json`.

## Run locally

Unzip `Prasad_Shivneel_BAX423_Final.zip`, open a terminal in the extracted folder, then copy and paste:

```bat
cd code
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app opens at **http://localhost:8501**. Offline data loads from `data/opportunities_snapshot.csv` — no API keys required. On macOS/Linux, use `python3` instead of `py` if needed.

PDF version: `brief.pdf` · Editable version: `brief.docx` (generated by `code/scripts/generate_brief.py`)
