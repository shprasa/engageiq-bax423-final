# prompts.md — EngageIQ Final Project

**Student:** Shivneel Prasad · BAX-423 Spring 2026  
**Format:** Follows Lecture 10 *Prototype to Production* prompt iteration model (v1 weak → v2 specific → v3 production-aware).

---

## Prompt 1 — Project bootstrap (EngageIQ architecture)

### v1 · Weak
```
Build the EngageIQ project.
```
**Purpose:** Too vague — would produce unrelated code.  
**Modification:** Discarded; replaced with capability-mapped prompt.

### v2 · Specific
```
Build a prototype of EngageIQ: engagement opportunity scorer using GitHub API and GitHub Archive.
Include ranking, dashboard, and offline dataset.
```
**Purpose:** Names the project and core outputs but misses rubric structure.  
**Modification:** Added explicit mapping to all 6 core capabilities and 2 BAX-423 techniques.

### v3 · Production-aware (used)
```
Build EngageIQ prototype satisfying all 6 core capabilities from the Spring 2026 brief:
(1) multi-source ingest + streaming + URL dedup + DuckDB,
(2) TF-IDF/SVD embeddings + ANN retrieval,
(3) multi-stage ranking with relevance/health/visibility/effort + NDCG@10,
(4) Thompson-sampling bandit from engage/skip/bookmark + 60-round benchmark,
(5) DuckDB batch trend analytics,
(6) Streamlit dashboard with Why-this explanations, suggested actions, CSV/PDF brief.
Data sources: GitHub Search API + GitHub Archive (gharchive.org) only.
BAX-423 techniques: Recommendation + Reinforcement Learning (2 total).
Use folder layout: code/, data/, prompts.md, brief.pdf.
```
**Purpose:** Anchored deliverables to rubric dimensions so nothing caps score at 60/100.  
**Modification:** Output became `EngageIQ_Final/` scaffold with modular `engageiq/` package.

---

## Prompt 2 — API ingestion & secrets

### v1 · Weak
```
Add API support.
```

### v2 · Specific
```
Add GitHub API and GitHub Archive scrapers with .env for tokens and a script to build opportunities_snapshot.csv.
```

### v3 · Production-aware (used)
```
Implement secure API ingestion:
- .env + python-dotenv + truststore for Windows SSL
- scrape_github.py (Search API repos + good-first issues)
- scrape_gharchive.py (hourly JSON.gz from data.gharchive.org — IssuesEvent, PullRequestEvent, comments)
- build_snapshot.py merges LIVE rows first, then synthetic backup to guarantee ≥10,000 records across 15 domains
- migrate_to_gharchive.py for dataset migration
- Never commit secrets; document CMD setup for grader
```
**Purpose:** Production-safe ingestion path graders can run offline.  
**Modification:** Added `http_client.py` retries, `.gitignore`, and merge logic preferring live URLs on dedup.

---

## Prompt 3 — Submission completion (hands-off)

### v3 · Production-aware (used)
```
Finish everything without asking:
- Student: Shivneel Prasad, GitHub: shprasa, public repo, Streamlit Cloud deploy
- Data: GitHub API + GitHub Archive live + synthetic backup
- Write brief.pdf (3–4 pages), prompts.md (iteration format), persona benchmarks, final ZIP Prasad_Shivneel_BAX423_Final.zip
```
**Purpose:** Single authorization block for end-to-end completion.  
**Modification:** Drove benchmark scripts, technical brief PDF, GitHub publish, and deployment docs.

---

## Prompt 4 — Ranking explainability (in-app)

### v3 · Production-aware (used)
```
For each ranked opportunity show component scores (match, activity, visibility, effort)
and suggested actions by source (GitHub API issue/PR path, GitHub Archive event comment template).
Add sort/filter (platform, live/offline, effort, domain) and PDF/CSV brief export with top-20 and trending domains.
```
**Purpose:** Satisfies “Why this?” and downloadable brief requirements.  
**Modification:** Extended `brief_export.py` with ReportLab PDF writer; added Discover sort/filter panel.

---

## How I used AI output

| Step | AI output | My change |
|------|-----------|-----------|
| Architecture | Module layout | Verified against rubric checklist (6 capabilities) |
| Scrapers | GitHub + GH Archive fetchers | Added SSL fix, rate-limit retries, synthetic merge |
| UI | Streamlit cards | Persona selector, sort/filter, plain-language labels |
| Brief | Markdown template | Expanded `generate_brief.py` → 3-page `brief.pdf` with benchmark JSON |
| Migration | HN → GH Archive | Replaced second data source per assignment scope |

---

## Iteration lesson (Lecture 10)

**v3 prompts win** because they specify inputs, validation, fallbacks, logging/benchmarks, and deployment constraints — the same fundamentals as production ML systems.
