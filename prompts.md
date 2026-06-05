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
Build a prototype of EngageIQ: engagement opportunity scorer using GitHub, Reddit, HN.
Include ranking, dashboard, and offline dataset.
```
**Purpose:** Names the project and core outputs but misses rubric structure.  
**Modification:** Added explicit mapping to all 6 core capabilities and BAX-423 techniques.

### v3 · Production-aware (used)
```
Build EngageIQ prototype satisfying all 6 core capabilities from the Spring 2026 brief:
(1) multi-source ingest + streaming + Bloom dedup + DuckDB,
(2) TF-IDF/SVD embeddings + ANN retrieval,
(3) multi-stage ranking with relevance/health/visibility/effort + NDCG@10,
(4) Thompson-sampling bandit from engage/skip/bookmark + 60-round benchmark,
(5) DuckDB batch trend analytics,
(6) Streamlit dashboard with Why-this explanations, suggested actions, CSV/PDF brief.
Use folder layout: code/, data/, prompts.md, brief.pdf. Match course labs (sketching, Kafka optional).
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
Add GitHub and Reddit scrapers with .env for tokens and a script to build opportunities_snapshot.csv.
```

### v3 · Production-aware (used)
```
Implement secure API ingestion:
- .env + python-dotenv + truststore for Windows SSL
- scrape_github.py, scrape_hn.py, scrape_reddit.py
- build_snapshot.py merges LIVE rows first, then synthetic backup to guarantee ≥10,000 records across 15 domains
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
- Data: live + synthetic
- Write brief.pdf, prompts.md (iteration format), persona benchmarks, final ZIP Prasad_Shivneel_BAX423_Final.zip
```
**Purpose:** Single authorization block for end-to-end completion.  
**Modification:** Drove benchmark scripts, technical brief PDF, GitHub publish, and deployment docs.

---

## Prompt 4 — Ranking explainability (in-app)

### v3 · Production-aware (used)
```
For each ranked opportunity show component scores (relevance, health, visibility, effort)
and rule-based suggested actions by source (GitHub PR path, Reddit comment template, HN response).
Add PDF weekly brief export with top-20 and trending domains.
```
**Purpose:** Satisfies “Why this?” and downloadable brief requirements.  
**Modification:** Extended `brief_export.py` with ReportLab PDF writer.

---

## How I used AI output

| Step | AI output | My change |
|------|-----------|-----------|
| Architecture | Module layout | Verified against rubric checklist |
| Scrapers | GitHub/HN fetchers | Added SSL fix, rate-limit retries, synthetic merge |
| UI | Streamlit cards | Wired persona selector + PDF export |
| Brief | Markdown template | Converted to `generate_brief.py` → `brief.pdf` with benchmark JSON |

---

## Iteration lesson (Lecture 10)

**v3 prompts win** because they specify inputs, validation, fallbacks, logging/benchmarks, and deployment constraints — the same fundamentals as production ML systems.
