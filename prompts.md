# prompts.md — EngageIQ Final Project

**Student:** Shivneel Prasad · BAX-423 Big Data · Spring 2026  
**Project:** EngageIQ — Smart Engagement Opportunity Scorer

Key prompts used during development. Each entry includes one sentence on purpose and one on how the output was modified.

---

## Prompt 1 — Project bootstrap & rubric mapping

```
Build EngageIQ for BAX-423: all 6 core capabilities, 2 techniques (recommendation + RL),
GitHub API + GH Archive sources, 15 domains, 4 personas, ≥10k offline snapshot,
Streamlit dashboard, persona_eval.py, user_test_loop.py.
```

**Purpose:** Scaffold the full submission layout against the official rubric so nothing required was missed.  
**How I modified the output:** Mapped each capability to a dedicated module (`streaming.py`, `embedding.py`, `ranking.py`, etc.) instead of keeping everything in one script.

---

## Prompt 2 — Secure data ingestion (GitHub API + GitHub Archive)

```
Implement scrape_github.py, scrape_gharchive.py, and build_snapshot.py with URL dedup,
DOMAIN_QUERIES per domain, GITHUB_TOKEN in .env, and synthetic padding to ≥10,000 rows.
```

**Purpose:** Build Capability 1 with two real API sources and a grader-ready offline CSV.  
**How I modified the output:** Added Windows SSL via truststore, retry logic on 403/429, and merge order that prefers live URLs over `example.local` synthetic rows.

---

## Prompt 3 — Embeddings & similarity retrieval (Capability 2)

```
Implement TF-IDF/SVD embeddings and cosine NearestNeighbors retrieval returning top-200
candidates; cache the index; report NDCG@10.
```

**Purpose:** Satisfy the embedding + ANN retrieval requirement without heavy GPU models.  
**How I modified the output:** Used 128-dim SVD instead of Sentence-BERT for faster cold-start and bundled the metric as **Interest match %** in the UI.

---

## Prompt 4 — Multi-stage ranking & scoring (Capability 3)

```
Implement candidate generation → augment_candidates → rerank with relevance, health,
visibility, effort, recency; expose component scores on cards.
```

**Purpose:** Score engagement value with a multi-signal pipeline, not a single sort key.  
**How I modified the output:** Added persona-specific boosts (GFI for Sofia, niche DevOps repos for David, recency for Lina) and a +0.18 live-URL boost over synthetic rows.

---

## Prompt 5 — Reinforcement learning bandit (Capability 4)

```
Implement Thompson-sampling bandit over 15 domains with engage/bookmark/skip rewards;
60-round RL vs no-RL benchmark.
```

**Purpose:** Implement BAX-423 technique 2 with measurable improvement after feedback.  
**How I modified the output:** Set rewards to engage +1.0, bookmark +0.85, skip 0.0 and wired bandit weights directly into the rerank `dom_boost` vector.

---

## Prompt 6 — Streaming pipeline & DuckDB storage (Capability 1)

```
Implement OpportunityStream queue with URL dedup and OpportunityStore DuckDB ingest;
sidebar Produce batch / Consume → store controls.
```

**Purpose:** Demonstrate streaming ingest and deduplication, not just static CSV load.  
**How I modified the output:** Kept an in-process queue (not Kafka) and surfaced produced/ingested/deduped counts in the sidebar for demo visibility.

---

## Prompt 7 — Batch analytics & trends (Capability 5)

```
Implement DuckDB batch queries for domain volume, daily trends, and week-over-week
rising domains; display in Analytics tab with Altair charts.
```

**Purpose:** Provide aggregate trend views required for the dashboard and Lina persona.  
**How I modified the output:** Added pandas fallback when DuckDB is unavailable and a WoW chart specifically for “rising” opportunity detection.

---

## Prompt 8 — Dashboard UI, structured personas & cards (Capability 6)

```
Build Discover/Bookmarks/Activity/Analytics tabs with five-field persona profiles from
the course PDF, ranked cards, Engage/Bookmark/Skip, and plain-language labels.
```

**Purpose:** Make the app usable by non-data-scientists with the exact persona wording from the brief.  
**How I modified the output:** Replaced a single interest text box with five structured fields and added custom profile save/load via `profile_store.py`.

---

## Prompt 9 — Suggested actions & explainability

```
Add template suggested actions per source type and decision_facts chips on each card;
optional LLM APIs with offline fallback; PDF/CSV brief export.
```

**Purpose:** Meet the “Why this?” and suggested-action requirements without requiring API keys.  
**How I modified the output:** Defaulted to source-aware templates (GFI vs repo vs archive thread) and only call Gemini/Groq when keys are present.

---

## Prompt 10 — Sort, filter & UX polish

```
Add sort/filter panel (source, live/offline, domain, effort, GFI); fix GH Archive
ranking scores; rename NDCG@10 to Interest match % in the UI.
```

**Purpose:** Give graders and users control over ranked results without re-scraping.  
**How I modified the output:** Split health/visibility formulas by source (GitHub vs GH Archive) and made the filter panel always visible instead of hidden in an expander.

---

## Prompt 11 — Exact PDF persona pass criteria & automated testing

```
Implement persona_eval.py with exact PDF pass criteria, capability_pass matrix,
user_test_loop.py (15 checks), and run_benchmarks.py.
```

**Purpose:** Validate against the official persona rules, not simplified proxies.  
**How I modified the output:** Added sub-criteria checks (niche repos for David, RL skip test for Raj, brief <1 hr for Sofia) and GH Archive threads as Reddit/blog proxies.

---

## Prompt 12 — GitHub Archive migration (replace Hacker News)

```
Replace Hacker News with GitHub Archive across scrapers, ranking, UI, tests, and
offline snapshot; regenerate CSVs and re-run benchmarks.
```

**Purpose:** Align the second data source with trend/discussion events on GitHub.  
**How I modified the output:** Ran `migrate_to_gharchive.py` on synthetic rows and updated all source filters from `hackernews` to `gharchive`.

---

## Prompt 13 — Live refresh, benchmark hooks & submission packaging

```
Add live API refresh in sidebar, re-run benchmarks on save/refresh, package_submission.py
for Canvas ZIP with code/, data/, brief.pdf, prompts.md.
```

**Purpose:** Keep benchmark artifacts current and produce the exact Canvas folder layout.  
**How I modified the output:** Bundled `data/opportunities_snapshot.csv` (≥10k rows) inside the ZIP so graders never need API tokens.

---

## Prompt 14 — Deploy, validate & demo prep

```
Deploy to Streamlit Cloud, verify live app, write DEMO.md, target user_test_loop 15/15 PASS.
```

**Purpose:** Satisfy the hosted-deployment and Saturday demo deliverables.  
**How I modified the output:** Added bundled `code/data/` snapshot copies so the app finds data on both local unzip and Streamlit Cloud paths.
