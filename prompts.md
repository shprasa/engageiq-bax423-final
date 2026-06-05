# prompts.md — EngageIQ Final Project

**Student:** Shivneel Prasad · BAX-423 Big Data · Spring 2026  
**Project:** EngageIQ — Smart Engagement Opportunity Scorer

Production-aware AI prompts used to build this project start-to-finish. Each prompt maps to a major build phase and directly references the official brief (6 core capabilities, 2 BAX-423 techniques, 4 personas, ≥10k offline records, live deployment).

---

## Prompt 1 — Project bootstrap & rubric mapping

```
Build EngageIQ — Smart Engagement Opportunity Scorer for BAX-423 Spring 2026.

MUST satisfy ALL 6 core capabilities (missing any caps score at 60/100):
1. Multi-source ingest + streaming pipeline + URL dedup + structured store (DuckDB)
2. TF-IDF/SVD embeddings + approximate nearest-neighbor retrieval
3. Multi-stage ranking (relevance, health, visibility, effort) + NDCG@10 metric
4. Adaptive learning from engage/skip/bookmark feedback — RL bandit, 50+ round benchmark
5. Batch analytics: trending domains, volume over time, category breakdown
6. Streamlit dashboard: ranked list, Why-this scores, suggested actions, PDF/CSV brief export

BAX-423 techniques (exactly 2): Recommendation system + Reinforcement learning.

Data sources (exactly 2): GitHub Search API + GitHub Archive (gharchive.org hourly JSON.gz).
NOT using Reddit or Hacker News.

Deliverables folder layout:
  EngageIQ_Final/code/          (app.py, engageiq/ package, scripts/)
  EngageIQ_Final/data/          (opportunities_snapshot.csv, benchmark_results.json)
  EngageIQ_Final/brief.pdf
  EngageIQ_Final/prompts.md

Include 15 domains from domains.py, 4 persona presets (Sofia, David, Lina, Raj),
and automated persona_eval.py + run_benchmarks.py.
Single command: cd code && py -m streamlit run app.py
```

**Output:** Modular `engageiq/` package, `app.py` with 4 tabs, rubric-aligned scaffold.

---

## Prompt 2 — Secure data ingestion (GitHub API + GitHub Archive)

```
Implement production-safe multi-source ingestion:

scrape_github.py:
- GitHub Search API for repos + good-first-issue queries per domain (DOMAIN_QUERIES)
- Map to schema: id, source=github, domain, title, text, url, community, created_at,
  upvotes, comments, author, lang, stars, forks, issues_open, good_first_issue
- Require GITHUB_TOKEN in .env; retry on 403/429 via http_client.py

scrape_gharchive.py:
- Download hourly files from https://data.gharchive.org/YYYY-MM-DD-H.json.gz
- Parse IssuesEvent, PullRequestEvent, IssueCommentEvent, PullRequestReviewCommentEvent
- source=gharchive, real GitHub html_url, domain keyword matching
- No auth required; limit hours_back and max_events for prototype scale

build_snapshot.py:
- Merge LIVE rows first, dedupe by URL, pad synthetic backup to ≥10,000 rows
- Prefer live URLs over example.local on conflict
- Print source mix and domain count

Security: python-dotenv, truststore for Windows SSL, .gitignore for .env
Never commit secrets. Document grader setup in README.
```

**Output:** `scrape_github.py`, `scrape_gharchive.py`, `build_snapshot.py`, `http_client.py`.

---

## Prompt 3 — Embeddings & similarity retrieval (Capability 2)

```
Implement embedding.py for Capability 2:
- Build TF-IDF vectorizer on corpus (title + text + domain), truncate with SVD (100 dims)
- sklearn NearestNeighbors with cosine metric, algorithm=brute (ANN at prototype scale)
- Query = persona interest text + last 5 liked items from RL session
- Return top-200 candidate indices + cosine distances
- Cache index with @st.cache_resource keyed on corpus size/hash
- Support live_only and english_only filters via ranking_corpus() in data_utils.py
Report retrieval quality via NDCG@10 labels derived from interest keyword overlap.
```

**Output:** `embedding.py`, cached index in `app.py`, NDCG@10 in Discover tab.

---

## Prompt 4 — Multi-stage ranking & scoring (Capability 3)

```
Implement ranking.py multi-stage pipeline for Capability 3:

Stage 1 — ANN retrieval: top-200 candidates from embedding.py
Stage 2 — augment_candidates(): inject domain-relevant GitHub GFI + GH Archive events
         when persona interest mentions portfolio, DevOps, ML, or archive keywords
Stage 3 — rerank() composite score:
  - score_relevance (embedding similarity)
  - score_health (GitHub: stars/forks/activity; GH Archive: comment volume)
  - score_visibility (upvotes/comments/stars or archive activity)
  - score_effort (penalize high effort; cap GFI at low effort)
  - score_recency (created_at decay)
  - persona keyword boosts, GFI boost for portfolio personas, GH Archive boost
  - live URL boost (+0.18) over example.local synthetic rows
  - RL domain boost from Thompson bandit weights

Expose component scores on each card ("Why ranked here").
Measure NDCG@10 per persona in persona_eval.py.
Default result_limit=100 for Discover tab.
```

**Output:** `ranking.py`, score chips on cards, persona-specific augment logic.

---

## Prompt 5 — Reinforcement learning bandit (Capability 4 + BAX-423 technique 2)

```
Implement reinforcement_learning.py + bandit.py:

Formulation: contextual multi-armed bandit
- Arms: 15 technical domains (DOMAINS list)
- State: user interest text + liked_texts history + session feedback
- Actions: domain-weight sampling during rerank (dom_boost vector)
- Rewards: engage +1.0, bookmark +0.5, skip -0.3, unbookmark -0.2
- Policy: Thompson sampling with Beta posteriors per domain

UI: Engage/Bookmark/Skip on each card → observe_feedback() → toast with reward
Sidebar: render_rl_policy() showing learned domain weights

Benchmark (run_benchmarks.py / Analytics tab):
- 60 rounds with RL vs 60 rounds without RL on live corpus
- Report: avg_reward_last10, cumulative_reward, reward_improvement_last10
- Target: measurable improvement (achieved +0.7 reward delta)
Persist agent in st.session_state across reruns.
```

**Output:** `reinforcement_learning.py`, `bandit.py`, RL sidebar, 60-round benchmark JSON.

---

## Prompt 6 — Streaming pipeline & DuckDB storage (Capability 1)

```
Implement Capability 1 streaming + storage:

data.py — OpportunityStore:
- DuckDB table `opportunities` matching CSV schema
- load_df(), ingest_batch(), max_id(), ensure_loaded_from_snapshot()
- URL dedup on ingest; prefer live rows; purge stale DB if CSV has live but DB empty

streaming.py — OpportunityStream:
- produce_rows(batch from snapshot CSV)
- consume(max_items) → ingest_fn with dedup count metrics
- Sidebar: batch size slider, Produce batch / Consume → store buttons
- Display produced/ingested/deduped/queue counts

Demonstrate in demo: Produce 300 → Consume → row count increases, dedup skips duplicates.
```

**Output:** `data.py`, `streaming.py`, sidebar streaming controls.

---

## Prompt 7 — Batch analytics & trends (Capability 5)

```
Implement analytics.py for Capability 5 batch processing:

DuckDB SQL queries (fallback to pandas if DB unavailable):
- by_domain: count, avg_upvotes, avg_comments grouped by domain (30-day window)
- volume_over_time: daily opportunity count
- compute_wow_domain_growth(): week-over-week delta per domain

Analytics tab in app.py:
- Altair bar chart: volume by domain
- Altair line chart: daily volume
- Altair bar chart: top 10 rising domains (WoW)
- RL benchmark button (runs learning_benchmark inline)
- Export CSV/PDF brief buttons (brief_export.py)

Satisfies brief requirement for trending topics, active communities, volume over time,
and category/domain distributions on dashboard.
```

**Output:** `analytics.py`, Analytics tab with 3 charts + export.

---

## Prompt 8 — Dashboard UI, personas & card design (Capability 6)

```
Build Capability 6 dashboard in app.py + ui.py:

Tabs: Discover | Bookmarks | Activity | Analytics

Discover:
- 4 persona presets with interest text (Sofia, David, Lina, Raj)
- Plain-language header: Interest match %, GitHub API count, GH Archive count, Live count
- Sort & filter panel: platform, live/offline, effort, domain, show up to 100
- Opportunity cards: title, source badge, domain, summary, fact chips (10-11px),
  estimated engagement time, component scores, suggested action, Engage/Bookmark/Skip

Bookmarks + Activity log tabs with filter and CSV download.

Hero banner with dataset stats. inject_theme() for Inter font, compact card layout.
Cards must work for non-data-scientists — no raw jargon in primary labels.
Use render_opportunity_card() with live vs offline badge.
```

**Output:** `app.py`, `ui.py`, persona-driven Discover experience.

---

## Prompt 9 — LLM suggested actions & explainability

```
Implement suggestions.py with free-tier fallback chain:

1. Try Gemini API (GEMINI_API_KEY, gemini-2.5-flash-lite) — free from Google AI Studio
2. Fallback Groq API (GROQ_API_KEY)
3. Fallback OpenAI if configured
4. Default: interest-aware template suggestions (no key required)

Templates by source:
- github + GFI: "open issue, ask clarifying question, small PR within an hour"
- github repo: "read README, pick docs/test issue, open PR"
- gharchive: "read issue/PR thread, leave helpful comment or small follow-up PR"

Use truststore + certifi for Windows SSL. Cache suggestions in session_state.
Show provider label on card. llm_configured() check for sidebar caption.

Also implement brief_export.py: ReportLab PDF + CSV export of top-20 ranked + trends.
```

**Output:** `suggestions.py`, `brief_export.py`, template + optional LLM path.

---

## Prompt 10 — Sort, filter & UX polish

```
Add client-side sort/filter in Discover (data_utils.py):

SORT_OPTIONS: best match, quickest to contribute, most visible, most active community,
most recent
SOURCE_FILTER: All | GitHub API only | GitHub Archive only
ORIGIN_FILTER: All | Live from web | Offline practice data
MAX_EFFORT: Any | Under 1 hour | Under 2 hours
Domain multiselect, GFI checkbox, show up to 100

Fix GH Archive visibility in ranking (source-aware health/visibility scores,
gha_boost when interest mentions archive events).

Replace jargon: NDCG@10 → "Interest match %" for primary users; keep NDCG in
grader expander. Always-visible filter panel (not hidden expander).
```

**Output:** Sort/filter panel, GH Archive ranking fixes, plain-language metrics.

---

## Prompt 11 — Persona benchmarks & automated testing

```
Implement persona_eval.py + scripts/user_test_loop.py:

Personas (from brief):
- Sofia: ≥3 GFI in top-10, no C++/Rust, ML domain focus
- David: ≥5 DevOps/K8s in top-10
- Lina: visibility score ≥ relevance (trend/recency persona)
- Raj: ≥4 devtools/B2B/Cloud API in top-10

For each persona: capability_pass matrix (6 capabilities PASS/FAIL).
learning_benchmark(): 60 rounds RL vs no-RL, report reward improvement.

user_test_loop.py — 15 automated checks:
submission files, dataset ≥10k + 15 domains, multi_source ingest (github + gharchive),
streaming pipeline, persona benchmarks, RL improvement, card quality, live titles, etc.

run_benchmarks.py writes data/benchmark_results.json for brief.pdf generation.
Target: 15/15 PASS before Canvas submission.
```

**Output:** `persona_eval.py`, `user_test_loop.py`, `run_benchmarks.py`, all personas PASS.

---

## Prompt 12 — GitHub Archive migration (replace Hacker News)

```
Migrate second data source from hackernews to gharchive:

1. Create scrape_gharchive.py (hourly JSON.gz parser)
2. Update domains.py: "hn" keywords → "gharchive" keywords
3. Replace all source=hackernews with gharchive in ranking, UI, data_utils, tests
4. migrate_to_gharchive.py: transform offline HN rows, drop live HN, scrape live GHA,
   regenerate snapshot CSVs + live_opportunities.csv
5. Update personas interest text to mention GitHub Archive events
6. Re-run benchmarks; confirm ≥10,000 rows (currently 10,944), 15 domains, 2 sources

Delete scrape_hn.py. Update prompts.md and brief.pdf to document GitHub API + GH Archive only.
```

**Output:** Full GH Archive migration, 2,908 live rows, tests still 15/15 PASS.

---

## Prompt 13 — Technical brief, prompts doc & Canvas ZIP

```
Generate submission artifacts:

generate_brief.py → brief.pdf (max 4 pages):
- Explain what EngageIQ is, the problem it solves, and how the system works end-to-end
- Sections: project overview, personas, data sources, architecture, scoring, recommendation
  pipeline, RL learning, analytics, dashboard workflow, validation, limitations
- Pull live stats from data/benchmark_results.json

prompts.md: production-aware prompts used to build the project (this document)

package_submission.py → Prasad_Shivneel_BAX423_Final.zip:
  code/, data/, brief.pdf, prompts.md (exclude .env, __pycache__)

Push to github.com/shprasa/engageiq-bax423-final for Streamlit Cloud redeploy.
```

**Output:** `generate_brief.py`, `package_submission.py`, Canvas-ready ZIP.

---

## Prompt 14 — Deploy, validate & demo prep

```
Final deployment checklist:

1. push_to_github.py via GitHub REST API (no local git required)
2. Streamlit Cloud: main module code/app.py, secrets for GEMINI_API_KEY optional
3. Verify live app shows: GH Archive labels, Sort & filter, Interest match %,
   10,944 dataset / 2,908 live counts, 4 personas
4. DEMO.md: 10-minute walkthrough script covering all 6 capabilities
5. Re-run: py scripts/user_test_loop.py → 15/15 PASS
6. Upload Prasad_Shivneel_BAX423_Final.zip to Canvas

Live URL: https://engageiq-bax423-final.streamlit.app/
```

**Output:** Public repo, Streamlit deployment, demo script, validated submission.

---

## How I used AI output

| Phase | AI generated | I verified / changed |
|-------|----------------|----------------------|
| Scaffold | Package layout, app tabs | Mapped each module to rubric capability |
| Ingestion | Scrapers + merge logic | SSL fix, rate limits, GH Archive migration |
| Ranking | TF-IDF + rerank formula | Persona boosts, GFI/GH Archive augment, live boost |
| RL | Bandit + benchmark | Reward tuning, 60-round validation |
| UI | Streamlit cards | Compact fonts, plain language, sort/filter |
| Data | 10k snapshot | GH Archive live scrape, synthetic backup |
| Docs | Brief + prompts | Project-focused 4-page brief + production prompts |
| Tests | user_test_loop | Fixed source checks for gharchive, 15/15 pass |
