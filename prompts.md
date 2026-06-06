# prompts.md — EngageIQ Final Project

**Student:** Shivneel Prasad · BAX-423 Big Data · Spring 2026  
**Project:** EngageIQ — Smart Engagement Opportunity Scorer

---

## Prompt 1: Project bootstrap & rubric mapping

```
Build EngageIQ - Smart Engagement Opportunity Scorer for BAX-423 Spring 2026 that:
- satisfies ALL 6 core capabilities (missing any caps score at 60/100)
- integrates exactly 2 BAX-423 techniques: recommendation system + reinforcement learning
- uses GitHub Search API + GitHub Archive as the two data sources (no Reddit/HN)
- covers all 15 domains in domains.py and 4 course personas (Sofia, David, Lina, Raj)
- ships folder layout: code/, data/, brief.docx, prompts.md
- includes persona_eval.py, run_benchmarks.py, user_test_loop.py
- runs locally with: cd code && py -m streamlit run app.py


```

**Purpose:** Scaffold the full submission against the official rubric so no required capability or deliverable is missed.  
**How I modified the output:** Split the monolithic prototype into `streaming.py`, `embedding.py`, `ranking.py`, `reinforcement_learning.py`, `analytics.py`, and `ui.py` instead of one script.

---

## Prompt 2 - Secure data ingestion (GitHub API + GitHub Archive)

```
Implement production-safe multi-source ingestion that:
- scrapes GitHub Search API repos + good-first-issue queries per domain via scrape_github.py
- parses GH Archive hourly JSON.gz for IssuesEvent, PullRequestEvent, and comment events via scrape_gharchive.py
- maps rows to schema: id, source, domain, title, text, url, community, created_at, upvotes, comments, author, lang, stars, forks, issues_open, good_first_issue
- requires GITHUB_TOKEN in .env for GitHub API; retries 403/429 via http_client.py
- deduplicates by URL and writes ≥10,000 live API rows to data/opportunities_snapshot.csv via build_snapshot.py
- escalates GH Archive hour-window until the 10k minimum is met with real URLs only
- never commits secrets; uses python-dotenv + truststore for Windows SSL


```

**Purpose:** Build Capability 1 with two real API sources and a grader-ready offline CSV that runs without API keys.  
**How I modified the output:** Removed synthetic padding entirely and added GH Archive hour-window escalation plus SSL/retry handling for Windows grading environments.

---

## Prompt 3 - Embeddings & similarity retrieval (Capability 2)

```
Implement embedding.py for approximate nearest-neighbor retrieval that:
- builds TF-IDF vectors on title + text + domain, reduced to 128 dimensions with truncated SVD
- indexes corpus with sklearn NearestNeighbors (cosine metric, brute algorithm)
- queries using persona interest text plus last 5 liked items from the RL session
- returns top-200 candidate indices and cosine distances
- caches the index with @st.cache_resource keyed on corpus size/hash
- supports live_only and english_only filters via ranking_corpus() in data_utils.py
- reports retrieval quality with NDCG@10 derived from interest keyword overlap


```

**Purpose:** Satisfy the embedding + ANN retrieval requirement without heavy deep-learning dependencies.  
**How I modified the output:** Chose TF-IDF/SVD over Sentence-BERT for cold-start speed and surfaced the metric as **Interest match %** in the UI for non-technical users.

---

## Prompt 4 - Multi-stage ranking & scoring (Capability 3)

```
Implement ranking.py multi-stage pipeline that:
- retrieves top-200 candidates from embedding.py (Stage 1)
- augments with persona-relevant GFIs, DevOps repos, and GH Archive events via augment_candidates() (Stage 2)
- reranks on score_relevance, score_health, score_visibility, score_effort, score_recency (Stage 3)
- applies persona keyword boosts, GFI boost for portfolio personas, and GH Archive boost for archive interests
- splits health/visibility formulas by source (GitHub vs GH Archive)
- exposes component scores on each card as "Why ranked here" chips
- measures NDCG@10 per persona in persona_eval.py
- returns top-100 results for the Discover tab (result_limit=100)


```

**Purpose:** Score engagement value with a multi-signal pipeline rather than a single sort key.  
**How I modified the output:** Added persona-specific boosts (GFI for Sofia, niche DevOps for David, recency for Lina) and a portfolio-mode penalty for slow high-effort items.

---

## Prompt 5 - Reinforcement learning bandit (Capability 4)

```
Implement reinforcement_learning.py + bandit.py contextual bandit that:
- treats 15 technical domains as arms with Thompson-sampling Beta posteriors
- accepts feedback rewards: engage +1.0, bookmark +0.85, skip 0.0
- shifts domain weights in rerank() via dom_boost vector after each action
- persists agent state in st.session_state across Streamlit reruns
- renders learned domain weights in sidebar via render_rl_policy()
- runs 60-round RL vs no-RL simulation in run_benchmarks.py
- reports avg_reward_last10, cumulative_reward, and reward_improvement_last10 in benchmark_results.json
- targets measurable improvement over the no-RL baseline (achieved +0.93 last-10 delta)


```

**Purpose:** Implement BAX-423 technique 2 with measurable learning over 50+ simulated feedback rounds.  
**How I modified the output:** Tuned rewards to engage/bookmark/skip and connected bandit weights directly into the reranker instead of a standalone RL service.

---

## Prompt 6 - Streaming pipeline & DuckDB storage (Capability 1)

```
Implement Capability 1 streaming + structured storage that:
- loads opportunities_snapshot.csv into DuckDB via OpportunityStore in data.py
- deduplicates on URL during ingest and tracks max_id()
- simulates streaming with OpportunityStream queue in streaming.py (produce_rows / consume)
- exposes sidebar controls: batch size slider, Produce batch, Consume → store buttons
- displays produced / ingested / deduped / queue counts after each operation
- demonstrates dedup by producing overlapping batches without duplicate URLs in the store

```

**Purpose:** Demonstrate streaming ingest and deduplication beyond a one-time static CSV load.  
**How I modified the output:** Kept an in-process queue instead of Kafka and surfaced dedup metrics in the sidebar for the live demo walkthrough.

---

## Prompt 7 - Batch analytics & trends (Capability 5)

```
Implement analytics.py batch processing that:
- runs DuckDB SQL for domain volume, daily trends, and week-over-week rising domains
- falls back to pandas aggregates when DuckDB is unavailable
- renders Altair charts in Analytics tab: volume by domain, daily volume, top-10 rising domains (WoW)
- adds RL benchmark button that runs learning_benchmark inline
- wires PDF/CSV engagement brief export via brief_export.py
- satisfies trending topics, active communities, volume over time, and domain distribution requirements

```

**Purpose:** Provide aggregate trend views required for the dashboard and Lina persona validation.  
**How I modified the output:** Added a dedicated WoW rising-domains chart for Lina's "rising opportunities" pass criteria and pandas fallback for environments without DuckDB write access.

---

## Prompt 8 - Dashboard UI, structured personas & cards (Capability 6)

```
Build Streamlit dashboard in app.py + ui.py that:
- provides Discover, Bookmarks, Activity, and Analytics tabs
- loads four course personas with five structured fields (background, interests, goal, platforms, time budget)
- shows ranked cards with title, source badge, domain, summary, fact chips, effort estimate, and component scores
- includes Engage / Bookmark / Skip action buttons on every card
- supports custom profile save/load via profile_store.py
- displays plain-language header metrics: Interest match %, GitHub API count, GH Archive count
- uses inject_theme() for compact Inter-font card layout readable by non-data-scientists


```

**Purpose:** Make the app usable by non-data-scientists with the exact persona profiles from the brief.  
**How I modified the output:** Replaced a single free-text interest box with five structured profile fields and added custom profile persistence with benchmark re-run on save.

---

## Prompt 9 - Suggested actions & explainability

```
Implement suggestions.py and card explainability that:
- shows decision_facts chips and component score breakdown on each opportunity card
- generates suggested actions via source-aware templates (GFI issue, repo contribution, archive thread comment)
- optionally calls Gemini or Groq when API keys are present in .env / Streamlit secrets
- falls back to offline templates when no LLM key is configured (default for graders)
- caches suggestions in session_state to avoid repeated API calls
- implements brief_export.py for PDF + CSV export of top-20 ranked opportunities and trend summary


```

**Purpose:** Meet the "Why this?" and suggested-action requirements without requiring API keys for offline grading.  
**How I modified the output:** Defaulted to interest-aware offline templates and only invoked LLM providers when keys were explicitly configured.

---

## Prompt 10 - Sort, filter & UX polish

```
Add client-side sort and filter controls that:
- sort by: best match, quickest to contribute, most visible, most active community, most recent
- filter by source: All | GitHub API only | GitHub Archive only
- filter by domain multiselect, GFI-only checkbox, and max effort (Any / Under 1 hr / Under 2 hr)
- show up to 100 ranked results with acted-on session tracking
- rename primary metric from NDCG@10 to "Interest match %" for end users
- keep NDCG@10 visible in grader expander only
- render filter panel always visible (not hidden in expander)

```

**Purpose:** Give graders and users control over ranked results without triggering a new API scrape.  
**How I modified the output:** Split GitHub vs GH Archive health/visibility scoring and moved the filter panel out of a collapsed expander into the main Discover layout.

---

## Prompt 11 - Persona benchmarks & automated testing

```
Implement persona_eval.py and scripts/user_test_loop.py that:
- validates Sofia: ≥3 GFI in top-10, no C++/Rust, ML-focused discussions, brief <1 hr per opportunity
- validates David: ≥5 DevOps/K8s in top-10, niche high-activity repos, discussion-oriented items
- validates Lina: recency/velocity over skill match, WoW analytics present, rising domains in brief
- validates Raj: ≥4 devtools hits in top-10, discussion threads, RL deprioritizes after simulated skips
- records capability_pass matrix (6 capabilities × 4 personas = 24 cells)
- runs learning_benchmark(): 60 rounds RL vs no-RL with reward improvement report
- executes user_test_loop.py with 15 checks: submission files, dataset ≥10k, 15 domains, streaming, RL, card quality
- writes results to data/benchmark_results.json via run_benchmarks.py
- targets 15/15 PASS and 4/4 persona PASS before Canvas submission

```

**Purpose:** Validate against official persona rules with automated checks graders can reproduce locally.  
**How I modified the output:** Added sub-criteria checks (niche repos for David, RL skip test for Raj, brief <1 hr for Sofia) and GH Archive threads as Reddit/blog proxies.

---

## Prompt 12 - GitHub Archive migration (replace Hacker News)

```
Migrate second data source from Hacker News to GitHub Archive that:
- creates scrape_gharchive.py to parse hourly JSON.gz engagement events with real html_url values
- updates domains.py keyword maps from "hn" to "gharchive" for all 15 domains
- replaces source=hackernews with source=gharchive across ranking, UI, data_utils, and tests
- runs migrate_to_gharchive.py to transform legacy rows and regenerate snapshot CSVs
- updates persona interest text to reference GitHub Archive events where applicable
- re-runs benchmarks confirming ≥10,000 rows, 15/15 domains, and 2 live API sources
- deletes scrape_hn.py and removes all hackernews filter labels from the UI

```

**Purpose:** Align the second data source with GitHub-native trend and discussion events per the project brief.  
**How I modified the output:** Updated all source filters, scrapers, and tests from `hackernews` to `gharchive` and re-ran persona benchmarks on the new snapshot.

---

## Prompt 13 - Submission packaging & Canvas ZIP

```
Implement scripts/package_submission.py that:
- builds Prasad_Shivneel_BAX423_Final.zip with exact layout: code/, data/, brief.docx, prompts.md
- excludes .env, __pycache__, .duckdb, and other secrets or runtime artifacts
- bundles data/opportunities_snapshot.csv so graders run offline without API tokens
- includes code/requirements.txt and code/README.md with single-command run instructions
- verifies brief.docx and prompts.md exist at project root before zipping
- prints file count and ZIP size on completion

```

**Purpose:** Produce the exact Canvas folder layout with all required artifacts in one command.  
**How I modified the output:** Bundled the full offline snapshot inside the ZIP and excluded all secret/runtime files via an explicit skip list.

---

## Prompt 14 - Deploy, validate & demo prep

```
Complete deployment and validation checklist that:
- pushes code to github.com/shprasa/engageiq-bax423-final via push_to_github.py (GitHub REST API)
- deploys Streamlit Cloud with main module code/app.py and bundled code/data/ snapshot paths
- verifies live app shows: 4 personas, Sort & filter, Interest match %, GH Archive labels, Engage/Skip RL
- writes DEMO.md with 10-minute walkthrough covering all 6 capabilities and both BAX-423 techniques
- re-runs py scripts/user_test_loop.py and confirms 15/15 PASS
- confirms live URL https://engageiq-bax423-final.streamlit.app/ responds at grading time

```

**Purpose:** Satisfy hosted-deployment and Saturday demo deliverables with a reproducible validation gate.  
**How I modified the output:** Added mirrored `code/data/` snapshot copies so the app resolves data paths on both local unzip and Streamlit Cloud deployments.
