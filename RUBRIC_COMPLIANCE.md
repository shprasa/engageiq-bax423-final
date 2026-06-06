# EngageIQ — Official Brief Compliance Audit

**Sources:** BAX423 Final One-Pager + EngageIQ Spring 2026 brief  
**Student:** Shivneel Prasad · **Updated:** June 6, 2026

> Missing **any one** of the 6 core capabilities caps score at **60/100**.

---

## Global submission requirements

| Required item | Brief rule | Status | Evidence |
|---------------|------------|--------|----------|
| Live public URL | Hosted deployment | ✅ | https://engageiq-bax423-final.streamlit.app/ |
| Source repo | Public code | ✅ | github.com/shprasa/engageiq-bax423-final |
| ZIP layout | `code/`, `data/`, `brief.docx`, `prompts.md` | ✅ | `Prasad_Shivneel_BAX423_Final.zip` |
| Single-command run | README + requirements | ✅ | `cd code && py -m streamlit run app.py` |
| ≥10,000 offline records | All 15 domains | ✅ | 12,750 rows, 15/15 domains |
| ≥2 BAX-423 techniques | Different lectures, benchmarked | ✅ | Recommendation + RL |
| 4 personas + pass/fail table | In technical brief | ✅ | Sections 7–8 in `brief.docx` |
| prompts.md | Key development prompts | ✅ | 14 v3 prompts with purpose + modification |
| Saturday 10-min demo | Live walkthrough | ⏳ | User action |

---

## BAX-423 techniques (2 required — student scope)

| Technique | Implementation | Benchmark |
|-----------|----------------|-----------|
| **Recommendation** | TF-IDF/SVD → cosine NearestNeighbors → multi-stage rerank | NDCG@10 per persona; Interest match % in UI |
| **Reinforcement learning** | Thompson-sampling contextual bandit over domains | 60-round sim: +0.93 reward vs no-RL baseline |

---

## Capability 1 — Multi-Source Data Ingestion & Streaming

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| ≥2 data sources | ✅ | GitHub API + GitHub Archive |
| Real-time / streaming pipeline | ✅ | Sidebar: Produce batch → Consume → DuckDB |
| Deduplication | ✅ | URL dedup in stream queue + DuckDB ingest |
| Structured storage | ✅ | DuckDB `opportunities` table |
| ≥10,000 records | ✅ | 12,750 rows |
| All 15 domains | ✅ | `domains.py` + ingest benchmark |

---

## Capability 2 — Content Embedding & Similarity Retrieval

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Text + profile embeddings | ✅ | TF-IDF/SVD (`embedding.py`) |
| Approximate nearest-neighbor search | ✅ | sklearn `NearestNeighbors` cosine |
| Retrieves relevant candidates | ✅ | Top-200 candidates → rerank |

---

## Capability 3 — Engagement Scoring & Multi-Stage Ranking

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Relevance component | ✅ | Embedding similarity |
| Community health signals | ✅ | `score_health` |
| Visibility potential | ✅ | `score_visibility` |
| Estimated effort | ✅ | `score_effort` + engagement time labels |
| Multi-stage pipeline | ✅ | ANN → augment → rerank |
| Ranking metric reported | ✅ | NDCG@10 / Interest match % |
| “Why ranked here?” | ✅ | Component scores on each card |

---

## Capability 4 — Adaptive Learning from Feedback

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Feedback loop | ✅ | Engage / Bookmark / Skip → bandit update |
| Learning mechanism | ✅ | Thompson sampling |
| 50+ simulated rounds | ✅ | 60-round benchmark |
| Measurable improvement | ✅ | Reward 1.0 vs 0.075 (last-10 avg) |

---

## Capability 5 — Batch Analytics & Trend Detection

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Batch processing | ✅ | DuckDB SQL (`analytics.py`) |
| Trending topics / domains | ✅ | WoW rising domains chart |
| Volume over time | ✅ | Daily volume line chart |
| Category / domain distribution | ✅ | Volume by domain bar chart |
| Dashboard display | ✅ | Analytics tab |

---

## Capability 6 — Dashboard & Engagement Brief

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Ranked opportunities + scores | ✅ | Discover tab (up to 100 results) |
| “Why this?” explanations | ✅ | Match/activity/visibility/effort chips |
| Suggested actions | ✅ | Offline templates (+ optional API keys) |
| Engage / Skip / Bookmark | ✅ | Card action buttons |
| Trend visualisations | ✅ | Analytics tab |
| Downloadable brief (PDF/CSV) | ✅ | Export buttons |
| Usable UI | ✅ | Persona presets, sort/filter, plain language |

---

## Persona pass criteria

| Persona | Criteria | Status |
|---------|----------|--------|
| **Sofia** | ≥3 GFI; no C++/Rust; ML focus | ✅ PASS |
| **David** | ≥5 DevOps/K8s in top-10 | ✅ PASS |
| **Lina** | Visibility ≥ relevance | ✅ PASS |
| **Raj** | ≥4 devtools/B2B in top-10 | ✅ PASS |

All 6 capabilities: **PASS** for each persona in `benchmark_results.json`.

---

## Pre-submission checklist

- [ ] Redeploy Streamlit Cloud after latest push
- [ ] Upload `Prasad_Shivneel_BAX423_Final.zip` to Canvas by Fri Jun 5 11:59 PM PT
- [ ] Saturday 10-min demo using `DEMO.md`
