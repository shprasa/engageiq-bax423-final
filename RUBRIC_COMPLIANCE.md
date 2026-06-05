# EngageIQ — Official Brief Compliance Audit

**Sources:** `BAX423_Final_OnePager.pdf`, `BAX423_FinalProject_EngageIQ_Spring2026_v2.docx`  
**Student:** Shivneel Prasad · **Date:** June 3, 2026

> Missing **any one** of the 6 core capabilities caps score at **60/100**.

---

## Global submission requirements

| Required item | Brief rule | Status | Evidence |
|---------------|------------|--------|----------|
| Live public URL | Hosted deployment | ✅ | https://engageiq-bax423-final.streamlit.app/ |
| Source repo | Public code | ✅ | github.com/shprasa/engageiq-bax423-final |
| ZIP layout | `code/`, `data/`, `brief.pdf`, `prompts.md` | ✅ | `Prasad_Shivneel_BAX423_Final.zip` |
| Single-command run | README + requirements | ✅ | `cd code && py -m streamlit run app.py` |
| ≥10,000 offline records | All 15 domains | ✅ | 10,000 rows, 15/15 domains verified |
| ≥2 BAX-423 techniques | Different lectures, benchmarked | ✅ | Recommendation + RL (see below) |
| 4 personas + pass/fail table | In brief.pdf | ✅ | `benchmark_results.json` → brief.pdf table |
| prompts.md | Key AI prompts | ✅ | `prompts.md` |
| Saturday 10-min demo | Live walkthrough | ⏳ | User action |

---

## BAX-423 techniques (2 required)

| Technique | Implementation | Benchmark |
|-----------|----------------|-----------|
| **Recommendation** | TF-IDF/SVD → cosine NearestNeighbors → multi-stage rerank | NDCG@10 in Discover tab; persona eval |
| **Reinforcement learning** | Thompson-sampling contextual bandit over domains | 60-round sim: +0.7 reward vs no-RL baseline |

---

## Capability 1 — Multi-Source Data Ingestion & Streaming

**Brief requires:** ≥2 sources; stream items; deduplicate; store structured data.

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| ≥2 data sources | ✅ | GitHub (4,964) + Hacker News (5,036) in snapshot |
| Real-time / streaming pipeline | ✅ | Sidebar: Produce batch → Consume → DuckDB (`streaming.py`) |
| Deduplication | ✅ | URL dedup in stream queue + DuckDB ingest |
| Structured storage | ✅ | DuckDB `opportunities` table (`data.py`) |
| ≥10,000 records | ✅ | 10,000 rows |
| All 15 domains | ✅ | `domains.py` + dataset check |

**Notes:** Reddit and GH Archive are optional among the four listed sources; GitHub + HN satisfies “≥2 of 4.” Reddit was removed per your request.

---

## Capability 2 — Content Embedding & Similarity Retrieval

**Brief requires:** Dense embeddings; ANN retrieval for recommendations.

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Text + profile embeddings | ✅ | TF-IDF/SVD on corpus + query text (`embedding.py`) |
| Approximate nearest-neighbor search | ✅ | sklearn `NearestNeighbors` cosine |
| Retrieves relevant candidates | ✅ | Top-200 candidates → rerank |

**Notes:** Sentence-BERT/FAISS listed as optional; TF-IDF/SVD is acceptable for prototype.

---

## Capability 3 — Engagement Scoring & Multi-Stage Ranking

**Brief requires:** Composite score; candidate → score → rerank; ≥1 ranking metric.

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Relevance component | ✅ | Embedding similarity |
| Community health signals | ✅ | `score_health` (stars, activity) |
| Visibility potential | ✅ | `score_visibility` |
| Estimated effort | ✅ | `score_effort` |
| Multi-stage pipeline | ✅ | ANN candidates → augment → rerank (`ranking.py`) |
| Ranking metric reported | ✅ | NDCG@10 in UI + benchmarks |
| “Why ranked here?” | ✅ | Component scores on each card |

---

## Capability 4 — Adaptive Learning from Feedback

**Brief requires:** Learning from engage/skip/bookmark; measurable improvement over 50+ rounds.

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Feedback loop | ✅ | Engage / Bookmark / Skip → bandit update |
| Learning mechanism | ✅ | Thompson sampling (`reinforcement_learning.py`) |
| 50+ simulated rounds | ✅ | 60-round benchmark |
| Measurable improvement | ✅ | Reward +0.7 (1.0 vs 0.3 last-10 avg) |

---

## Capability 5 — Batch Analytics & Trend Detection

**Brief requires:** Batch processing; trending topics; active communities; volume over time; category distributions; dashboard charts.

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Batch / distributed framework | ✅ | DuckDB SQL batch queries (`analytics.py`) |
| Trending topics / domains | ✅ | WoW rising domains chart |
| Volume over time | ✅ | Daily volume line chart |
| Category / domain distribution | ✅ | Volume by domain bar chart |
| Dashboard display | ✅ | Analytics tab |

---

## Capability 6 — Dashboard & Engagement Brief

**Brief requires:** Ranked list; scores; Why this?; LLM suggested actions; feedback controls; trends; downloadable brief.

| Required sub-item | Status | Evidence |
|-------------------|--------|----------|
| Ranked opportunities + scores | ✅ | Discover tab cards |
| “Why this?” explanations | ✅ | Component score breakdown per card |
| LLM-generated engagement ideas | ✅ | **Free** Gemini or Groq API (no payment); smart templates if no key |
| Engage / Skip / Bookmark | ✅ | Card action buttons |
| Trend visualisations | ✅ | Analytics tab |
| Downloadable brief (PDF/CSV) | ✅ | Export buttons in Analytics |
| Non–data-scientist usable UI | ✅ | Persona presets, plain-language cards |

**Action for full LLM credit:** Add a **free** `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) (no credit card). Or use Groq free tier. Smart templates work with zero keys.

---

## Persona pass criteria (brief)

| Persona | Pass criteria (summary) | Status |
|---------|---------------------------|--------|
| **Sofia** | ≥3 GFI in top-10; no C++/Rust; ML focus | ✅ PASS |
| **David** | ≥5 DevOps/K8s in top-10 | ✅ PASS |
| **Lina** | Visibility ≥ relevance (recency/velocity) | ✅ PASS |
| **Raj** | ≥4 devtools/B2B/Cloud in top-10 | ✅ PASS |

All 6 capabilities: **PASS** for each persona in `benchmark_results.json`.

**Caveat:** Personas reference Reddit; app uses GitHub + HN only. Ranking still meets numeric pass thresholds on live data.

---

## Grading rubric dimensions (100 pts)

| Dimension | Full credit requires | Status |
|-----------|---------------------|--------|
| Data Pipeline (15) | Streaming works; dedup; ≥10k records | ✅ |
| Matching & Ranking (20) | Retrieval + reasonable order + metric | ✅ |
| Adaptive Learning & Techniques (15) | 2 techniques + benchmarked improvement | ✅ |
| Hosting & Deployment (20) | Clean UI; explain feature; hosted | ✅ (redeploy after latest UI fix) |
| Brief & Demo (30) | 4-page brief; organized demo | ✅ brief / ⏳ demo |

---

## Known gaps / mitigations

1. **Card font size** — Fixed with inline 10px/11px fact chips; redeploy required for live URL.
2. **LLM suggestions** — Template fallback works; set `OPENAI_API_KEY` for brief’s exact “LLM-generated” wording.
3. **Reddit / GH Archive** — Not implemented; GitHub + HN satisfies minimum source count.
4. **Hidden personas** — Not testable locally; 4 provided personas + 15 domains covered.

---

## Pre-submission checklist

- [ ] Redeploy Streamlit Cloud after latest push
- [ ] Verify card fact text is small on live app
- [ ] Optional: set `OPENAI_API_KEY` for AI suggestions
- [ ] Upload ZIP to Canvas by Fri Jun 5 11:59 PM PT
- [ ] Saturday demo using `DEMO.md`
