# EngageIQ — Rubric Checklist (Final)

**Student:** Shivneel Prasad  
**Live URL:** https://engageiq-bax423-final.streamlit.app/  
**Full audit:** See `RUBRIC_COMPLIANCE.md` (cross-checked against official one-pager + EngageIQ brief)

## Critical rules

- [x] All **6 core capabilities** present (missing any caps at 60/100)
- [x] **2 BAX-423 techniques** integrated and benchmarked (Recommendation + RL)
- [x] Code runs with single command
- [ ] Live URL verified after latest redeploy (user)

## 6 core capabilities — required items

| # | Capability | Required (from brief) | Status |
|---|------------|----------------------|--------|
| 1 | Multi-source ingest + **streaming** + dedup | ≥2 sources; stream pipeline; dedup; structured store; 10k rows / 15 domains | ✅ |
| 2 | Embeddings + ANN retrieval | Dense embeddings + nearest-neighbor retrieval | ✅ |
| 3 | Multi-stage ranking + metric | Composite scores; candidate→score→rerank; NDCG@10 | ✅ |
| 4 | Adaptive learning / RL | Feedback loop; **50+ rounds**; measurable improvement | ✅ |
| 5 | Batch analytics + trends | Batch queries; trends; volume; domain breakdown | ✅ |
| 6 | Dashboard + brief export | Ranked list; Why this?; **suggested actions**; feedback; trends; PDF/CSV | ✅ |

\* Capability 6 LLM: use free **Gemini** or **Groq** API key (see `.env.example`). No payment required. Without any key, interest-aware templates still power suggestions.

## Data

| Item | Status |
|------|--------|
| 10,000 rows | ✅ |
| 15/15 domains | ✅ |
| GitHub + Hacker News sources | ✅ |
| 1,964 live API rows | ✅ |

## Personas (all PASS on 6 capabilities)

| Persona | Status |
|---------|--------|
| Sofia (ML / portfolio) | ✅ |
| David (DevOps) | ✅ |
| Lina (trend spotter) | ✅ |
| Raj (startup / devtools) | ✅ |

## Before Canvas + demo

- [ ] Redeploy Streamlit — confirm compact card fonts on live app
- [ ] Upload `Prasad_Shivneel_BAX423_Final.zip`
- [ ] Saturday 10-min demo (`DEMO.md`)
