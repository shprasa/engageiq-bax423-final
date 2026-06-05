# EngageIQ — Rubric Compliance Checklist (Final)

**Student:** Shivneel Prasad  
**Live URL:** https://engageiq-bax423-final.streamlit.app/  
**Repo:** https://github.com/shprasa/engageiq-bax423-final  
**ZIP:** `C:\Users\prasa\OneDrive\Desktop\Prasad_Shivneel_BAX423_Final.zip`

## Submission package

| Item | Required | Status |
|------|----------|--------|
| `code/` + `requirements.txt` + README | Yes | Done |
| Single-command run | Yes | `cd code && py -m streamlit run app.py` |
| `data/` snapshot >=10,000 | Yes | 11,964 rows |
| `brief.pdf` <=4 pages | Yes | Regenerated with live stats |
| `prompts.md` | Yes | Done |
| Live public URL | Yes | Streamlit Cloud |

## Live data (fixed for cloud deploy)

| Item | Status |
|------|--------|
| Local CSV live rows | 1,964 (928 GitHub + 1,036 HN) |
| Bundled `code/data/live_opportunities.csv` | 692 KB — deployed to GitHub |
| Full snapshot on GitHub | `data/` + `code/data/` both updated |
| App default | "Show live API opportunities only" ON |

**Why you saw 0 live:** GitHub had the old synthetic-only CSV; Streamlit Cloud never received the scraped data. Fixed by pushing live CSV + auto-reload logic.

## 6 core capabilities

| # | Capability | Status |
|---|------------|--------|
| 1 | Multi-source ingest + streaming + dedup | Done |
| 2 | Embeddings + ANN retrieval | Done |
| 3 | Multi-stage ranking + NDCG@10 | Done |
| 4 | Adaptive learning / RL (50+ rounds) | Done | Thompson sampling contextual bandit |
| 5 | Batch analytics + trends | Done |
| 6 | Dashboard + brief export | Done |

## UI features (production pass)

| Feature | Status |
|---------|--------|
| Professional styling (hero, cards, badges) | Done |
| Bookmarks tab | Done |
| Activity log + CSV export | Done |
| Real URLs + descriptions on cards | Done |
| Source-specific titles + decision facts on cards | Done |
| Automated user-test loop (`scripts/user_test_loop.py`) | Done — 11/11 checks |
| Live vs offline backup badges | Done |

## Before Canvas upload (your actions)

- [ ] Open live URL — confirm hero shows **1,964 Live API** (after Streamlit redeploy ~2 min)
- [ ] Upload `Prasad_Shivneel_BAX423_Final.zip` to Canvas
- [ ] Attend Saturday demo using `DEMO.md`

## Redeploy note

If Streamlit still shows old data: **Manage app → Reboot** or wait for auto-redeploy from GitHub push (completed).
