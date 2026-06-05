# EngageIQ — 10-Minute Live Demo Script

**Student:** Shivneel Prasad · **URL:** https://engageiq-bax423-final.streamlit.app/

## 1. Problem (30 sec)
EngageIQ helps professionals decide *where* to invest limited time online — ranking engagement opportunities from GitHub and Hacker News by relevance, community health, visibility, and effort.

## 2. Data pipeline (2 min)
- Show **11,964-record offline snapshot** (1,964 live GitHub + HN + synthetic backup)
- Click **Ingest next streaming batch** → simulated streaming
- Show **Bloom/CMS/HLL sketch metrics** → dedup + trend counting (BAX-423 sketching)

## 3. Embeddings + ranking (2 min)
- Select persona **Sofia (ML Student)** → Load profile
- Show **ranked top-10** with component scores (Why this?)
- Point out **NDCG@10** metric in UI

## 4. Adaptive learning (1 min)
- Click **Run simulation** (60 rounds) → show NDCG improvement
- Click **Engage / Skip** on one card → bandit updates domain weights

## 5. Batch analytics (1 min)
- Scroll to **trend charts** + **week-over-week rising domains** (Lina persona)

## 6. Export + personas (2 min)
- Switch to **David (DevOps)** → infra-focused top-10
- **Export brief (PDF)** → downloadable weekly summary
- Mention **4 personas tested** + pass/fail table in `brief.pdf`

## 7. Architecture wrap-up (1 min)
- DuckDB batch store, TF-IDF/SVD + ANN, Thompson sampling bandit
- Hosted on Streamlit Cloud; code on GitHub

## Backup if live app fails
Run locally: `cd code && py -m streamlit run app.py`
