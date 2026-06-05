# EngageIQ — 10-Minute Live Demo Script

**Student:** Shivneel Prasad · **URL:** https://engageiq-bax423-final.streamlit.app/

## 1. Problem (30 sec)
EngageIQ helps professionals decide *where* to invest limited time online — ranking engagement opportunities from GitHub and Hacker News by relevance, community health, visibility, and effort.

## 2. Data pipeline (2 min)
- Show **10,000-record snapshot** (1,964 live GitHub + HN)
- Sidebar **Streaming pipeline**: click **1 · Produce batch** then **2 · Consume (dedup → store)**
- Point out dedup counters + **Bloom/CMS/HLL sketch metrics**
- Optional: enable **Try Kafka publish** if Docker Kafka is running locally

## 3. Embeddings + ranking (2 min)
- Select persona **Sofia (ML Student)** → Load profile
- Show **ranked top-10** with component scores (Why this?)
- Point out **NDCG@10** metric in UI

## 4. Adaptive learning (1 min)
- Open **RL Policy** tab → click **Run RL benchmark** → show **+0.7 reward improvement** vs random
- Click **Engage / Skip** on a card → bandit updates domain weights in sidebar

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
