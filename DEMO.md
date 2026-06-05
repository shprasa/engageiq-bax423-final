# EngageIQ — 10-Minute Live Demo Script

**Student:** Shivneel Prasad · **URL:** https://engageiq-bax423-final.streamlit.app/

## 1. Problem (30 sec)
EngageIQ helps professionals decide *where* to invest limited time online — ranking engagement opportunities from GitHub and Hacker News by relevance, community health, visibility, and effort.

## 2. Data + ingest (1 min)
- Show **10,000-record snapshot** (1,964 live GitHub + HN, **15 domains**)
- Mention multi-source ingest with URL dedup into DuckDB

## 3. Recommendation system (2 min)
- Select persona **Sofia (ML Student)** → Load profile
- **Discover** tab: ranked top-10 with component scores and suggested actions
- Point out **NDCG@10** metric

## 4. Reinforcement learning (2 min)
- Sidebar **RL policy** — domain weights from Thompson sampling
- Click **Engage / Skip** on a card → bandit updates
- **Analytics** tab → **Run RL benchmark** → show **+0.7 reward** vs no-RL baseline

## 5. Analytics + export (2 min)
- Trend charts + week-over-week rising domains
- **Export PDF brief**

## 6. Personas (2 min)
- Switch to **David (DevOps)** → infra-focused top-10
- Mention **4 personas** + 6-capability pass table in `brief.pdf`

## 7. Wrap-up (30 sec)
- **Two BAX-423 techniques:** recommendation (embeddings + ranking) + RL bandit
- Streamlit Cloud + GitHub repo

## Backup if live app fails
Run locally: `cd code && py -m streamlit run app.py`
