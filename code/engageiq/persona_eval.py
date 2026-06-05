from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bandit import BetaBandit
from .embedding import build_index
from .ranking import RankConfig, ndcg_at_k, rerank


PERSONAS: dict[str, str] = {
    "Sofia (ML Student / Portfolio Builder)": (
        "Machine learning, NLP, data pipelines, beginner-friendly open source, good first issues, "
        "Python, pandas, GitHub issues, Reddit ML discussion."
    ),
    "David (DevOps / Niche Community)": (
        "Kubernetes, Terraform, CI/CD, observability, cloud-native infra, high-activity repos, "
        "few contributors, Reddit r/devops and r/kubernetes."
    ),
    "Lina (Data Journalist / Trend Spotter)": (
        "Trending repos, viral discussions, emerging tools, fast-growing communities, recency, velocity, "
        "Hacker News, GitHub trending, Reddit multi-domain."
    ),
    "Raj (Startup Founder / Marketing-Focused)": (
        "Developer tools, APIs, CLI tools, open-source business, B2B SaaS, discussions where devtools are relevant, "
        "Reddit r/programming r/SideProject r/startups."
    ),
}


@dataclass
class PersonaResult:
    persona: str
    ndcg10: float
    top10_github_gfi: int
    top10_cpp_rust: int
    top10_ml_hits: int
    top10_infra_hits: int
    top10_devtools_hits: int
    pass_sofia: bool
    pass_david: bool
    pass_lina: bool
    pass_raj: bool


def _rank_for_persona(df: pd.DataFrame, interest: str, bandit: BetaBandit | None = None) -> pd.DataFrame:
    index = build_index(df)
    idxs, dists = index.query(interest, top_k=RankConfig().candidate_k)
    candidates = df.iloc[idxs].copy().reset_index(drop=True)
    rel = np.clip(1.0 - np.asarray(dists, dtype=float), 0.0, 1.0)
    rng = np.random.default_rng(42)
    return rerank(candidates, rel, bandit, rng, RankConfig(), interest_text=interest)


def evaluate_personas(df: pd.DataFrame) -> list[PersonaResult]:
    results: list[PersonaResult] = []

    for name, interest in PERSONAS.items():
        ranked = _rank_for_persona(df, interest)
        top10 = ranked.head(10)

        gfi = int(
            (
                (top10["source"] == "github")
                & (top10["good_first_issue"].fillna(0).astype(int) == 1)
            ).sum()
        )
        cpp_rust = int(
            top10["lang"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(["c++", "rust"])
            .sum()
        )
        ml_hits = int(top10["domain"].astype(str).str.contains("Machine Learning|AI Research", case=False).sum())
        infra_hits = int(top10["domain"].astype(str).str.contains("DevOps", case=False).sum())
        devtools_hits = int(
            top10["domain"].astype(str).str.contains("Developer Tools|B2B SaaS|Cloud APIs", case=False).sum()
        )

        labels = [
            1 if any(tok.lower() in str(ranked.loc[i, "domain"]).lower() for tok in interest.split(","))
            else 0
            for i in range(min(10, len(ranked)))
        ]
        ndcg = ndcg_at_k(labels, 10)

        pass_sofia = gfi >= 3 and cpp_rust == 0 and ml_hits >= 3
        pass_david = infra_hits >= 5
        pass_lina = float(top10["score_visibility"].mean()) >= float(top10["score_relevance"].mean())
        pass_raj = devtools_hits >= 4

        results.append(
            PersonaResult(
                persona=name,
                ndcg10=ndcg,
                top10_github_gfi=gfi,
                top10_cpp_rust=cpp_rust,
                top10_ml_hits=ml_hits,
                top10_infra_hits=infra_hits,
                top10_devtools_hits=devtools_hits,
                pass_sofia=pass_sofia,
                pass_david=pass_david,
                pass_lina=pass_lina,
                pass_raj=pass_raj,
            )
        )

    return results


def learning_benchmark(df: pd.DataFrame, interest: str, rounds: int = 60) -> dict[str, float]:
    domains = sorted(df["domain"].dropna().unique().tolist())
    rng = np.random.default_rng(42)
    index = build_index(df)

    def run(use_bandit: bool) -> list[float]:
        bandit = BetaBandit(arms=domains) if use_bandit else None
        ndcgs: list[float] = []
        for _ in range(rounds):
            idxs, dists = index.query(interest, top_k=RankConfig().candidate_k)
            cand = df.iloc[idxs].copy().reset_index(drop=True)
            rel = np.clip(1.0 - np.asarray(dists), 0.0, 1.0)
            ranked = rerank(cand, rel, bandit, rng, RankConfig(), interest_text=interest)
            if use_bandit and bandit is not None:
                chosen = int(rng.integers(0, len(ranked)))
                dom = str(ranked.loc[chosen, "domain"])
                reward = 1 if any(k in dom.lower() for k in interest.lower().split()) else 0
                bandit.update(dom, reward)
            labels = [
                1 if any(k in str(ranked.loc[i, "domain"]).lower() for k in interest.lower().split())
                else 0
                for i in range(min(10, len(ranked)))
            ]
            ndcgs.append(ndcg_at_k(labels, 10))
        return ndcgs

    without = run(use_bandit=False)
    with_b = run(use_bandit=True)
    return {
        "ndcg@10_first10_avg": float(np.mean(with_b[:10])),
        "ndcg@10_last10_avg": float(np.mean(with_b[-10:])),
        "ndcg@10_without_bandit_last10": float(np.mean(without[-10:])),
        "improvement": float(np.mean(with_b[-10:]) - np.mean(without[-10:])),
    }
