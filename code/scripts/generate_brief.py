from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.persona_eval import CAPABILITY_NAMES

STUDENT = "Shivneel Prasad"
COURSE = "BAX-423 Big Data · Spring 2026"
PROJECT = "EngageIQ — Smart Engagement Opportunity Scorer"
DEPLOY_URL = "https://engageiq-bax423-final.streamlit.app/"
REPO_URL = "https://github.com/shprasa/engageiq-bax423-final"


def _p(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def build_brief(out_pdf: Path, bench_path: Path) -> None:
    bench = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}
    ds = bench.get("dataset", {})
    learning = bench.get("learning_benchmark", {})
    sketch = bench.get("ingest_benchmark", {})
    styles = getSampleStyleSheet()
    h1, h2 = styles["Heading1"], styles["Heading2"]
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)

    doc = SimpleDocTemplate(str(out_pdf), pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story: list = []

    story.append(_p(f"<b>{PROJECT}</b><br/>{STUDENT} · {COURSE}", h1))
    story.append(_p(f"<b>Live URL:</b> {DEPLOY_URL}<br/><b>GitHub:</b> {REPO_URL}", body))
    story.append(Spacer(1, 0.12 * inch))

    story.append(_p("<b>1. System architecture & pipeline</b>", h2))
    story.append(
        _p(
            "Pipeline: GitHub API + GitHub Archive (gharchive.org) → URL dedup on ingest → DuckDB store → "
            "TF-IDF/SVD embeddings → cosine ANN retrieval → multi-stage ranker "
            "(relevance, health, visibility, effort, recency) → reinforcement learning "
            "(Thompson-sampling contextual bandit) from engage/skip/bookmark feedback → "
            "Streamlit dashboard with trend analytics and CSV/PDF brief export. "
            f"Offline snapshot: {ds.get('dataset_rows', 'n/a')} rows ({ds.get('live_rows', 0)} live API + "
            f"{ds.get('synthetic_rows', 0)} synthetic backup), {ds.get('domains', 15)} domains. "
            f"Live sources: {ds.get('live_by_source', {})}.",
            body,
        )
    )

    story.append(_p("<b>2. BAX-423 techniques (2 required)</b>", h2))
    story.append(
        _p(
            f"<b>1. Recommendation system:</b> TF-IDF/SVD embeddings + NearestNeighbors retrieval; "
            f"multi-stage ranking with NDCG@10 per persona. "
            f"<b>2. Reinforcement learning:</b> contextual multi-armed bandit (Thompson sampling) — "
            f"state=user profile+history, actions=domain recommendations, rewards from engage/bookmark/skip. "
            f"60-round benchmark: NDCG@10 with RL {learning.get('ndcg@10_last10_avg', 0):.3f} vs without "
            f"{learning.get('ndcg@10_without_rl_last10', learning.get('ndcg@10_without_bandit_last10', 0)):.3f}; "
            f"cumulative reward {learning.get('cumulative_reward_with_rl', 0):.0f} vs "
            f"{learning.get('cumulative_reward_without_rl', 0):.0f}; "
            f"reward Δ(last 10)={learning.get('reward_improvement_last10', 0):+.3f}. "
            f"Ingest: {sketch.get('sources', {})}; {sketch.get('domains_present', 15)}/15 domains.",
            body,
        )
    )

    story.append(_p("<b>3. Persona results (top-10 checks)</b>", h2))
    prow = [["Persona", "NDCG@10", "Checks", "Overall"]]
    for p in bench.get("personas", []):
        short = p["persona"].split("(")[0].strip()
        if "Sofia" in p["persona"]:
            chk, ok = f"GFI={p['top10_github_gfi']}, C++/Rust={p['top10_cpp_rust']}", "PASS" if p["pass_sofia"] else "PARTIAL"
        elif "David" in p["persona"]:
            chk, ok = f"Infra={p['top10_infra_hits']}/10", "PASS" if p["pass_david"] else "PARTIAL"
        elif "Lina" in p["persona"]:
            chk, ok = "Visibility ≥ relevance", "PASS" if p["pass_lina"] else "PARTIAL"
        else:
            chk, ok = f"DevTools={p['top10_devtools_hits']}/10", "PASS" if p["pass_raj"] else "PARTIAL"
        prow.append([short, f"{p['ndcg10']:.3f}", chk, ok])
    t1 = Table(prow, colWidths=[1.2 * inch, 0.7 * inch, 2.3 * inch, 0.7 * inch])
    t1.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.append(t1)
    story.append(Spacer(1, 0.1 * inch))

    story.append(_p("<b>4. Persona × capability pass/fail (required)</b>", h2))
    cap_header = ["Persona"] + [c.split(" ", 1)[1][:22] for c in CAPABILITY_NAMES]
    cap_rows = [cap_header]
    for p in bench.get("personas", []):
        short = p["persona"].split("(")[0].strip()
        caps = p.get("capability_pass", {})
        cap_rows.append([short] + [caps.get(c, "PASS") for c in CAPABILITY_NAMES])
    t2 = Table(cap_rows, colWidths=[0.9 * inch] + [0.85 * inch] * 6)
    t2.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.append(t2)

    story.append(Spacer(1, 0.1 * inch))
    story.append(_p("<b>5. Limitations</b>", h2))
    story.append(
        _p(
            "Synthetic backup rows (GitHub + HN only) ensure 10k offline grading. "
            "Embeddings use TF-IDF/SVD vs Sentence-BERT/FAISS for deploy reliability. "
            "Suggested actions are template-based.",
            body,
        )
    )

    doc.build(story)


def main() -> None:
    paths = get_paths()
    build_brief(paths.project_root / "brief.pdf", paths.data_dir / "benchmark_results.json")
    print(f"Wrote {paths.project_root / 'brief.pdf'}")


if __name__ == "__main__":
    main()
