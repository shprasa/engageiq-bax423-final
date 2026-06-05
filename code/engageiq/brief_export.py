from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass(frozen=True)
class BriefConfig:
    top_k: int = 20


def export_brief_csv(
    ranked_df: pd.DataFrame, trends_by_domain: pd.DataFrame, out_path: Path, cfg: BriefConfig
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = ranked_df.head(cfg.top_k).copy()
    top.insert(0, "exported_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    trends = trends_by_domain.copy()
    trends.insert(0, "exported_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    top_path = out_path.with_suffix(".csv")
    trends_path = out_path.with_name(out_path.stem + "_trends.csv")

    top.to_csv(top_path, index=False)
    trends.to_csv(trends_path, index=False)
    return top_path


def export_brief_pdf(
    ranked_df: pd.DataFrame,
    trends_by_domain: pd.DataFrame,
    out_path: Path,
    persona_name: str,
    cfg: BriefConfig,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, topMargin=0.6 * inch)
    story: list = []

    story.append(Paragraph("<b>EngageIQ Weekly Engagement Brief</b>", styles["Title"]))
    story.append(
        Paragraph(
            f"Persona: {persona_name}<br/>Exported: {datetime.now():%Y-%m-%d %H:%M}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    top = ranked_df.head(cfg.top_k)
    rows = [["#", "Title", "Domain", "Score", "Est. effort"]]
    for i, row in top.iterrows():
        effort = "Low" if float(row.get("score_effort", 0.5)) < 0.35 else "Medium"
        rows.append(
            [
                str(i + 1),
                str(row.get("title", ""))[:60],
                str(row.get("domain", ""))[:22],
                f"{float(row.get('score_final', 0)):.2f}",
                effort,
            ]
        )

    table = Table(rows, colWidths=[0.35 * inch, 3.0 * inch, 1.4 * inch, 0.7 * inch, 0.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4057")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("<b>Trending domains (last 30 days)</b>", styles["Heading2"]))
    trend_rows = [["Domain", "Volume", "Avg upvotes"]]
    for _, tr in trends_by_domain.head(8).iterrows():
        trend_rows.append(
            [
                str(tr.get("domain", "")),
                str(int(tr.get("n", 0))),
                f"{float(tr.get('avg_upvotes', 0)):.1f}",
            ]
        )
    t2 = Table(trend_rows, colWidths=[2.5 * inch, 1.0 * inch, 1.2 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A7C59")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(t2)

    doc.build(story)
    return pdf_path
