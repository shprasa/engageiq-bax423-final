"""Convert brief.docx to brief.pdf (no Word/LibreOffice required)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph as RlParagraph
from reportlab.platypus import SimpleDocTemplate, Spacer

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _runs_to_markup(paragraph: Paragraph) -> str:
    parts: list[str] = []
    for run in paragraph.runs:
        text = _escape(run.text)
        if not text:
            continue
        if run.bold and run.italic:
            text = f"<b><i>{text}</i></b>"
        elif run.bold:
            text = f"<b>{text}</b>"
        elif run.italic:
            text = f"<i>{text}</i>"
        parts.append(text)
    return "".join(parts) if parts else _escape(paragraph.text)


def _style_for(paragraph: Paragraph, styles: dict[str, ParagraphStyle]) -> ParagraphStyle:
    name = (paragraph.style.name if paragraph.style else "").lower()
    text = paragraph.text.strip()
    if name.startswith("heading 1") or re.match(r"^\d+\.\s", text):
        return styles["h1"]
    if name.startswith("heading 2"):
        return styles["h2"]
    if name.startswith("heading 3"):
        return styles["h3"]
    if text.startswith("- ") or text.startswith("• "):
        return styles["bullet"]
    if text.startswith("cd ") or "streamlit run" in text or "pip install" in text:
        return styles["code"]
    return styles["body"]


def convert(docx_path: Path, pdf_path: Path) -> Path:
    doc = Document(str(docx_path))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle(
            "BriefH1",
            parent=base["Heading1"],
            fontSize=13,
            leading=16,
            spaceAfter=8,
            spaceBefore=10,
        ),
        "h2": ParagraphStyle(
            "BriefH2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceAfter=6,
            spaceBefore=8,
        ),
        "h3": ParagraphStyle(
            "BriefH3",
            parent=base["Heading3"],
            fontSize=10,
            leading=13,
            spaceAfter=4,
            spaceBefore=6,
        ),
        "body": ParagraphStyle(
            "BriefBody",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "BriefBullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            leftIndent=14,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "BriefCode",
            parent=base["Code"],
            fontSize=9,
            leading=11,
            leftIndent=10,
            spaceAfter=2,
            fontName="Courier",
        ),
    }

    story: list = []
    for paragraph in doc.paragraphs:
        text = _runs_to_markup(paragraph).strip()
        if not text:
            story.append(Spacer(1, 0.08 * inch))
            continue
        style = _style_for(paragraph, styles)
        display = text[2:] if text.startswith("- ") else text
        if style is styles["bullet"] and not display.startswith("•"):
            display = f"• {display.lstrip('• ')}"
        story.append(RlParagraph(display, style))

    doc_pdf = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    doc_pdf.build(story)
    return pdf_path


def main() -> None:
    docx = PROJECT_ROOT / "brief.docx"
    pdf = PROJECT_ROOT / "brief.pdf"
    if len(sys.argv) > 1:
        docx = Path(sys.argv[1])
    if len(sys.argv) > 2:
        pdf = Path(sys.argv[2])
    if not docx.exists():
        raise SystemExit(f"Missing {docx}")
    out = convert(docx, pdf)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
