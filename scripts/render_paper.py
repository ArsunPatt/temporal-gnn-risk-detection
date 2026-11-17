"""Export the editable Markdown paper with ReportLab; no scientific computation."""

import argparse
from html import escape
import importlib.util
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]


def font_directory():
    module = importlib.util.find_spec("matplotlib")
    if module is not None:
        return Path(module.origin).parent / "mpl-data/fonts/ttf"
    candidates = list((ROOT / ".venv").glob("lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf"))
    if not candidates:
        raise FileNotFoundError("Install the research environment or pass --font-dir with DejaVu fonts")
    return candidates[0]


def inline(text):
    text = text.replace("–", "-").replace("—", "-").replace("−", "-").replace("‑", "-")
    text = escape(text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2" color="#18527a">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def render_paper(source, output, fonts=None):
    fonts = Path(fonts) if fonts else font_directory()
    for name, filename in [("ResearchSerif","DejaVuSerif.ttf"), ("ResearchSerif-Bold","DejaVuSerif-Bold.ttf"),
                           ("ResearchSerif-Italic","DejaVuSerif-Italic.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("ResearchSerif", normal="ResearchSerif", bold="ResearchSerif-Bold", italic="ResearchSerif-Italic", boldItalic="ResearchSerif-Bold")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ResearchBody",fontName="ResearchSerif",fontSize=10,leading=14.5,spaceAfter=8,alignment=TA_LEFT)
    title = ParagraphStyle("ResearchTitle",parent=body,fontName="ResearchSerif-Bold",fontSize=19,leading=24,spaceAfter=15)
    heading = ParagraphStyle("ResearchHeading",parent=body,fontName="ResearchSerif-Bold",fontSize=13,leading=17,spaceBefore=12,spaceAfter=8,keepWithNext=True)
    subheading = ParagraphStyle("ResearchSubheading",parent=heading,fontSize=11,leading=14)
    caption = ParagraphStyle("ResearchCaption",parent=body,fontName="ResearchSerif-Italic",fontSize=9,leading=13,spaceBefore=10)
    small = ParagraphStyle("ResearchCell",parent=body,fontSize=7.5,leading=10,spaceAfter=0)
    normal_size = A4
    def footer(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#bcc8d0"))
        canvas.line(46,height-34,width-46,height-34)
        canvas.setFont("ResearchSerif",8)
        canvas.setFillColor(colors.HexColor("#425563"))
        canvas.drawString(46,height-25,"Elliptic benchmark | Temporal graph learning | Research report")
        canvas.drawString(46,23,"Known-label evaluation | Source feature availability unresolved")
        canvas.drawRightString(width-46,23,str(doc.page))
        canvas.restoreState()
    frame = Frame(46,43,A4[0]-92,A4[1]-94,id="normal",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    lines = source.read_text().splitlines()
    bylines = [line for line in lines if line.startswith("**") and "Independent Researcher" in line]
    if len(bylines) != 1:
        raise ValueError("Expected one Markdown author byline before the abstract")
    author = bylines[0].split("**", 2)[1]
    doc = BaseDocTemplate(str(output), pagesize=A4, title=lines[0].removeprefix("# "), author=author)
    doc.addPageTemplates([PageTemplate("portrait",frames=[frame],onPage=footer,pagesize=A4)])
    story=[]
    i=0
    while i < len(lines):
        line=lines[i].strip()
        if not line:
            i+=1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]),title));i+=1
        elif line.startswith("## Appendix A"):
            story.extend([PageBreak(), Paragraph(inline(line[3:]),heading)])
            i+=1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]),heading));i+=1
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]),subheading));i+=1
        elif line.startswith("!["):
            match=re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)",line)
            if not match:
                raise ValueError("Unsupported figure Markdown")
            text,path=match.groups()
            image=Image(str(source.parent/path))
            maximum_height = 300 if "editorial_intervals" in path else 390
            scale=min((A4[0]-92)/image.imageWidth, maximum_height/image.imageHeight)
            image.drawWidth=image.imageWidth*scale
            image.drawHeight=image.imageHeight*scale
            story.append(KeepTogether([image, Paragraph(inline(text), caption), Spacer(1,8)]))
            i+=1
        elif line.startswith("|"):
            table_lines=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip());i+=1
            rows=[]
            for value in table_lines:
                parts=[s.strip() for s in value.strip("|").split("|")]
                if all(re.fullmatch(r"[:\-]+",s) for s in parts):
                    continue
                rows.append([Paragraph(inline(v),small) for v in parts])
            count=len(rows[0])
            width=normal_size[0]-92
            widths=[width/count]*count
            if count==6:
                widths=[width*.19]+[width*.162]*5
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8eff3")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#748d9b")),
                ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#bcc8d0")),
                ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,10)])
        else:
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#","|","![")):
                paragraph.append(lines[i].strip());i+=1
            story.append(Paragraph(inline(" ".join(paragraph)),body))
    doc.build(story)
    print("Exported editable Markdown to",output.name)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir",type=Path)
    args=parser.parse_args()
    render_paper(ROOT/"reports/research_paper.md",ROOT/"reports/research_paper.pdf",args.font_dir)
