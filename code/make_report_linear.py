"""Render report/REPORT.md to report/REPORT.pdf with ReportLab."""
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, PageBreak

SRC = Path('../report/REPORT.md')
OUT = Path('../report/REPORT.pdf')
ss = getSampleStyleSheet()
styles = {
    'h1': ParagraphStyle('h1', parent=ss['Heading1'], fontName='Helvetica-Bold', fontSize=17, leading=21, spaceAfter=8),
    'h2': ParagraphStyle('h2', parent=ss['Heading2'], fontName='Helvetica-Bold', fontSize=12.5, leading=15, spaceBefore=9, spaceAfter=4),
    'h3': ParagraphStyle('h3', parent=ss['Heading3'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=3),
    'body': ParagraphStyle('body', parent=ss['BodyText'], fontName='Helvetica', fontSize=10, leading=13.5, spaceAfter=4),
    'cell': ParagraphStyle('cell', parent=ss['BodyText'], fontName='Helvetica', fontSize=8.6, leading=11),
    'cellh': ParagraphStyle('cellh', parent=ss['BodyText'], fontName='Helvetica-Bold', fontSize=8.6, leading=11),
}

def esc(s):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8.4">\1</font>', s)
    return s

def is_special(line):
    return (line.startswith('#') or line.startswith('|') or line.startswith('- ') or not line.strip())

lines = SRC.read_text(encoding='utf-8').splitlines()
story = []
i = 0
while i < len(lines):
    line = lines[i]
    if not line.strip():
        story.append(Spacer(1, 4)); i += 1; continue
    if line in ('## 2. Method','## 3. Validation results','### 3.2 Historical information controls and uncertainty','## 5. Resources'):
        story.append(PageBreak())
    if line.startswith('### '):
        story.append(Paragraph(esc(line[4:]), styles['h3'])); i += 1; continue
    if line.startswith('## '):
        story.append(Paragraph(esc(line[3:]), styles['h2'])); i += 1; continue
    if line.startswith('# '):
        story.append(Paragraph(esc(line[2:]), styles['h1'])); i += 1; continue
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                rows.append(cells)
            i += 1
        ncol = max(len(r) for r in rows)
        data = []
        for ri, r in enumerate(rows):
            r = r + [''] * (ncol - len(r))
            data.append([Paragraph(esc(c), styles['cellh' if ri == 0 else 'cell']) for c in r])
        if ncol == 3:
            widths = [8.2*cm, 4.2*cm, 4.2*cm]
        elif ncol == 4:
            widths = [4.2*cm, 4.6*cm, 3.9*cm, 3.9*cm]
        elif ncol == 2:
            widths = [7*cm, 9.6*cm]
        else:
            widths = [16.6*cm/ncol] * ncol
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT', splitByRow=0)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eeeeee')),
            ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#999999')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t); story.append(Spacer(1, 5)); continue
    if line.startswith('- '):
        items = []
        while i < len(lines) and lines[i].startswith('- '):
            items.append(ListItem(Paragraph(esc(lines[i][2:]), styles['body']), leftIndent=10))
            i += 1
        story.append(ListFlowable(items, bulletType='bullet', start='circle', leftIndent=12, bulletFontSize=7))
        continue
    para = [line]
    i += 1
    while i < len(lines) and lines[i].strip() and not is_special(lines[i]):
        para.append(lines[i]); i += 1
    story.append(Paragraph(esc(' '.join(para)), styles['body']))

OUT.parent.mkdir(exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title='Selecting Reliable LLM Reasoning with Listwise Verification', author='DASE7506 MP2')
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(2*cm,1.1*cm,'DASE7506 MP2 | Prepared final v4 linear | 391/467')
    canvas.drawRightString(A4[0]-2*cm,1.1*cm,str(doc.page))
    canvas.restoreState()

doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
pages=len(PdfReader(OUT).pages)
if pages>10:
    raise RuntimeError(f'Report exceeds 10-page limit: {pages}')
print('pages',pages)
print('wrote', OUT, OUT.stat().st_size, 'bytes')
