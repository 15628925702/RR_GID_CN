from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

OUT = Path(r"H:\RR_GID_CN_data\reports")
OUT.mkdir(parents=True, exist_ok=True)
HTML = OUT / "stage_summary_20260909_v2.html"
PDF = OUT / "stage_summary_20260909_v2.pdf"

rows = [
 ("Synthetic B2000", "synthetic_main_b2000_v2", 80, "AUDIT_B2000_v2.md: ACCEPTED", "RR-GID 3.293 [2.605, 3.981]; Uniform 14.970 [7.183, 22.756]"),
 ("Synthetic B4000", "synthetic_main_b4000_v1", 80, "AUDIT_B4000_v1.md: PASS", "RR-GID 5.321 [2.439, 8.203]; Uniform 21.431 [12.543, 30.319]"),
 ("Synthetic B8000", "synthetic_main_b8000_v1", 80, "AUDIT_B8000_v1.md: PASSED", "RR-GID 18.283 [8.545, 28.022]; Uniform 60.233 [35.195, 85.270]"),
 ("Synthetic B16000", "synthetic_main_b16000_v2", 80, "AUDIT_B16000_v2.md: PASS", "RR-GID 27.873 [11.295, 44.452]; Uniform 150.230 [98.961, 201.500]"),
 ("Synthetic B32000", "synthetic_main_b32000_v1", 80, "AUDIT_B32000_v1.md: PASS", "RR-GID 55.257 [27.936, 82.578]; Uniform 212.659 [119.696, 305.622]"),
 ("Nonlinearity v1", "nonlinearity_v1", 240, "AUDIT_nonlinearity_v1.md: PASS", "CI available by alpha/method in report.json (16 cells)"),
 ("Reuse v2", "reuse_v2", 200, "AUDIT_reuse_v2.md: PASS", "CI available by sequence/method in report.json (10 cells)"),
 ("Gas semisynthetic B400", "gas_semisynthetic_B400_v3", 80, "AUDIT_B400_v3.md: PASS", "Audit records structural/provenance PASS; no CI in certificate"),
 ("Gas semisynthetic B800", "gas_semisynthetic_B800_v1", 80, "AUDIT_B800_v1.md: PASS", "Audit records structural/provenance PASS; no CI in certificate"),
 ("Gas semisynthetic B1600", "gas_semisynthetic_B1600_v1", 80, "AUDIT_B1600_v1.md: PASS", "Audit records structural/provenance PASS; no CI in certificate"),
 ("Gas semisynthetic B3200", "gas_semisynthetic_B3200_v1", 80, "AUDIT_B3200_v1.md: PASS", "RR-GID 0.06304 [0.05813, 0.06795]; Uniform 0.07657 [0.07256, 0.08059]"),
 ("P10 gas natural clean", "gas_natural_v1_clean", 720, "AUDIT_P10_v1_clean.md: PASS", "Structural/provenance audit; no reliable CI reported"),
]

def make_html():
    trs = ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in r) + '</tr>' for r in rows)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>RR_GID_CN 阶段总结 v2</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;color:#1f2937;line-height:1.55;max-width:1100px;margin:28px auto;padding:0 22px}}h1{{color:#123b5d}}h2{{color:#0b6374;border-bottom:1px solid #cbd5e1;padding-bottom:4px;margin-top:24px}}.meta{{color:#64748b}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #cbd5e1;padding:6px;text-align:left;vertical-align:top}}th{{background:#e6f4f1}}.note{{background:#f8fafc;border-left:4px solid #0b8793;padding:9px;margin:10px 0}}</style></head><body>
<h1>RR_GID_CN 论文实验阶段总结报告</h1><p class="meta">v2 | 2026-09-09 | 只读汇总，未重跑实验</p>
<h2>已通过批次</h2><table><tr><th>批次</th><th>目录</th><th>rows</th><th>审计证书</th><th>关键 CI（risk_ratio_raw，95%）</th></tr>{trs}</table>
<h2>结论与边界</h2><div class="note"><b>状态：</b>列出的 12 个批次均有审计 PASS/ACCEPTED，累计 rows=1880（synthetic 400 + nonlinearity 240 + reuse 200 + gas semisynthetic 320 + P10 720）。P10 clean 是 3 campaigns x 3 budgets x 20 reps x 4 methods = 720 行。</div>
<p>CI 仅在证书或对应 report.json 可可靠读取时列出；nonlinearity_v1 与 reuse_v2 的完整 CI 单元格保留在各自 report.json。Gas B400/B800/B1600 的审计证书未提供 CI，故不臆填。</p>
<p><b>重要方法学限制：</b>cached-QMC 不能等同 adaptive gold convergence；正式论文仍需严格 paired-SE，并在适用时补充 gold adaptive/gold convergence 证据。阶段审计 PASS 不等于论文 publication eligibility 自动批准。</p>
<p><b>数据处置：</b>旧 <code>gas_natural_v1</code> 中存在冲突 rows，已 quarantine；该旧目录及 quarantine rows 不可作为结果。结果口径仅使用 <code>gas_natural_v1_clean</code>（P10 audit PASS）。</p>
<h2>来源</h2><p>代码/脚本：G:\\0-newResearch\\4.RR_GID_CN；运行目录：H:\\RR_GID_CN_data\\active\\paper_runs；本报告：H:\\RR_GID_CN_data\\reports。未修改 rows/config，未启动实验。</p>
</body></html>'''

def build_pdf():
    pdfmetrics.registerFont(TTFont('SC', r'C:\Windows\Fonts\simhei.ttf'))
    s = getSampleStyleSheet(); title=ParagraphStyle('t',parent=s['Title'],fontName='SC',fontSize=18,leading=23,alignment=TA_CENTER,textColor=colors.HexColor('#123b5d')); h=ParagraphStyle('h',parent=s['Heading2'],fontName='SC',fontSize=12,leading=16,textColor=colors.HexColor('#0b6374')); body=ParagraphStyle('b',parent=s['BodyText'],fontName='SC',fontSize=8.5,leading=13)
    table_head = ParagraphStyle('th', parent=body, fontSize=7, leading=9, alignment=TA_CENTER, wordWrap='CJK')
    table_cell = ParagraphStyle('td', parent=body, fontSize=6.6, leading=8.5, wordWrap='CJK')
    story=[Paragraph('RR_GID_CN 论文实验阶段总结报告 v2',title),Paragraph('日期：2026-09-09；只读汇总，未重跑实验',body),Spacer(1,8),Paragraph('已通过批次',h)]
    headers = ['批次', '目录', 'rows', '审计', '关键 CI']
    data = [[Paragraph(text, table_head) for text in headers]]
    data += [[Paragraph(escape(str(value)), table_cell) for value in row] for row in rows]
    # 185 mm stays within the 186 mm A4 content width after margins.
    t=Table(data,colWidths=[29*mm,38*mm,11*mm,36*mm,71*mm],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f4f1')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cbd5e1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)])); story += [t,Spacer(1,10),Paragraph('结论与边界',h)]
    for p in ['列出的 12 个批次均有审计 PASS/ACCEPTED，累计 rows=1880（synthetic 400 + nonlinearity 240 + reuse 200 + gas semisynthetic 320 + P10 720）。P10 clean 为 720 行（3 campaigns x 3 budgets x 20 reps x 4 methods）。','cached-QMC 不能等同 adaptive gold convergence；正式论文仍需严格 paired-SE，并在适用时补充 gold adaptive/gold convergence 证据。','旧 gas_natural_v1 的冲突 rows 已 quarantine，不可作为结果；结果口径仅使用 gas_natural_v1_clean。','CI 只在证书或 report.json 可可靠读取时列出；未提供者不臆填。未修改 rows/config，未启动实验。']: story.append(Paragraph(p,body))
    SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=12*mm,leftMargin=12*mm,topMargin=12*mm,bottomMargin=12*mm).build(story)

HTML.write_text(make_html(),encoding='utf-8'); build_pdf(); print(HTML); print(PDF)
