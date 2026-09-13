from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

OUT = Path(r"H:\RR_GID_CN_data\reports")
OUT.mkdir(parents=True, exist_ok=True)
HTML = OUT / "stage_summary_20260909.html"
PDF = OUT / "stage_summary_20260909.pdf"

sections = [
    ("结论摘要", [
        "截至 2026-09-09，本阶段已形成可用于论文分析的独立审计 PASS 结果：Synthetic B=2000/8000/16000/32000、nonlinearity_v1、reuse_v2，以及 Gas semisynthetic B400_v3/B800_v1/B1600_v1。",
        "Synthetic B=4000 结果已生成，但当前 H 盘快照未找到同目录 AUDIT_B4000* 独立审计文件，发布前需补审计/核对。Gas semisynthetic B3200_v1 已有 80 行结果和 completed provenance，但同样尚无 AUDIT_B3200，标为未验收，不能正式完成或放行。Gas natural drift 的 batch7、batches8_9、batch10 尚未完成/尚未放行。",
    ]),
    ("阶段状态", [
        "Synthetic 主线：B=2000、8000、16000、32000 有独立审计 PASS；B=4000 仅确认结果已生成，当前快照缺同目录独立审计文件，不能写成 PASS。B2000 使用 v2；早期非 v2 目录不作为证据。",
        "机制与复用：nonlinearity_v1 PASS；reuse_v2 PASS。",
        "Gas 半合成：B400_v3、B800_v1、B1600_v1 PASS；B3200_v1 有 80 行和 completed provenance，但尚无 AUDIT_B3200，未验收。B400 v1 是 smoke，不作为证据；B400 v2 为 REJECT，不作为证据。",
        "Gas natural drift：batch7、batches8_9、batch10 尚未完成/未放行。",
        "模型资产：128 维 Gas VAEAC checkpoint 已训练，路径为 H:\\RR_GID_CN_data\\reusable\\checkpoints\\vaeac_gas_128_v1.pt。",
    ]),
    ("完成比例（按阶段）", [
        "Synthetic budget ladder：4/5 个目标预算有独立审计 PASS；B=4000 已生成但待补审计/核对。",
        "机制/复用：2/2 个已列入本阶段的目标实验已 PASS。",
        "Gas semisynthetic：3/4 个目标预算已 PASS；第 4 个 B3200 已产出但未审计，因此按“待审计”计，不计入正式完成。",
        "Gas natural drift：0/3 个批次组已完成放行。",
        "以上是分阶段比例，不合并为单一总百分比。",
    ]),
    ("核心难关与剩余任务", [
        "核心难关是补做 Synthetic B4000 与 Gas B3200 的独立审计，以及 natural drift 三个批次组的完成、质量门和放行；还需在最终汇总阶段独立核算 paired-SE 和不确定性。",
        "剩余任务：1) 补审 Synthetic B4000；2) 审计/验收 Gas B3200_v1；3) 完成并审计 batch7、batches8_9、batch10；4) 生成最终汇总表和图；5) 复核论文所需的 adaptive gold/gold convergence 证据。",
        "耗时估计：Gas 每个 batch 已实测约 4-6 分钟。B3200 审计本身取决于核查与修复，不按运行时间承诺；natural drift 按实际 runner 估计，当前不确定，不能用半合成 batch 时间直接替代。",
    ]),
    ("方法学边界与残余风险", [
        "bulk cached-QMC 行中的 gold_converged 不能解读为 adaptive gold 证明。若论文要求严格的 adaptive gold 或 gold convergence，需要额外验证。",
        "审计 PASS 证明的是各自阶段的结构、数值、配对和 provenance 门槛，不自动预批准其他预算或 natural drift 输出。",
        "最终论文汇总中的 paired-SE/不确定性必须独立核算，不能仅复用单阶段 CI。",
    ]),
    ("数据、输出与 H 盘规则", [
        "代码与配置：G:\\0-newResearch\\4.RR_GID_CN；运行结果、rows、logs、provenance 和审计材料：H:\\RR_GID_CN_data\\active\\paper_runs。",
        "复用 checkpoint：H:\\RR_GID_CN_data\\reusable\\checkpoints；本报告输出：H:\\RR_GID_CN_data\\reports\\stage_summary_20260909.html 和 .pdf。",
        "H 盘规则：正式 run 的 rows/config/cache/log/provenance 应留在同一 H 盘版本化目录内；审计只读核对，禁止以覆盖 rows/config、删除文件或启动实验来修复状态。",
    ]),
    ("核对范围与状态矛盾", [
        "本报告只读核对了 G:\\0-newResearch\\4.RR_GID_CN 与 H:\\RR_GID_CN_data\\active\\paper_runs 的现有目录、审计文件和关键 checkpoint；未启动任何实验，未修改 rows/config，未删除文件。",
        "未发现数据状态之间的实质矛盾。B4000“结果已生成但缺同目录独立审计文件”、Gas B3200“已有产出但未验收”、natural drift“未完成/未放行”、B400 v1 smoke 与 B400 v2 REJECT 均按其证据等级单独表述。",
    ]),
]

def make_html():
    blocks = []
    for title, paras in sections:
        blocks.append(f"<section><h2>{escape(title)}</h2>" + "".join(f"<p>{escape(p)}</p>" for p in paras) + "</section>")
    return """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>RR_GID_CN 阶段总结报告 2026-09-09</title>
<style>@page{size:A4;margin:18mm 17mm}body{font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;color:#1f2937;line-height:1.65;max-width:900px;margin:0 auto}h1{font-size:25px;margin:0 0 4px;color:#123b5d}h2{font-size:16px;color:#0b6374;border-bottom:1px solid #cbd5e1;padding-bottom:4px;margin:22px 0 8px}p{font-size:13px;margin:6px 0}.meta{color:#64748b;font-size:12px;border-left:4px solid #0b8793;padding-left:10px;margin:12px 0 20px}.tag{display:inline-block;background:#e6f4f1;color:#075e63;padding:2px 8px;margin-right:5px;border-radius:3px;font-size:12px}section{break-inside:avoid}@media print{body{max-width:none}h2{break-after:avoid}}</style></head><body><h1>RR_GID_CN 论文实验阶段总结报告</h1><div class='meta'>日期：2026-09-09　范围：只读审计与状态汇总　<span class='tag'>中文为主</span><span class='tag'>未启动实验</span></div>""" + ''.join(blocks) + "</body></html>"

def build_pdf():
    pdfmetrics.registerFont(TTFont('NotoSC', r'C:\Windows\Fonts\simhei.ttf'))
    pdfmetrics.registerFont(TTFont('NotoSC-Bold', r'C:\Windows\Fonts\simhei.ttf'))
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], fontName='NotoSC-Bold', fontSize=19, leading=25, textColor=colors.HexColor('#123b5d'), alignment=TA_CENTER, spaceAfter=5)
    meta = ParagraphStyle('meta', parent=styles['Normal'], fontName='NotoSC', fontSize=9, leading=14, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER, spaceAfter=12)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontName='NotoSC-Bold', fontSize=13, leading=18, textColor=colors.HexColor('#0b6374'), spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle('body', parent=styles['BodyText'], fontName='NotoSC', fontSize=9.5, leading=15, spaceAfter=5)
    story = [Paragraph('RR_GID_CN 论文实验阶段总结报告', title), Paragraph('日期：2026-09-09　范围：只读审计与状态汇总　未启动实验', meta)]
    for heading, paras in sections:
        story.append(Paragraph(escape(heading), h2))
        for p in paras:
            story.append(Paragraph(escape(p), body))
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont('NotoSC', 8); canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(17*mm, 10*mm, 'RR_GID_CN stage summary | read-only status report')
        canvas.drawRightString(193*mm, 10*mm, f'第 {doc.page} 页'); canvas.restoreState()
    SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=15*mm, bottomMargin=17*mm, title='RR_GID_CN 论文实验阶段总结报告').build(story, onFirstPage=footer, onLaterPages=footer)

HTML.write_text(make_html(), encoding='utf-8')
build_pdf()
print(HTML)
print(PDF)
