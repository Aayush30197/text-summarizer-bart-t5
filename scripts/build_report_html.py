"""
Script to build an academic, printable report.html from report/report.md.
"""
import re
from pathlib import Path
from markdown_it import MarkdownIt

def build_academic_html():
    report_md_path = Path("report/report.md")
    output_html_path = Path("report/report.html")

    if not report_md_path.exists():
        raise FileNotFoundError(f"{report_md_path} not found.")

    with open(report_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Initialize markdown-it with table support and without linkify requirement
    md = MarkdownIt("commonmark", {"linkify": False}).enable("table")
    rendered_body = md.render(md_text)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Comparative Evaluation of BART and T5 for Abstractive Summarization | Final Capstone Report</title>
  <style>
    @page {{
      size: letter;
      margin: 0.8in;
      @bottom-right {{
        content: counter(page);
      }}
    }}

    :root {{
      --primary: #1e3a8a;
      --text: #1f2937;
      --text-muted: #4b5563;
      --bg: #ffffff;
      --border: #e5e7eb;
      --code-bg: #f3f4f6;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Charter", "Georgia", "Cambria", serif;
      color: var(--text);
      background-color: #f8fafc;
      line-height: 1.68;
      font-size: 15px;
      margin: 0;
      padding: 0;
    }}

    /* Floating Print / PDF Header */
    .action-bar {{
      position: sticky;
      top: 0;
      background: #1e293b;
      color: white;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      z-index: 100;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    .btn-print {{
      background: #38bdf8;
      color: #0f172a;
      border: none;
      font-weight: 700;
      font-size: 14px;
      padding: 8px 18px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s;
    }}
    .btn-print:hover {{ background: #7dd3fc; }}

    .paper-container {{
      max-width: 860px;
      margin: 36px auto;
      background: white;
      padding: 60px 70px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      border-radius: 8px;
      border: 1px solid var(--border);
    }}

    /* Typography */
    h1 {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 26px;
      font-weight: 800;
      color: #0f172a;
      margin-bottom: 8px;
      line-height: 1.25;
      text-align: center;
    }}

    .meta-box {{
      text-align: center;
      color: var(--text-muted);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 2px solid var(--border);
    }}

    h2 {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 20px;
      font-weight: 700;
      color: var(--primary);
      margin-top: 36px;
      margin-bottom: 14px;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 6px;
    }}

    h3 {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 16px;
      font-weight: 700;
      color: #1e293b;
      margin-top: 24px;
      margin-bottom: 10px;
    }}

    h4 {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 15px;
      font-weight: 600;
      color: #334155;
      margin-top: 18px;
      margin-bottom: 8px;
    }}

    p {{
      margin-bottom: 14px;
      text-align: justify;
    }}

    ul, ol {{
      margin: 10px 0 16px 24px;
    }}

    li {{
      margin-bottom: 6px;
    }}

    /* Blockquotes & Callouts */
    blockquote {{
      background: #f0fdf4;
      border-left: 4px solid #22c55e;
      padding: 12px 18px;
      margin: 16px 0;
      color: #14532d;
      font-size: 14px;
      border-radius: 0 6px 6px 0;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 13.5px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    th, td {{
      padding: 9px 12px;
      border: 1px solid #cbd5e1;
      text-align: left;
    }}

    th {{
      background: #f1f5f9;
      color: #0f172a;
      font-weight: 700;
    }}

    tr:nth-child(even) {{
      background: #f8fafc;
    }}

    /* Code */
    code {{
      font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
      background: var(--code-bg);
      padding: 2px 5px;
      border-radius: 4px;
      font-size: 13px;
      color: #0369a1;
    }}

    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 14px 18px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.5;
      margin: 16px 0;
    }}

    pre code {{
      background: transparent;
      color: inherit;
      padding: 0;
    }}

    hr {{
      border: none;
      height: 1px;
      background: var(--border);
      margin: 28px 0;
    }}

    /* Print Stylesheet for Canvas PDF Export */
    @media print {{
      body {{
        background: white;
        font-size: 13pt;
      }}
      .action-bar {{
        display: none !important;
      }}
      .paper-container {{
        box-shadow: none;
        border: none;
        padding: 0;
        margin: 0;
        max-width: 100%;
      }}
      h2 {{
        page-break-after: avoid;
      }}
      table, pre, blockquote {{
        page-break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>

  <div class="action-bar">
    <div>
      <strong>CSCI 390 Final Capstone Report</strong> — Fisk University
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <span style="font-size: 13px; color: #94a3b8;">Click button or press Cmd+P to Save as PDF</span>
      <button class="btn-print" onclick="window.print()">
        📄 Save as Canvas PDF (Print)
      </button>
    </div>
  </div>

  <div class="paper-container">
    {rendered_body}
  </div>

</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"[success] Generated academic printable report at {output_html_path}")

if __name__ == "__main__":
    build_academic_html()
