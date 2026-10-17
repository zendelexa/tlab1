"""Render the Markdown reports to PDF: Markdown (CommonMark + GFM tables) -> HTML -> headless Microsoft Edge.

usage: python tools/make_pdf.py            (from the repository root)
needs: markdown-it-py (pip install markdown-it-py) and Microsoft Edge (or Chrome: set BROWSER=path\\to\\chrome.exe)
"""
import html
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".omc" / "pylib"))          # local tool install (not part of the project env)
from markdown_it import MarkdownIt  # noqa: E402

REPORTS = [(ROOT / "task1_autofe" / "README.md", ROOT / "task1_autofe" / "report_task1.pdf"),
           (ROOT / "task2_rl" / "README.md", ROOT / "task2_rl" / "report_task2.pdf")]
BROWSER = os.environ.get("BROWSER", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: A4; margin: 14mm 12mm 14mm 12mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 18pt; margin: 0 0 8pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
h4 { font-size: 10pt; margin: 10pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 18pt; margin: 3pt 0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt; background: #f3f3f3; padding: 0 2pt;
       border-radius: 2pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 5pt; font-size: 7.6pt; white-space: pre-wrap;
      word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; }
td code, th code { font-size: 7.2pt; }
table { border-collapse: collapse; margin: 5pt 0; font-size: 7.6pt; width: 100%; }
th, td { border: 1px solid #c8c8c8; padding: 2pt 3pt; vertical-align: top; }
th { background: #eef1f5; }
tr { break-inside: avoid; }
img { max-width: 100%; break-inside: avoid; display: block; margin: 4pt auto; }
blockquote { margin: 4pt 0; padding: 2pt 8pt; border-left: 3px solid #bbb; color: #444; }
hr { border: none; border-top: 1px solid #ccc; }
a { color: #1a4fa0; text-decoration: none; }
"""


def render(md_path: Path, pdf_path: Path, tmp: Path):
    md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    body = md.render(md_path.read_text(encoding="utf-8"))
    title = md_path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    page = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f'<base href="{md_path.parent.as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>')
    page_path = tmp / (pdf_path.stem + ".html")
    page_path.write_text(page, encoding="utf-8")
    subprocess.run([BROWSER, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                    f"--user-data-dir={tmp / 'profile'}", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf_path}", page_path.as_uri()], check=True, timeout=300,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{md_path.relative_to(ROOT)} -> {pdf_path.relative_to(ROOT)} ({pdf_path.stat().st_size / 1e6:.1f} MB)")


def main():
    with tempfile.TemporaryDirectory() as d:
        for md_path, pdf_path in REPORTS:
            render(md_path, pdf_path, Path(d))


if __name__ == "__main__":
    main()
