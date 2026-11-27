#!/usr/bin/env python3
"""The manual as a PDF: README.md (the manual) set in the plug-in's own look, black and white, Inter and IBM Plex
Mono, hairline rules, the screenshots from docs/img. Writes docs/Neurosonde-Manual.pdf.

  python3 tools/manual/make_manual.py        (needs the `markdown` module and Google Chrome)
"""
import os, re, subprocess, sys, tempfile

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONTS = os.path.join(ROOT, "Resources", "fonts")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@font-face { font-family: Inter; src: url('file://%(f)s/Inter-Regular.ttf'); font-weight: 400; }
@font-face { font-family: Inter; src: url('file://%(f)s/Inter-Medium.ttf'); font-weight: 600; }
@font-face { font-family: Plex; src: url('file://%(f)s/IBMPlexMono-Light.ttf'); font-weight: 300; }
@font-face { font-family: Plex; src: url('file://%(f)s/IBMPlexMono-Medium.ttf'); font-weight: 500; }
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
body { font-family: Inter; font-size: 9.6pt; line-height: 1.45; color: #000; }
h1 { font-family: Inter; font-weight: 600; font-size: 34pt; letter-spacing: -0.5pt; margin: 0 0 4mm 0; }
h2 { font-family: Plex; font-weight: 500; font-size: 12pt; text-transform: lowercase; margin: 8mm 0 2mm 0;
     padding-top: 2mm; border-top: 0.5pt solid #000; }
h3 { font-family: Plex; font-weight: 500; font-size: 10pt; margin: 5mm 0 1.5mm 0; }
code, pre { font-family: Plex; font-weight: 300; font-size: 8.4pt; }
pre { border-left: 0.5pt solid #000; padding: 1mm 0 1mm 3mm; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%%; margin: 2mm 0 3mm 0; font-size: 8.6pt; }
th { font-family: Plex; font-weight: 500; text-align: left; border-bottom: 0.5pt solid #000; padding: 1mm 2mm 1mm 0; }
td { border-bottom: 0.25pt solid #999; padding: 1mm 2mm 1mm 0; vertical-align: top; }
img { max-width: 100%%; border: 0.5pt solid #000; margin: 2mm 0; }
ul { padding-left: 4.5mm; } li { margin: 0.6mm 0; }
a { color: #000; }
.cover { font-family: Plex; font-weight: 300; font-size: 9pt; margin-bottom: 6mm; }
"""


def github_lists(text):
    """GitHub's list rules for Python-Markdown: nesting by 2 spaces becomes 4, and a list needs a blank line before it"""
    out, prev = [], ""
    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        lead = len(line) - len(stripped)
        if lead and stripped:
            line = " " * (lead * 2) + stripped
        is_item = re.match(r"^\s*([-*]|\d+\.) ", line) is not None
        prev_item = re.match(r"^\s*([-*]|\d+\.) ", prev) is not None
        if is_item and prev.strip() and not prev_item:
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def main():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    version = re.search(r'project\(Neurosonde VERSION ([0-9.]+)', open(os.path.join(ROOT, "CMakeLists.txt")).read()).group(1)
    body = markdown.markdown(github_lists(text), extensions=["tables", "fenced_code"])
    body = body.replace("<h1>neurosonde</h1>", "<h1>neurosonde</h1><div class='cover'>manual &nbsp;/&nbsp; version %s</div>" % version, 1)
    # image paths relative to the repository
    body = re.sub(r'src="(docs/[^"]+)"', lambda m: 'src="file://%s"' % os.path.join(ROOT, m.group(1)), body)
    html = "<html><head><meta charset='utf-8'><style>%s</style></head><body>%s</body></html>" % (CSS % {"f": FONTS}, body)
    out = os.path.join(ROOT, "docs", "Neurosonde-Manual.pdf")
    if os.path.exists(out):
        os.remove(out)
    with tempfile.TemporaryDirectory() as tmp:
        h = os.path.join(tmp, "manual.html")
        open(h, "w", encoding="utf-8").write(html)
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                        "--user-data-dir=" + os.path.join(tmp, "chrome"), "--print-to-pdf=" + out, "file://" + h],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # (Chrome often crashes on exit, after writing)
        if not os.path.exists(out):
            sys.exit("no PDF written")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
