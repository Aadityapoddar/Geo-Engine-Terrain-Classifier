"""Layout-preserving Paper.pdf -> Paper.docx, with the [cite: N] artefacts removed.

pdf2docx reconstructs the two-column page; everything else here repairs the
places where it, or the citation removal, disturbs the original layout:

* pdf2docx drops whitespace-only spans, and this PDF stores every inter-word
  gap of a justified line as exactly such a span, so words ran together;
* the PDF's URW / Computer Modern faces are remapped to metric-compatible
  fonts Word actually has, otherwise line breaking drifts;
* super/subscripts are carried purely as a shifted baseline in the PDF, which
  pdf2docx discards, so "km2" came out looking like a subscript;
* [cite: N] markers are stripped at paragraph level, and again across
  paragraph boundaries, because a marker is often split by a line break;
* lines that pdf2docx positions by right-aligning them shift sideways once a
  marker is deleted, so their right indent is grown by the width it occupied.
"""
import re, sys, docx, pymupdf
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from pdf2docx import Converter
from pdf2docx.text.Spans import Spans
from pdf2docx.text.TextSpan import TextSpan
from pdf2docx.image.ImageSpan import ImageSpan

SRC, OUT = sys.argv[1], sys.argv[2]
EMU_PER_PT = 12700
TOP_MARGIN_FIX_PT = 6.4     # pdf2docx starts the text block this much too high

CITE = re.compile(r"\s*\[cite:\s*\d+(?:\s*,\s*\d+)*\]")
DANGLING = re.compile(r"\s*\[cite:\s*\d*")
CLOSER = re.compile(r"^\s*\d+(?:\s*,\s*\d+)*\]([.,;:])?\s*")
W_T = qn("w:t")


# ------------------------------------------------------------------ convert
def restore(self, raws):
    """As upstream, but keep whitespace-only spans: here they are the spaces."""
    for raw_span in raws:
        if 'image' in raw_span:
            span = ImageSpan(raw_span)
        else:
            span = TextSpan(raw_span)
            if not span.text and not span.style:
                span = None
        self.append(span)
    return self


Spans.restore = restore

cv = Converter(SRC)
cv.convert(OUT, multi_processing=False)
cv.close()

d = docx.Document(OUT)


# ------------------------------------------------------------- text helpers
def block_paragraphs(root):
    """Every paragraph in the document, in document order.

    Walks the XML rather than the Table API: pdf2docx emits merged and ragged
    grids that python-docx's row.cells silently skips over.
    """
    for el in root.iter(qn("w:p")):
        yield Paragraph(el, d)


def runs_text(p):
    """The paragraph's <w:t> nodes and their concatenation.

    Not the same as p.text, which also splices in tabs and breaks that carry
    no character of their own - offsets into it would not line up.
    """
    ts = [t for r in p.runs for t in r._r.findall(W_T)]
    return ts, "".join(t.text or "" for t in ts)


def node_at(ts, offset):
    """The <w:t> holding a given offset into the paragraph's concatenated text."""
    pos = 0
    for t in ts:
        n = len(t.text or "")
        if pos <= offset < pos + n:
            return t
        pos += n
    return ts[-1] if ts else None


def edit_text(p, spans):
    """Delete character ranges from a paragraph, keeping its run boundaries."""
    ts, full = runs_text(p)
    if not ts or not spans:
        return
    keep = [True] * len(full)
    for a, b in spans:
        for k in range(a, b):
            keep[k] = False
    pos = 0
    for t in ts:
        s = t.text or ""
        t.text = "".join(c for j, c in enumerate(s) if keep[pos + j])
        t.set(qn("xml:space"), "preserve")
        pos += len(s)


PARAS = list(block_paragraphs(d.element.body))

removed = 0
for p in PARAS:                                  # markers inside one paragraph
    spans = [(m.start(), m.end()) for m in CITE.finditer(runs_text(p)[1])]
    edit_text(p, spans)
    removed += len(spans)

opener = None                                    # markers split by a line break
for p in PARAS:
    ts, full = runs_text(p)
    if opener is not None:
        m = CLOSER.match(full)
        if m:
            edit_text(p, [(m.start(), m.end())])
            # the sentence's full stop sat after the marker; put it back on the
            # line the sentence actually ends on
            if m.group(1) and opener[0] is not None:
                opener[0].text = (opener[0].text or "").rstrip() + m.group(1)
            opener = None
            removed += 1
            continue
    m = DANGLING.search(full)                    # a complete one is already gone,
    if m:                                        # so anything left is an opener
        edit_text(p, [(m.start(), m.end())])
        opener = (node_at(ts, max(m.start() - 1, 0)),)

# A marker deleted from around a tab leaves the tab stranded, pushing the
# sentence's full stop across the column. Close the gap.
W_TAB = qn("w:tab")
for p in PARAS:
    parts = [n for r in p.runs for n in r._r if n.tag in (W_T, W_TAB)]
    for i, n in enumerate(parts):
        if n.tag != W_TAB or not (0 < i < len(parts) - 1):
            continue
        before = "".join(x.text or "" for x in parts[:i] if x.tag == W_T)
        after = "".join(x.text or "" for x in parts[i + 1:] if x.tag == W_T)
        if before[-1:].isalnum() and after[:1] in ".,;:":
            n.getparent().remove(n)

# ------------------------------------------------------------------- fonts
FONT_MAP = {"NimbusRomNo9L": "Times New Roman", "NimbusSanL": "Arial",
            "NimbusMonL": "Courier New"}
FONT_PREFIX = {"CMR": "Times New Roman", "CMBX": "Times New Roman",
               "CMTI": "Times New Roman", "CMMI": "Cambria Math",
               "CMSY": "Cambria Math", "CMEX": "Cambria Math",
               "CMTT": "Courier New", "CMSS": "Arial"}


def substitute(name):
    if name in FONT_MAP:
        return FONT_MAP[name]
    for pre, repl in FONT_PREFIX.items():
        if name.startswith(pre):
            return repl
    return None


fonts = 0
for rf in d.element.body.iter(qn("w:rFonts")):
    new = substitute(rf.get(qn("w:ascii")) or rf.get(qn("w:hAnsi")) or "")
    if new:
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(attr), new)
        fonts += 1


# ------------------------------------------------------- super/subscripts
def scan_scripts(path):
    """text -> 'super'/'sub', learnt from baseline shifts in the source PDF."""
    out = {}
    for page in pymupdf.open(path):
        for b in page.get_text("rawdict")["blocks"]:
            if b["type"]:
                continue
            for line in b["lines"]:
                spans = line["spans"]
                big = max(sp["size"] for sp in spans)
                for j, sp in enumerate(spans):
                    txt = "".join(c["c"] for c in sp["chars"]).strip()
                    if not txt or sp["size"] >= big * 0.8 or j == 0:
                        continue
                    dy = sp["origin"][1] - spans[j - 1]["origin"][1]
                    if abs(dy) >= 1:
                        out[txt] = "super" if dy < 0 else "sub"
    return out


SCRIPTS = scan_scripts(SRC)


def apply_scripts(p):
    sizes = [r.font.size for r in p.runs if r.font.size]
    if not sizes:
        return 0
    big, n = max(sizes), 0
    for r in p.runs:
        kind = SCRIPTS.get(r.text.strip())
        if kind and r.font.size and r.font.size < big * 0.8:
            r.font.superscript = kind == "super"
            r.font.subscript = kind == "sub"
            n += 1
    return n


# ------------------------------------------------- right-aligned fragments
def scan_cite_widths(path):
    """cite-stripped line text -> width in points its [cite: N] runs took up."""
    out = {}
    for page in pymupdf.open(path):
        for b in page.get_text("rawdict")["blocks"]:
            if b["type"]:
                continue
            for line in b["lines"]:
                chars = [c for sp in line["spans"] for c in sp["chars"]]
                txt = "".join(c["c"] for c in chars)
                if "[cite" not in txt:
                    continue
                width = 0.0
                for m in CITE.finditer(txt):
                    seg = chars[m.start():m.end()]
                    width += seg[-1]["bbox"][2] - seg[0]["bbox"][0]
                key = " ".join(CITE.sub("", txt).split())
                if key:
                    out.setdefault(key, width)
    return out


CITE_W = scan_cite_widths(SRC)


def repin(p):
    """A right-aligned line slides once its text shrinks; give back the width."""
    if p.alignment != WD_ALIGN_PARAGRAPH.RIGHT:
        return 0
    w = CITE_W.get(" ".join(p.text.split()))
    if not w:
        return 0
    p.paragraph_format.right_indent = int((p.paragraph_format.right_indent or 0)
                                          + w * EMU_PER_PT)
    return 1


scripts = pinned = 0
for p in PARAS:
    scripts += apply_scripts(p)
    pinned += repin(p)

# wrapped text in a table cell is left aligned in the paper; pdf2docx centres
# some of it because it only ever sees one line at a time
for tb in d.tables:
    for row in tb.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                if len(p.text) > 30:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    if p.runs and p.runs[0].text.startswith(" "):
                        p.runs[0].text = p.runs[0].text.lstrip()

# the underscore in this address has no glyph in the PDF's font
for p in PARAS:
    if "aniruddha" in p.text and "phd@" in p.text:
        for r in p.runs:
            if r.text.strip() == "aniruddha":
                r.text = r.text.replace("aniruddha", "aniruddha_")

# ------------------------------------------------------------- references
# pdf2docx separates the entries with line breaks. Deleting the markers freed
# enough room for its reflow to pull an entry up onto the previous line, so
# put the missing breaks back.
REF_START = re.compile(r"\[\d{1,2}\]\s")
W_BR = qn("w:br")


def break_before(p, offset):
    """Insert a line break at a text offset, unless one is already there."""
    parts = [n for r in p.runs for n in r._r if n.tag in (W_T, W_BR)]
    pos, node, local = 0, None, 0
    for i, n in enumerate(parts):
        if n.tag is W_BR or n.tag == W_BR:
            continue
        length = len(n.text or "")
        if pos <= offset < pos + length:
            node, local, idx = n, offset - pos, i
            break
        pos += length
    if node is None:
        return False
    if local == 0 and idx and parts[idx - 1].tag == W_BR:
        return False                              # already starts a line
    br = node.makeelement(W_BR, {})
    if local == 0:
        node.addprevious(br)
        return True
    text = node.text or ""
    node.text = text[:local]
    tail = node.makeelement(W_T, {qn("xml:space"): "preserve"})
    tail.text = text[local:]
    node.addnext(tail)
    node.addnext(br)
    return True


breaks = 0
for p in PARAS[next((i for i, q in enumerate(PARAS)
                     if q.text.strip() == "REFERENCES"), len(PARAS)) + 1:]:
    sizes = [r.font.size for r in p.runs if r.font.size]
    if not sizes or max(sizes) > 8 * EMU_PER_PT:
        continue                                  # 8pt is the reference size
    for m in reversed(list(REF_START.finditer(runs_text(p)[1]))):
        if m.start() > 0 and break_before(p, m.start()):
            breaks += 1

for sec in d.sections:
    sec.top_margin += int(TOP_MARGIN_FIX_PT * EMU_PER_PT)

d.save(OUT)
print(f"{OUT}: removed {removed} [cite: N] markers, remapped {fonts} fonts, "
      f"fixed {scripts} super/subscripts, re-pinned {pinned} lines, "
      f"restored {breaks} reference line breaks")
