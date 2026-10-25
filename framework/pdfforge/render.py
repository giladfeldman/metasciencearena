"""Paint a `Paper` into a PDF with reportlab, recording where everything landed.

Returns (pdf_bytes, placement): `placement["fragments"]` lists every painted block
fragment as {page, block_id, kind, text}, in reading order, so page-level gold is
known from the painter, never re-read from the PDF. `placement["furniture"]` lists the
exact strings painted outside the body (headers, footers, stamps, watermarks, line
numbers), which is the text gold's must_exclude.

Layout knobs live in `paper.layout` (set by the house style and by layout defects):
  columns (1|2), gutter (pt), page ("letter"|"a4"), margin (pt), body_size, leading,
  header_left / header_right / footer_text / page_numbers / stamp / side_stamp /
  watermark / line_numbers / first_page_footnote (list of strings), char_space_labels.
"""
from __future__ import annotations

import html
import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, FrameBreak,
                                KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table as RLTable, TableStyle)

from .model import Block, Paper, Run, Table

FONT_DIR = Path(__file__).resolve().parent / "fonts"
BODY, BOLD, ITAL, MATH = "PF-Body", "PF-Bold", "PF-Italic", "PF-Math"
_REGISTERED: set[str] = set()
_BODY_CMAP: set[int] = set()


def register_fonts() -> None:
    if BODY in _REGISTERED:
        return
    from fontTools.ttLib import TTFont as FT
    from reportlab.lib.fonts import addMapping
    for name, f in [(BODY, "STIXTwoText-Regular.ttf"), (BOLD, "STIXTwoText-Bold.ttf"),
                    (ITAL, "STIXTwoText-Italic.ttf"), (MATH, "STIXTwoMath-Regular.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f)))
        _REGISTERED.add(name)
    addMapping(BODY, 0, 0, BODY)
    addMapping(BODY, 1, 0, BOLD)
    addMapping(BODY, 0, 1, ITAL)
    addMapping(BODY, 1, 1, BOLD)
    _BODY_CMAP.update(FT(str(FONT_DIR / "STIXTwoText-Regular.ttf")).getBestCmap().keys())


def register_alias(alias: str, base_file: str = "STIXTwoText-Regular.ttf") -> str:
    """A second registration of a face under a new name, so a post-processor can rewrite
    THAT font's character map and leave every other glyph alone (catalog F)."""
    if alias not in _REGISTERED:
        pdfmetrics.registerFont(TTFont(alias, str(_renamed_copy(alias, base_file))))
        _REGISTERED.add(alias)
    return alias


def alias_ps_name(alias: str) -> str:
    """The PostScript name an alias embeds under (what a post-processor searches for)."""
    return "PFAlias" + "".join(ch for ch in alias if ch.isalnum())


def _renamed_copy(alias: str, base_file: str) -> Path:
    """reportlab merges two registrations of one file into one embedded font, so an alias
    needs its own file: the committed face with a new PostScript/family name, written to
    a cache dir (regenerated, never committed)."""
    import tempfile

    from fontTools.ttLib import TTFont as FT
    cache = Path(tempfile.gettempdir()) / "pdfforge_font_aliases"
    cache.mkdir(exist_ok=True)
    out = cache / f"{alias_ps_name(alias)}__{base_file}"
    if not out.exists():
        f = FT(str(FONT_DIR / base_file))
        ps = alias_ps_name(alias)
        for rec in f["name"].names:
            if rec.nameID in (1, 3, 4, 6, 16):
                rec.string = ps
        f.save(str(out))
    return out


# ------------------------------------------------------------------ run -> markup
def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _wrap_missing(text: str) -> str:
    """Characters the body face lacks go to the math face (STIX Two Math)."""
    out, cur, in_math = [], [], False
    for ch in text:
        miss = ord(ch) not in _BODY_CMAP and ch not in "\n"
        if miss != in_math and cur:
            seg = _escape("".join(cur))
            out.append(f'<font name="{MATH}">{seg}</font>' if in_math else seg)
            cur = []
        in_math = miss
        cur.append(ch)
    if cur:
        seg = _escape("".join(cur))
        out.append(f'<font name="{MATH}">{seg}</font>' if in_math else seg)
    return "".join(out)


def runs_markup(runs: list[Run]) -> str:
    parts = []
    for r in runs:
        if r.image:
            parts.append(f'<img src="{r.image}" width="6" height="7" valign="middle"/>')
            continue
        if r.text == "\n":
            parts.append("<br/>")
            continue
        s = _wrap_missing(r.text)
        if r.font:
            s = f'<font name="{r.font}">{_escape(r.text)}</font>'
        if r.style in ("i", "bi"):
            s = f"<i>{s}</i>"
        if r.style in ("b", "bi"):
            s = f"<b>{s}</b>"
        if r.style == "sup":
            s = f"<super>{s}</super>"
        if r.style == "sub":
            s = f"<sub>{s}</sub>"
        parts.append(s)
    return "".join(parts)


class PFParagraph(Paragraph):
    """A Paragraph that carries its block id through splits (page gold per fragment)."""
    _pf_block: str | None = None
    _pf_kind: str | None = None

    def split(self, aW, aH):
        parts = super().split(aW, aH)
        for p in parts:
            if isinstance(p, Paragraph):
                p.__class__ = PFParagraph
                p._pf_block, p._pf_kind = self._pf_block, self._pf_kind
        return parts


class Rotated(Flowable):
    """A flowable drawn rotated by 90 degrees (landscape table on a portrait page, L22)."""

    def __init__(self, inner: Flowable):
        super().__init__()
        self.inner = inner

    def wrap(self, aW, aH):
        w, h = self.inner.wrap(aH, aW)
        self._iw, self._ih = w, h
        return h, w

    def draw(self):
        c = self.canv
        c.saveState()
        c.rotate(90)
        self.inner.drawOn(c, 0, -self._ih)
        c.restoreState()


class Chart(Flowable):
    """A small bar chart: bars plus y-axis tick labels painted as text (catalog T19)."""

    def __init__(self, ticks, width):
        super().__init__()
        self.ticks, self.w, self.h = ticks, min(width, 220), 90

    def wrap(self, aW, aH):
        return self.w, self.h + 6

    def draw(self):
        c = self.canv
        c.setLineWidth(0.6)
        c.line(18, 6, 18, self.h)
        c.line(18, 6, self.w, 6)
        c.setFont(BODY, 6.5)
        for k, t in enumerate(self.ticks):
            y = 6 + k * (self.h - 10) / max(1, len(self.ticks) - 1)
            c.drawRightString(14, y - 2, t)
        for k, hgt in enumerate((0.45, 0.7, 0.55, 0.85)):
            c.setFillGray(0.55 if k % 2 else 0.25)
            c.rect(30 + k * 40, 6, 24, (self.h - 12) * hgt, fill=1, stroke=0)
        c.setFillGray(0)


# ------------------------------------------------------------------- document
class _Doc(BaseDocTemplate):
    def __init__(self, buf, paper: Paper, placement: dict, **kw):
        super().__init__(buf, **kw)
        self.paper = paper
        self.placement = placement
        self._notes_on_page: dict[int, list] = {}

    def afterFlowable(self, fl):
        bid = getattr(fl, "_pf_block", None)
        if bid is None:
            return
        page = self.page
        text = fl.getPlainText() if isinstance(fl, Paragraph) else getattr(fl, "_pf_text", "")
        self.placement["fragments"].append({"page": page, "block_id": bid,
                                            "kind": getattr(fl, "_pf_kind", None), "text": text})
        for nid in getattr(self.paper, "_note_blocks", {}).get(bid, []):
            if nid not in self.placement["note_pages"]:
                self.placement["note_pages"][nid] = page
                self._notes_on_page.setdefault(page, []).append(nid)


def _styles(paper: Paper) -> dict:
    L = paper.layout
    size = L.get("body_size", 9.5 if L.get("columns", 1) == 2 else 10.5)
    lead = L.get("leading", size * 1.25)
    base = ParagraphStyle("body", fontName=BODY, fontSize=size, leading=lead,
                          alignment=TA_JUSTIFY if L.get("justify", True) else TA_LEFT,
                          spaceAfter=0 if L.get("no_para_space") else lead * 0.5,
                          firstLineIndent=0 if L.get("no_indent") else 12)
    return {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontName=BOLD, fontSize=16, leading=19,
                                alignment=TA_LEFT, firstLineIndent=0, spaceAfter=8),
        "authors": ParagraphStyle("authors", parent=base, fontSize=10.5, alignment=TA_LEFT,
                                  firstLineIndent=0, spaceAfter=2),
        "affil": ParagraphStyle("affil", parent=base, fontName=ITAL, fontSize=8.5, leading=10,
                                alignment=TA_LEFT, firstLineIndent=0, spaceAfter=1),
        "abstract": ParagraphStyle("abstract", parent=base, fontSize=size - 0.5, leading=lead - 0.6,
                                   firstLineIndent=0, leftIndent=0, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=base, fontName=BOLD, fontSize=size + 1.5, leading=lead + 3,
                             firstLineIndent=0, spaceBefore=lead * 0.6, spaceAfter=4, alignment=TA_LEFT),
        "h2": ParagraphStyle("h2", parent=base, fontName=BOLD, fontSize=size + 0.5, leading=lead + 1,
                             firstLineIndent=0, spaceBefore=lead * 0.4, spaceAfter=3, alignment=TA_LEFT),
        "h3": ParagraphStyle("h3", parent=base, fontName=ITAL, fontSize=size, leading=lead,
                             firstLineIndent=0, spaceBefore=lead * 0.3, spaceAfter=2, alignment=TA_LEFT),
        "caption": ParagraphStyle("caption", parent=base, fontSize=size - 1, leading=lead - 1.5,
                                  firstLineIndent=0, alignment=TA_LEFT, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=base, fontSize=size - 1.5, leading=lead - 2.5,
                               firstLineIndent=0, alignment=TA_LEFT, spaceAfter=0),
        "ref": ParagraphStyle("ref", parent=base, fontSize=size - 1, leading=lead - 1.5,
                              firstLineIndent=-12 if L.get("hanging_refs", True) else 0,
                              leftIndent=12 if L.get("hanging_refs", True) else 0,
                              alignment=TA_LEFT, spaceAfter=2),
        "note": ParagraphStyle("note", parent=base, fontSize=7.5, leading=9, firstLineIndent=0,
                               alignment=TA_LEFT),
    }


def _para(runs, style, bid, kind) -> PFParagraph:
    p = PFParagraph(runs_markup(runs), style)
    p._pf_block, p._pf_kind = bid, kind
    return p


def _table_flowable(t: Table, st: dict, avail_w: float, tight: bool = False):
    rows = t.header + t.body
    data = [[_para(c.runs, st["cell"], None, None) for c in row] for row in rows]
    # spans: reportlab needs the grid padded; expand a row with fewer cells than columns
    ncol = max(sum(c.colspan for c in row) for row in rows)
    grid, cmds = [], []
    for ri, row in enumerate(rows):
        out, ci = [], 0
        for c, p in zip(row, data[ri]):
            out.append(p)
            if c.colspan > 1 or c.rowspan > 1:
                cmds.append(("SPAN", (ci, ri), (ci + c.colspan - 1, ri + c.rowspan - 1)))
            out += [""] * (c.colspan - 1)
            ci += c.colspan
        out += [""] * (ncol - len(out))
        grid.append(out)
    rel = t.col_widths or [1.0] * ncol
    widths = [avail_w * w / sum(rel) for w in rel]
    tbl = RLTable(grid, colWidths=widths, repeatRows=len(t.header))
    h = len(t.header)
    pad = 0 if tight else 2
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("LEFTPADDING", (0, 0), (-1, -1), pad),
             ("RIGHTPADDING", (0, 0), (-1, -1), pad)] + cmds
    if t.ruled:
        style += [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                  ("LINEBELOW", (0, h - 1), (-1, h - 1), 0.5, colors.black),
                  ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black)]
    tbl.setStyle(TableStyle(style))
    return tbl


def _caption_runs(t: Table) -> list[Run]:
    if t.caption_style == "untitled":
        return [Run("")]
    if t.caption_style == "pipe":
        return [Run(f"{t.label} | ", style="b")] + t.caption
    if t.caption_style == "label_alone":
        return [Run(t.label, style="b")]
    return [Run(t.label, style="b"), Run(". ")] + t.caption


def build_story(paper: Paper, st: dict, frame_w: float) -> list:
    story: list = []
    two_col = paper.layout.get("columns", 1) == 2
    story.append(_para([Run(paper.title)], st["title"], "title", "title"))
    story.append(_para([Run(", ".join(f"{g} {s}" for s, g in paper.authors))], st["authors"], "authors", "authors"))
    for a in paper.affiliations:
        story.append(_para([Run(a)], st["affil"], "affil", "affil"))
    lie = paper.front_matter.get("masthead_font_alias", {})
    for line in paper.front_matter.get("masthead", []):
        runs, buf = [], ""
        for ch in line:
            if ch in lie:
                if buf:
                    runs.append(Run(buf))
                    buf = ""
                runs.append(Run(ch, font=lie[ch]))
            else:
                buf += ch
        if buf:
            runs.append(Run(buf))
        story.append(_para(runs, st["affil"], "masthead", "masthead"))
    story.append(Spacer(1, 6))
    story.append(_para([Run("Abstract", style="b")], st["h2"], "abstract_heading", "heading"))
    story.append(_para(paper.abstract, st["abstract"], "abstract", "abstract"))
    story.append(_para([Run("Keywords: ", style="i"), Run("; ".join(paper.keywords))], st["abstract"], "keywords", "keywords"))
    if two_col:
        story.append(FrameBreak())
        story.append(NextPageTemplate("cols"))
    if paper.layout.get("numbered_headings"):
        _auto_number(paper)
    pending_run_in = None
    for b in paper.blocks:
        if b.layout.get("painted_inside"):
            continue  # a float a defect placed inside another block's paragraph
        if b.kind == "heading" and b.layout.get("run_in"):
            pending_run_in = b
            continue
        if pending_run_in is not None and b.kind == "para":
            # run-in heading: "Participants. A total of ..." in one paragraph (catalog H02)
            head = [Run(r.text, style="b") for r in pending_run_in.runs] + [Run(". ")]
            story.append(_para(head + b.runs, st["body"], b.block_id, "para"))
            pending_run_in = None
            continue
        story += _block_flowables(b, st, frame_w)
    for head, text in paper.front_matter.get("back_matter", []):
        story.append(_para([Run(head, style="b")], st["h2"], f"back:{head}", "heading"))
        story.append(_para([Run(text)], st["body"], f"backtext:{head}", "nonbody"))
    story += _references_story(paper, st, frame_w)
    for k, bio in enumerate(paper.front_matter.get("bios", [])):
        if k == 0:
            story.append(_para([Run("About the authors", style="b")], st["h2"], "bios_heading", "heading"))
        story.append(_para([Run(bio)], st["body"], f"bio:{k}", "nonbody"))
    if paper.front_matter.get("appendix_refs"):
        story.append(_para([Run("Appendix References", style="b")], st["h2"], "apprefs_heading", "heading"))
        for k, t in enumerate(paper.front_matter["appendix_refs"]):
            story.append(_para([Run(t)], st["ref"], f"appref:{k}", "nonbody"))
    return story


def _references_story(paper: Paper, st: dict, frame_w: float) -> list:
    L = paper.layout
    out: list = []
    if L.get("refs_layout") == "two_col_under_table":
        out += [NextPageTemplate("refs2"), PageBreak()]
        ct = paper.front_matter.get("credit_table")
        if ct is not None:
            tf = _table_flowable(ct, st, 480.0)
            tf._pf_block, tf._pf_kind = "credit:table", "nonbody"
            tf._pf_text = " ".join(c.text for row in ct.header + ct.body for c in row)
            out += [_para([Run("CRediT author statement", style="b")], st["h2"], "credit:caption", "nonbody"), tf]
        out.append(FrameBreak())
    else:
        out.append(CondPageBreak(60))
    heading = L.get("refs_heading", "References")
    if heading:
        out.append(_para([Run(heading, style="b")], st["h1"], "refs_heading", "heading"))
    refs = paper.references
    reflow = L.get("refs_reflow")
    if reflow == "one_paragraph":
        runs = []
        for r in refs:
            runs += (r.printed_runs or [Run(r.raw_text)]) + [Run(" ")]
        out.append(_para(runs, st["ref"], "ref:all", "reference"))
    elif reflow == "pairs":
        for i in range(0, len(refs), 2):
            runs = []
            for r in refs[i:i + 2]:
                runs += (r.printed_runs or [Run(r.raw_text)]) + [Run(" ")]
            out.append(_para(runs, st["ref"], f"ref:pair{i}", "reference"))
    else:
        for r in refs:
            out.append(_para(r.printed_runs or [Run(r.raw_text)], st["ref"], f"ref:{r.ref_id}", "reference"))
    return out


def _auto_number(paper: Paper) -> None:
    """Decimal heading numbers (1., 1.1., 1.1.1.) for house styles that number sections."""
    ctr = [0, 0, 0]
    for b in paper.blocks:
        if b.kind != "heading":
            continue
        lv = min(b.level, 3) - 1
        ctr[lv] += 1
        for k in range(lv + 1, 3):
            ctr[k] = 0
        b.layout["auto_number"] = ".".join(str(x) for x in ctr[:lv + 1]) + "."


def _block_flowables(b: Block, st: dict, frame_w: float) -> list:
    out: list = []
    if b.layout.get("break_before"):
        out.append(PageBreak())
    if b.kind == "heading" and b.layout.get("run_in"):
        return []  # painted inline at the start of the next paragraph (see build_story)
    if b.kind == "heading":
        num = f"{b.number} " if b.number else ""
        if not num and b.layout.get("auto_number"):
            num = b.layout["auto_number"] + " "
        runs = ([Run(num, style="b")] if num else []) + b.runs
        p = _para(runs, st[f"h{min(b.level, 3)}"], b.block_id, "heading")
        p.keepWithNext = True
        out.append(p)
    elif b.kind in ("para", "list_item", "box"):
        style = st["body"]
        kind = b.kind if b.in_body else "nonbody"
        if b.kind == "box":
            style = ParagraphStyle("box", parent=st["caption"], borderWidth=0.5, borderColor=colors.black,
                                   borderPadding=4, backColor=colors.Color(0.94, 0.94, 0.94), spaceBefore=6, spaceAfter=8)
        if b.kind == "list_item":
            style = ParagraphStyle("li", parent=st["body"], leftIndent=12, firstLineIndent=-8)
        splits = sorted(b.layout.get("splits", []), key=lambda x: x["at"])
        if not splits:
            out.append(_para(b.runs, style, b.block_id, kind))
        else:
            # One logical paragraph painted in pieces: a column/page break or an inserted
            # float sits INSIDE it (catalog L04, L10, L11, L13, L16, L24). The gold keeps
            # it as one paragraph; only the painter splits it.
            cont = ParagraphStyle("cont", parent=style, firstLineIndent=0)
            prev = 0
            for k, sp in enumerate(splits):
                seg = b.runs[prev:sp["at"]]
                if seg:
                    out.append(_para(seg, style if k == 0 else cont, b.block_id, kind))
                if sp["kind"] == "frame":
                    out.append(FrameBreak())
                elif sp["kind"] == "page":
                    out.append(PageBreak())
                elif sp["kind"] == "para":
                    pass  # a new paragraph starts here (the cut itself is the defect)
                elif sp["kind"] == "insert" and sp.get("block") is not None:
                    out += _block_flowables(sp["block"], st, frame_w)
                prev = sp["at"]
            tail = b.runs[prev:]
            if tail:
                out.append(_para(tail, cont, b.block_id, kind))
    elif b.kind == "table" and b.table is not None:
        t = b.table
        cap = _para(_caption_runs(t), st["caption"], f"{b.block_id}:caption", "caption")
        rot_w = 540.0  # a landscape table spans the page HEIGHT minus margins and reserves
        tw = frame_w * b.layout.get("narrow", 1.0)
        if b.layout.get("tight"):
            tw = frame_w * 0.62
        tf = _table_flowable(t, st, tw if not t.rotate else rot_w, tight=bool(b.layout.get("tight")))
        if b.layout.get("side_by_side") is not None:
            t2 = b.layout["side_by_side"]
            half = frame_w / 2 - 6
            cap2 = _para(_caption_runs(t2), st["caption"], f"{b.block_id}:caption2", "caption")
            tf2 = _table_flowable(t2, st, half)
            tf2._pf_block, tf2._pf_kind = f"{b.block_id}:table2", "table"
            tf2._pf_text = " ".join(c.text for row in t2.header + t2.body for c in row)
            left = _table_flowable(t, st, half)
            # both captions on ONE line, as in the real failure (labels side by side)
            caps = RLTable([[_para(_caption_runs(t), st["caption"], f"{b.block_id}:caption", "caption"), cap2],
                            [left, tf2]], colWidths=[half + 6, half + 6])
            caps._pf_block, caps._pf_kind = f"{b.block_id}:table", "table"
            caps._pf_text = " ".join(c.text for tt in (t, t2) for row in tt.header + tt.body for c in row)
            out.append(caps)
            return out
        tf._pf_block, tf._pf_kind = f"{b.block_id}:table", "table"
        tf._pf_text = " ".join(c.text for row in t.header + t.body for c in row)
        note = _para(t.note, st["caption"], f"{b.block_id}:note", "table_note") if t.note else None
        if t.rotate:
            inner = RLTable([[cap], [tf]] + ([[note]] if note else []), colWidths=[rot_w])
            out += [PageBreak(), Rotated(inner), PageBreak()]
        else:
            out.append(KeepTogether([cap, tf] + ([note] if note else [])))
    elif b.kind == "figure" and b.figure is not None:
        f = b.figure
        out.append(Chart(f.ticks, frame_w))
        out.append(_para([Run(f.label + ". ", style="b")] + f.caption, st["caption"], f"{b.block_id}:caption", "caption"))
    return out


def render(paper: Paper) -> tuple[bytes, dict]:
    register_fonts()
    L = paper.layout
    pagesize = A4 if L.get("page") == "a4" else letter
    W, H = pagesize
    m = L.get("margin", 54)
    cols = L.get("columns", 1)
    gutter = L.get("gutter", 18)
    foot_reserve = L.get("foot_reserve", 48)
    head_reserve = 18
    placement = {"fragments": [], "furniture": [], "note_pages": {}, "pages": 0}
    # map block -> footnote ids (markers live in runs)
    paper._note_blocks = {b.block_id: [r.note_id for r in b.runs if r.note_id] for b in paper.blocks}  # type: ignore[attr-defined]
    buf = io.BytesIO()
    doc = _Doc(buf, paper, placement, pagesize=pagesize, leftMargin=m, rightMargin=m,
               topMargin=m, bottomMargin=m, title=paper.title, invariant=1,
               author="", subject="", creator="pdfforge", producer="pdfforge")
    body_w = W - 2 * m
    col_w = (body_w - gutter * (cols - 1)) / cols
    bottom = m + foot_reserve
    top = H - m - head_reserve
    if cols == 1:
        frames_first = [Frame(m, bottom, body_w, top - bottom, id="f0", leftPadding=0, rightPadding=0,
                              topPadding=0, bottomPadding=0)]
        frames = frames_first
    else:
        title_h = L.get("title_block_h") or _title_block_height(paper, _styles(paper), body_w)
        frames_first = [Frame(m, top - title_h, body_w, title_h, id="title", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
        frames_first += [Frame(m + i * (col_w + gutter), bottom, col_w, top - title_h - bottom - 6, id=f"c{i}",
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0) for i in range(cols)]
        frames = [Frame(m + i * (col_w + gutter), bottom, col_w, top - bottom, id=f"k{i}",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0) for i in range(cols)]
    st = _styles(paper)

    def on_page(c, d):
        _paint_furniture(c, d, paper, placement, W, H, m)

    def on_page_end(c, d):
        _paint_footnotes(c, d, paper, placement, W, m, bottom)

    band = 170
    half = (body_w - 14) / 2
    refs2 = [Frame(m, top - band, body_w, band, id="rband", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
    refs2 += [Frame(m + i * (half + 14), bottom, half, top - band - bottom - 6, id=f"r{i}",
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0) for i in range(2)]
    refs2_next = [Frame(m + i * (half + 14), bottom, half, top - bottom, id=f"rn{i}",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0) for i in range(2)]
    doc.addPageTemplates([
        PageTemplate(id="first", frames=frames_first, onPage=on_page, onPageEnd=on_page_end,
                     autoNextPageTemplate="cols"),
        PageTemplate(id="cols", frames=frames, onPage=on_page, onPageEnd=on_page_end),
        PageTemplate(id="refs2", frames=refs2, onPage=on_page, onPageEnd=on_page_end, autoNextPageTemplate="refs2n"),
        PageTemplate(id="refs2n", frames=refs2_next, onPage=on_page, onPageEnd=on_page_end),
    ])
    story = build_story(paper, st, col_w)
    doc.build(story)
    placement["pages"] = doc.page
    return repair_tounicode_nonbmp(buf.getvalue()), placement


def repair_tounicode_nonbmp(pdf_bytes: bytes) -> bytes:
    """reportlab writes a non-BMP character (e.g. U+1D45D) into ToUnicode as 5 hex digits,
    which is invalid; real PDFs use a UTF-16 surrogate pair. Fix it on every PDF so a
    tooling artefact is never mistaken for a defect under test."""
    import re as _re

    import pikepdf
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    changed = False
    seen = set()
    for page in pdf.pages:
        res = page.get("/Resources")
        if res is None or "/Font" not in res:
            continue
        for _, f in res.Font.items():
            if f.objgen in seen or "/ToUnicode" not in f:
                continue
            seen.add(f.objgen)
            cm = f.ToUnicode.read_bytes().decode("latin-1")

            def fix(m):
                cp = int(m.group(2), 16)
                return f"<{m.group(1)}> <{chr(cp).encode('utf-16-be').hex().upper()}>"
            new = _re.sub(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]{5,6})>", fix, cm)
            if new != cm:
                f.ToUnicode = pdf.make_stream(new.encode("latin-1"))
                changed = True
    if not changed:
        return pdf_bytes
    out = io.BytesIO()
    pdf.save(out, deterministic_id=True)
    return out.getvalue()


def _title_block_height(paper: Paper, st: dict, width: float) -> float:
    """Measured height of the full-width title block, so two-column text starts right under it."""
    h = 0.0
    for fl in build_story(paper, st, width)[:40]:
        if isinstance(fl, type(FrameBreak)):  # FrameBreak is an instance in reportlab
            break
        if isinstance(fl, (Paragraph, Spacer)):
            _, fh = fl.wrap(width, 10_000)
            h += fh + getattr(getattr(fl, "style", None), "spaceAfter", 0) + getattr(getattr(fl, "style", None), "spaceBefore", 0)
    return h + 14


def _paint_furniture(c, doc, paper: Paper, placement: dict, W, H, m) -> None:
    L = paper.layout
    pg = doc.page
    c.saveState()
    painted = []

    def text(x, y, s, size=7.5, font=BODY, align="left", rot=0, gray=0.0):
        c.setFont(font, size)
        c.setFillGray(gray)
        if rot:
            c.saveState()
            c.translate(x, y)
            c.rotate(rot)
            c.drawString(0, 0, s)
            c.restoreState()
        elif align == "right":
            c.drawRightString(x, y, s)
        elif align == "center":
            c.drawCentredString(x, y, s)
        else:
            c.drawString(x, y, s)
        painted.append(s)

    if pg > 1 or L.get("header_on_first"):
        if L.get("header_left"):
            text(m, H - m + 4, L["header_left"] if pg % 2 == 0 else L.get("header_right", L["header_left"]))
    if L.get("footer_text"):
        text(m, m - 14, L["footer_text"].format(page=pg))
    if L.get("page_numbers"):
        fmt = L["page_numbers"]
        text(W - m, m - 14, fmt.format(page=pg + L.get("first_page_number", 1) - 1), align="right")
    if L.get("stamp"):
        text(W / 2, 14, L["stamp"], size=6.5, align="center", gray=0.35)
    if L.get("side_stamp"):
        text(W - 14, 80, L["side_stamp"], size=6.5, rot=90, gray=0.35)
    if L.get("watermark"):
        c.saveState()
        c.setFillColor(colors.Color(0.6, 0.6, 0.6, alpha=0.25))
        c.setFont(BOLD, 54)
        c.translate(W / 2, H / 2)
        c.rotate(45)
        c.drawCentredString(0, 0, L["watermark"])
        c.restoreState()
        painted.append(L["watermark"])
    if L.get("line_numbers"):
        y = H - m - 18
        n = (pg - 1) * 60
        while y > m + 40:
            n += 1
            text(m - 22, y, str(n), size=6, align="left", gray=0.3)
            y -= paper.layout.get("leading", 12.5)
    if pg == 1 and paper.front_matter.get("spaced_label"):
        t = c.beginText(m, H - m + 4)
        t.setFont(BOLD, 7.5)
        t.setCharSpace(2.2)
        t.textOut(paper.front_matter["spaced_label"].replace(" ", ""))
        c.drawText(t)
        painted.append(paper.front_matter["spaced_label"].replace(" ", ""))
    if pg == 1 and L.get("crossmark"):
        c.setFont(BOLD, 7)
        c.drawString(W - m - 6, H - m - 22, "T")
        painted.append("T")
    if pg == 1:
        for k, s in enumerate(L.get("first_page_footnote", [])):
            text(m, m + 30 - 9 * k, s, size=7)
    c.restoreState()
    placement["furniture"].append({"page": pg, "strings": painted})


def _paint_footnotes(c, doc, paper: Paper, placement: dict, W, m, bottom) -> None:
    ids = doc._notes_on_page.get(doc.page, [])
    if not ids:
        return
    notes = {f.note_id: f for f in paper.footnotes}
    st = ParagraphStyle("fn", fontName=BODY, fontSize=7.5, leading=9)
    y = bottom - 4
    c.saveState()
    c.setLineWidth(0.4)
    c.line(m, y, m + 80, y)
    for nid in ids:
        f = notes.get(nid)
        if not f:
            continue
        p = PFParagraph(f"<super>{_escape(f.marker)}</super> " + runs_markup(f.runs), st)
        w, h = p.wrap(W - 2 * m, 100)
        y -= h + 2
        p.drawOn(c, m, y)
        placement["fragments"].append({"page": doc.page, "block_id": f"note:{nid}", "kind": "footnote",
                                       "text": p.getPlainText()})
    c.restoreState()
