"""Markdown / HTML -> the three PDF arenas' output shapes.

Marker and MinerU answer in markdown (Marker's table player reads Marker's JSON,
whose Table blocks carry HTML). This module turns that into what each arena
scores, and is deliberately the ONLY place that does, so a rule about markup is
written once for every markdown-emitting player.

TEXT (pdf-text-fidelity-v1): `md_to_plain`. Ported from ESCImate's
tests/harness/extractor_pilot/mdplain.py (2026-10-07), including its self-test
cases, which live in players/adapters/tests/test_markdown_projection.py. The arena
scores plain text, so `*p* < .05`, `$\\eta^2_p$` and `η<sup>2</sup>` must reach the
scorer as `p < .05`, `η²p`, `η²` -- or a tool is charged for markup rather than
content. It removes markup and maps a CLOSED list of LaTeX symbols; it never
rewrites a digit, sign or relational operator that is already text.

THE TRAP THIS EXISTS TO AVOID. A generic `<[^>]+>` tag stripper deletes
"p < .001, η<sup>" -- from the statistic's own "<" to the next tag's ">". Only a
"<" followed by a LETTER (or "/" and a letter) starts a tag here. Pinned by test.

NOTHING IS REMOVED FOR SCOPE. Unlike the GROBID adapter (TEI <p> only) or the
Docling projection (drops Docling-labelled furniture and references), the
markdown players' text is the WHOLE document the tool emitted, de-marked. The
tools draw no such boundary themselves, and drawing one here would credit them
with arena code. The v2 text scorer charges only `must_exclude` material
(bibliography, masthead, DOIs...), so a tool that emits its references pays for
them exactly as GROBID would if it emitted them.

SECTIONS (pdf-section-structure-v1): `md_outline` -> neutral blocks; see
players/adapters/_outline_sections.py for the label step.

TABLES (pdf-table-extraction-v1): `html_table` (one <table> element -> one arena
table), fed by Marker's JSON Table blocks and MinerU's structured `table_body`.
Nothing here reads tables off markdown: a pipe table carries no caption link and
no spans, and attaching the nearby "Table N." line would be arena code.
"""
from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser

from players.adapters._docling_common import strip_caption_label

# --------------------------------------------------------------------------- text

LATEX_SYMBOLS = {
    r"\chi": "χ", r"\eta": "η", r"\beta": "β", r"\alpha": "α", r"\omega": "ω",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\rho": "ρ", r"\tau": "τ", r"\mu": "μ",
    r"\sigma": "σ", r"\Delta": "Δ", r"\delta": "δ", r"\phi": "φ", r"\lambda": "λ",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\pm": "±",
    r"\times": "×", r"\cdot": "·", r"\approx": "≈", r"\sim": "~", r"\%": "%",
    r"\lt": "<", r"\gt": ">", r"\ldots": "...", r"\dots": "...", r"\,": " ", r"\;": " ",
    r"\ ": " ", r"\!": "", r"\quad": " ", r"\qquad": " ", r"\left": "", r"\right": "",
}
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

# A REAL tag: "<" + a KNOWN HTML tag name. "p < .001", "p<.01" and "x<y and y>z"
# are not tags and must survive -- a generic "<letter...>" pattern turned the
# last one into "xz" (Sonnet review 2026-10-07, reproduced).
_HTML_TAGS = ("a|abbr|b|big|blockquote|body|br|caption|center|code|col|colgroup|dd|del|div|dl|dt|"
              "em|figcaption|figure|font|h[1-6]|head|hr|html|i|img|ins|li|math|mfrac|mi|mn|mo|mrow|"
              "msub|msubsup|msup|ol|p|pre|s|small|span|strike|strong|sub|sup|table|tbody|td|tfoot|"
              "th|thead|tr|tt|u|ul|content-ref")
_TAG_RE = re.compile(r"</?(?:" + _HTML_TAGS + r")(?=[\s/>])(?:\s[^<>]*)?/?>", re.I)
_SEPARATOR_RE = re.compile(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*")


def _latex(expr: str) -> str:
    s = expr
    s = re.sub(r"\\(?:text|mathrm|mathit|mathbf|textit|textbf|operatorname|mathcal)\s*\{([^{}]*)\}", r"\1", s)
    for k in sorted(LATEX_SYMBOLS, key=len, reverse=True):
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda _m, v=LATEX_SYMBOLS[k]: v, s)
    s = re.sub(r"\^\{?(\d)\}?", lambda m: m.group(1).translate(_SUP), s)
    s = re.sub(r"_\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"_([A-Za-z0-9])", r"\1", s)
    return s.replace("{", "").replace("}", "")


def _inline_plain(line: str) -> str:
    """Inline markup only (emphasis, escapes). Shared by text and headings."""
    # Bold markers must hug the text and not be glued to a word, so two
    # significance stars ("r = .45** and r = .30**") are not one bold span
    # (Sonnet review 2026-10-07, reproduced).
    line = re.sub(r"(?<![\w*])(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?![\w*])", r"\2", line)
    line = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", line)
    line = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"\1", line)
    return line.replace("\\*", "*").replace("\\_", "_").replace("\\#", "#")


def _strip_markup(s: str) -> str:
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: _latex(m.group(1)), s, flags=re.S)
    s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: _latex(m.group(1)), s)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: _latex(m.group(1)), s)
    s = re.sub(r"<sup>\s*(\d)\s*</sup>", lambda m: m.group(1).translate(_SUP), s, flags=re.I)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"</?(?:td|th)[^>]*>", "\t", s, flags=re.I)
    s = re.sub(r"</tr\s*>", "\n", s, flags=re.I)
    s = _TAG_RE.sub("", s)
    return _html.unescape(s)


def md_to_plain(md: str) -> str:
    out = []
    for line in _strip_markup(md).splitlines():
        if _SEPARATOR_RE.fullmatch(line):
            continue
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            line = "\t".join(c.strip() for c in line.strip()[1:-1].split("|"))
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        # A bullet, never a sign: "- 2.31" and "+ .5" keep theirs.
        line = re.sub(r"^\s*[-*+]\s+(?![\d.])", "", line)
        out.append(_inline_plain(line))
    return "\n".join(out)


# ----------------------------------------------------------------------- outline

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")


def md_outline(md: str) -> list[dict]:
    """Markdown -> [{"kind": "heading"|"text", "text", "level"?}] in order.

    A heading is exactly what the TOOL marked as one (an ATX `#` line). Text is
    each non-empty de-marked paragraph. No title detection: neither Marker nor
    MinerU says which `#` is the title, and guessing "the first one" would be arena
    code dressed as tool output.
    """
    blocks: list[dict] = []
    para: list[str] = []

    def flush():
        if para:
            t = md_to_plain("\n".join(para)).strip()
            if t:
                blocks.append({"kind": "text", "text": t})
            para.clear()

    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            text = _inline_plain(_strip_markup(m.group(2))).strip()
            if text:
                blocks.append({"kind": "heading", "text": text, "level": len(m.group(1))})
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    return blocks


# ------------------------------------------------------------------------ tables

class _TableHTML(HTMLParser):
    """One <table> -> rows of (text, rowspan, colspan, is_header, in_thead)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict]] = []
        self._cell: dict | None = None
        self._in_thead = False
        self._sup = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self.close_cell()
            self.rows.append([])
        elif tag in ("td", "th"):
            # An unclosed previous cell is closed, never overwritten: "<td>a<td>b"
            # silently lost "a" (Sonnet review 2026-10-07, reproduced).
            self.close_cell()
            if not self.rows:
                self.rows.append([])

            def _span(key):
                try:
                    return max(1, int(a.get(key) or 1))
                except ValueError:
                    return 1

            self._cell = {"text": "", "rowspan": _span("rowspan"), "colspan": _span("colspan"),
                          "is_header": tag == "th" or self._in_thead}
        elif tag == "br" and self._cell is not None:
            self._cell["text"] += " "
        elif tag == "sup":
            self._sup += 1

    def close_cell(self):
        if self._cell is not None:
            self._cell["text"] = re.sub(r"\s+", " ", self._cell["text"]).strip()
            self.rows[-1].append(self._cell)
            self._cell = None

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag == "sup":
            self._sup = max(0, self._sup - 1)
        elif tag in ("td", "th", "tr", "table"):
            self.close_cell()

    def handle_data(self, data):
        if self._cell is not None:
            # Same convention as md_to_plain: a superscript DIGIT becomes the
            # Unicode superscript, so a cell and the text player agree on "η²".
            if self._sup and re.fullmatch(r"\s*\d\s*", data):
                data = data.strip().translate(_SUP)
            self._cell["text"] += data


def _grid(rows: list[list[dict]]) -> tuple[list[dict], int, int, int]:
    """Place cells on a grid honouring row/col spans (HTML table model).

    Returns (cells, n_rows, n_cols, header_rows). header_rows counts LEADING rows
    whose every cell is a header cell -- the same "leading only" rule the Docling
    projection uses, so a stray mid-table <th> cannot claim a 7-row header.
    """
    occupied: set[tuple[int, int]] = set()
    cells: list[dict] = []
    row_all_header: dict[int, bool] = {}
    n_cols = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied.add((r + dr, c + dc))
            cells.append({"r": r, "c": c, "rowspan": cell["rowspan"], "colspan": cell["colspan"],
                          "text": cell["text"], "is_header": bool(cell["is_header"])})
            row_all_header[r] = row_all_header.get(r, True) and bool(cell["is_header"])
            c += cell["colspan"]
        n_cols = max(n_cols, c)
    n_rows = max([r + 1 for (r, _c) in occupied] + [len(rows)], default=0)
    header_rows = 0
    while row_all_header.get(header_rows):
        header_rows += 1
    return cells, n_rows, n_cols, header_rows


def html_table(table_html: str, *, caption: str | None = None, page: int | None = None) -> dict:
    p = _TableHTML()
    p.feed(table_html)
    p.close()
    p.close_cell()
    # An empty <tr> still occupies a row; dropping it shifted every rowspan below
    # it (Sonnet review 2026-10-07). Only TRAILING empty rows are trimmed.
    rows = list(p.rows)
    while rows and not rows[-1]:
        rows.pop()
    cells, n_rows, n_cols, header_rows = _grid(rows)
    label, cap = strip_caption_label(caption)
    return {"label": label, "page": page, "caption": cap, "n_rows": n_rows,
            "n_cols": n_cols, "header_rows": header_rows, "cells": cells}


# ------------------------------------------------------------- Marker JSON tables

_PAGE_ID_RE = re.compile(r"^/page/(\d+)/")


def _block_text(block: dict) -> str:
    return re.sub(r"\s+", " ", md_to_plain(block.get("html") or "")).strip()


def marker_json_tables(doc: dict) -> list[dict]:
    """Marker's JSON render -> arena tables.

    A `TableGroup` holds the Table and its Caption(s); a bare `Table` has no
    caption Marker attached, and gets none here (attaching the nearest caption
    would be arena code). The page comes from the block id, `/page/<0-based>/`.
    A Table whose HTML holds no <td>/<th> is still a detected table, reported
    with zero cells rather than dropped: detection is scored separately from
    cell content, and dropping it would hide what Marker actually found.
    """
    out: list[dict] = []

    def page_of(block_id: str | None) -> int | None:
        m = _PAGE_ID_RE.match(block_id or "")
        return int(m.group(1)) + 1 if m else None

    def visit(block: dict, group_caption: str | None = None):
        kind = block.get("block_type")
        children = block.get("children") or []
        if kind == "TableGroup":
            caps = [_block_text(c) for c in children if c.get("block_type") == "Caption"]
            caption = " ".join(t for t in caps if t) or None
            # One caption, several Table blocks: it goes to the FIRST table only,
            # or two tables would claim the same "Table 1" and contend for one
            # gold match (Sonnet review 2026-10-07).
            for c in children:
                if c.get("block_type") == "Table":
                    out.append(html_table(c.get("html") or "", caption=caption, page=page_of(c.get("id"))))
                    caption = None
            return
        if kind == "Table":
            out.append(html_table(block.get("html") or "", caption=group_caption, page=page_of(block.get("id"))))
            return
        for c in children:
            visit(c)

    visit(doc)
    return out


# ------------------------------------------------------------ MinerU middle tables

def _mineru_text(node) -> str:
    """Concatenate the text spans under a MinerU middle-JSON node."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_mineru_text(n) for n in node)
    if isinstance(node, dict):
        return _mineru_text(node.get("content"))
    return ""


def mineru_middle_tables(pages: list[dict]) -> list[dict]:
    """MinerU's structured result (schema docvortex.middle 2.0) -> arena tables.

    A `table` block holds MinerU's own `table_caption` and `table_body` (HTML).
    The caption is attached only when MinerU put it inside the table block; a
    "Table 1." paragraph MinerU classified as plain text stays unattached, for
    the reason docling_tables.py records. MinerU's flash HTML uses <td> for
    every cell, so `header_rows` is 0 unless MinerU marked a header -- the pipe
    separator in its markdown is the renderer's convention, not a detection.
    """
    out: list[dict] = []
    for page in pages:
        page_no = int(page.get("page_idx", 0)) + 1
        for block in page.get("blocks") or []:
            if block.get("type") != "table":
                continue
            parts = block.get("content") if isinstance(block.get("content"), list) else []
            caption = " ".join(
                re.sub(r"\s+", " ", _mineru_text(p)).strip()
                for p in parts if isinstance(p, dict) and p.get("type") == "table_caption"
            ).strip() or None
            bodies = [p.get("content") for p in parts
                      if isinstance(p, dict) and p.get("type") == "table_body"
                      and isinstance(p.get("content"), str)]
            out.append(html_table(bodies[0] if bodies else "", caption=caption, page=page_no))
    return out
