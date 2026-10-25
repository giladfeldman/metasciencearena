"""Catalog family L: layout — columns, wraps, page breaks and floats in the wrong place.

Most recipes cut a statistic (or the sentence around it) at a chosen point: a forced
line break (a "\\n" run, painted as <br/>), a column break or a page break inside the
paragraph (block.layout["splits"]), or a float (table, figure caption) painted inside
the paragraph. The gold still has one paragraph and one statistic; only the painter
splits it.
"""
from __future__ import annotations

import random
import re

from ..model import Block, Figure, Paper, Run
from . import _clone, recipe, stat_blocks


def _stat_span(block: Block, sid: str) -> tuple[int, int]:
    idx = [i for i, r in enumerate(block.runs) if r.stat_id == sid]
    return idx[0], idx[-1]


def _body_stats(paper: Paper, rng: random.Random, pred=lambda rec: True) -> list[tuple[Block, str]]:
    c = [(b, sid) for b, sid in stat_blocks(paper, in_table=False) if pred(paper.stats[sid])]
    rng.shuffle(c)
    return c


def _break_after(block: Block, sid: str, pred) -> bool:
    """Insert a "\\n" run after the first run of `sid` satisfying pred(run_text)."""
    a, z = _stat_span(block, sid)
    for i in range(a, z + 1):
        r = block.runs[i]
        if pred(r):
            t = r.text
            m = re.search(r"=\s*$|\(\s*$|=\s", t)
            if m and m.end() < len(t):
                block.runs[i:i + 1] = [_clone(r, t[:m.end()].rstrip(" ")), Run("\n"), _clone(r, t[m.end():])]
            else:
                block.runs[i + 1:i + 1] = [Run("\n")]
            return True
    return False


def _split_block(block: Block, at_run: int, kind: str, insert: Block | None = None) -> None:
    block.layout.setdefault("splits", []).append({"at": at_run, "kind": kind, "block": insert})


# -------------------------------------------------------------- wraps inside a stat
@recipe("L03", "L", "Statistic split by a line break after '=' or inside the df", "St", "yes",
        "DP NORM:276; ME extraction_report:107; EM NEWS:4216")
def l03(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for b, sid in _body_stats(paper, rng)[:3]:
        where = rng.choice(["after_eq", "in_df"])
        if where == "in_df":
            ok = _break_after(b, sid, lambda r: r.text.startswith("(") and "," in r.text) or \
                 _break_after(b, sid, lambda r: r.text.startswith("("))
        else:
            ok = _break_after(b, sid, lambda r: r.text.rstrip().endswith("=") or " = " in r.text)
        if ok:
            paper.add_defect("L03", sid, where=where)
            n += 1
    return n > 0


@recipe("L05", "L", "Effect-size value wrapped to the start of the next line", "St", "yes", "EM NEWS:1931")
def l05(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng, lambda rec: rec.es_name in ("d", "dz", "eta2p", "phi", "V")):
        a, z = _stat_span(b, sid)
        for i in range(z, a - 1, -1):
            if b.runs[i].text.strip() == "=" or b.runs[i].text.endswith("= "):
                pass
        # break before the last numeric run of the ES (", d = " | "0.86")
        for i in range(a, z + 1):
            r = b.runs[i]
            if r.style == "i" and r.text in ("d", "g") or r.text in ("η", "φ", "V"):
                j = i + 1
                while j <= z and " = " not in b.runs[j].text:
                    j += 1
                if j <= z:
                    rr = b.runs[j]
                    k = rr.text.index(" = ") + 3
                    b.runs[j:j + 1] = [_clone(rr, rr.text[:k].rstrip()), Run("\n"), _clone(rr, rr.text[k:])]
                    paper.add_defect("L05", sid)
                    return True
    return False


@recipe("L06", "L", "Line after a statistic begins with an integer (bridge trap)", "St", "yes",
        "EM NEWS:1887")
def l06(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng):
        a, z = _stat_span(b, sid)
        if z + 1 < len(b.runs):
            tail = b.runs[z + 1]
            b.runs[z + 1:z + 2] = [_clone(tail, " (see Table"), Run("\n"), _clone(tail, "2)" + tail.text)]
            paper.add_defect("L06", sid)
            return True
    return False


@recipe("L12", "L", "U+2212 plus a line break inside the statistic (space+newline in pypdf)", "St", "yes",
        "EK 03 gallery #9, #11")
def l12(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng, lambda rec: "-" in rec.printed["text"]):
        a, z = _stat_span(b, sid)
        for i in range(a, z + 1):
            r = b.runs[i]
            m = re.search(r"= -(?=\d|\.)", r.text)
            if m:
                b.runs[i:i + 1] = [_clone(r, r.text[:m.start() + 1] + " "), Run("\n"),
                                   _clone(r, "−" + r.text[m.end():])]
                paper.add_defect("L12", sid)
                return True
    return False


# ---------------------------------------------------- column / page break in a stat
def _split_inside_stat(paper, rng, defect_id, kind, pred_run) -> bool:
    for b, sid in _body_stats(paper, rng):
        a, z = _stat_span(b, sid)
        for i in range(a, z + 1):
            r = b.runs[i]
            m = pred_run(r)
            if m is not None:
                b.runs[i:i + 1] = [_clone(r, r.text[:m]), _clone(r, r.text[m:])]
                _split_block(b, i + 1, kind)
                paper.add_defect(defect_id, sid, kind=kind)
                return True
    return False


@recipe("L04", "L", "Column break inside a decimal (p = .03 | 1)", "St", "yes", "EM NEWS:1399")
def l04(paper: Paper, rng: random.Random) -> bool:
    def at(r):
        m = re.search(r"\.\d\d(?=\d)", r.text)
        return m.end() if m and r.stat_id else None
    return _split_inside_stat(paper, rng, "L04", "frame", at)


@recipe("L10", "L", "Page break (form feed) inside a statistic", "St", "yes",
        "EM NEWS:936; ME inbox 2026-10-12")
def l10(paper: Paper, rng: random.Random) -> bool:
    def at(r):
        m = re.search(r",\s", r.text)
        return m.end() if m and r.stat_id else None
    return _split_inside_stat(paper, rng, "L10", "page", at)


@recipe("L11", "L", "Column break between an effect size and its test (two-column pairing trap)", "St", "yes",
        "EM NEWS:1365, 2042")
def l11(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng, lambda rec: rec.es_name is not None and rec.family in ("t", "F")):
        a, z = _stat_span(b, sid)
        for i in range(a, z + 1):
            if b.runs[i].style == "i" and b.runs[i].text in ("d", "dz") or b.runs[i].text == "η":
                _split_block(b, i, "frame")
                paper.add_defect("L11", sid)
                return True
    return False


@recipe("L24", "L", "Two results either side of a page break with header/footer between", "St", "yes",
        "EM NEWS:936")
def l24(paper: Paper, rng: random.Random) -> bool:
    res = [b for b in paper.blocks if b.kind == "para" and any(r.stat_id for r in b.runs)]
    if len(res) < 2:
        return False
    b = res[len(res) // 2]
    b.layout["break_before"] = True
    paper.add_defect("L24", b.block_id)
    return True


@recipe("L08", "L", "Result ends a page: page number and running head follow the p value", "St", "yes",
        "ME extraction_report:173")
def l08(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng):
        a, z = _stat_span(b, sid)
        if z + 1 < len(b.runs):
            _split_block(b, z + 1, "page")
            paper.layout.setdefault("page_numbers", "{page}")
            paper.layout.setdefault("header_left", " ".join(f"{s[0]}." for s, _ in paper.authors[:2]))
            paper.add_defect("L08", sid)
            return True
    return False


# ------------------------------------------------------------------ floats inside
@recipe("L13", "L", "Figure caption painted between an effect size and its CI", "St", "yes",
        "EM handoff of 2026-07-02:143")
def l13(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng, lambda rec: rec.ci_lo is not None):
        a, z = _stat_span(b, sid)
        for i in range(a, z + 1):
            r = b.runs[i]
            # stat_runs tokenises letters apart: ", 95% " | "CI" | " [..]". Cut before "95%".
            if r.text == "CI" and i > a and "%" in b.runs[i - 1].text:
                prev = b.runs[i - 1]
                k = prev.text.index("95%") if "95%" in prev.text else prev.text.rfind(" ") + 1
                fig = _figure_block(paper, rng)
                b.runs[i - 1:i] = [_clone(prev, prev.text[:k]), _clone(prev, prev.text[k:])]
                _split_block(b, i, "insert", fig)
                paper.add_defect("L13", sid, figure=fig.block_id)
                return True
    return False


def _figure_block(paper: Paper, rng: random.Random, legend: bool = False) -> Block:
    n = sum(1 for b in paper.blocks if b.kind == "figure") + 1
    cap = [Run("Mean ratings by condition. Error bars show 95% confidence intervals.")]
    if legend:
        cap += [Run(" *"), Run("p", style="i"), Run(" < .05. **"), Run("p", style="i"), Run(" < .01. ***"),
                Run("p", style="i"), Run(" < .001.")]
    fig = Figure(f"fig{n}", f"Figure {n}", cap, ticks=["1", "2", "3", "4", "5", "6", "7"])
    blk = Block(block_id=f"fig{n}", kind="figure", figure=fig, layout={"painted_inside": True})
    paper.blocks.append(blk)
    return blk


@recipe("L14", "L", "Figure legend significance key (*p < .05 ...) next to results text", "St", "yes",
        "EM NEWS:1111")
def l14(paper: Paper, rng: random.Random) -> bool:
    res = [b for b in paper.blocks if b.kind == "para" and any(r.stat_id for r in b.runs)]
    if not res:
        return False
    b = rng.choice(res)
    fig = _figure_block(paper, rng, legend=True)
    _split_block(b, len(b.runs), "insert", fig)
    paper.add_defect("L14", b.block_id, figure=fig.block_id)
    return True


@recipe("L16", "L", "Table painted between the lines of a result", "St Tb", "yes", "EM NEWS:3120")
def l16(paper: Paper, rng: random.Random) -> bool:
    tbl = next((b for b in paper.blocks if b.kind == "table"), None)
    if tbl is None:
        return False
    for b, sid in _body_stats(paper, rng, lambda rec: rec.ci_lo is not None and rec.es_name):
        a, z = _stat_span(b, sid)
        for i in range(a, z + 1):
            if b.runs[i].text == "CI" and i > a and "%" in b.runs[i - 1].text:
                tbl.layout["painted_inside"] = True
                _split_block(b, i - 1, "insert", tbl)
                paper.add_defect("L16", sid, table=tbl.block_id)
                return True
    return False


@recipe("L17", "L", "Paragraph break between a test's name and its numbers", "St", "yes", "EM NEWS:2721")
def l17(paper: Paper, rng: random.Random) -> bool:
    for b, sid in _body_stats(paper, rng, lambda rec: rec.family in ("H", "U", "chi2", "t")):
        a, _ = _stat_span(b, sid)
        if a > 0:
            _split_block(b, a, "para")
            paper.add_defect("L17", sid)
            return True
    return False


# ----------------------------------------------------------------- hyphenation
@recipe("L18", "L", "Compound hyphen at a line end (well-|being), also printed inline", "Tx Se Rf", "yes",
        "DP test_w0o_semantic_hyphen_layout; EK 03 #16; CL:1913")
def l18(paper: Paper, rng: random.Random) -> bool:
    paras = [b for b in paper.blocks if b.kind == "para" and not any(r.stat_id for r in b.runs)]
    if len(paras) < 2:
        return False
    b1, b2 = rng.sample(paras, 2)
    word = rng.choice(["self-control", "well-being", "decision-making", "long-term"])
    a, z = word.split("-")
    b1.runs += [Run(f" This pattern bears on {a}-"), Run("\n"), Run(f"{z} more broadly.")]
    b2.runs += [Run(f" The same holds for {word} in everyday settings.")]
    paper.add_defect("L18", b1.block_id, word=word)
    return True


@recipe("L19", "L", "Ordinary end-of-line hyphenation splitting a word", "Tx", "yes", "DP NORM:172")
def l19(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for b in [b for b in paper.blocks if b.kind == "para"][:6]:
        for k, r in enumerate(b.runs):
            if r.stat_id or r.cite_refs or r.font or r.text == "\n":
                continue
            m = re.search(r"\b([a-z]{4,})([a-z]{4,})\b", r.text)
            if m:
                head, tail = m.group(1), m.group(2)
                b.runs[k:k + 1] = [_clone(r, r.text[:m.start()] + head),
                                   Run("-", gold_text=""), Run("\n"),
                                   _clone(r, tail + r.text[m.end():])]
                n += 1
                paper.add_defect("L19", b.block_id, word=head + tail)
                break
    return n > 0


# --------------------------------------------------------------- page geometry
@recipe("L01", "L", "Two columns with a narrow gutter", "Tx Se St Rf", "yes",
        "DP test_rc1_general_column_correction; CG lesson 2026-03-13; EK 03 #12, #17")
def l01(paper: Paper, rng: random.Random) -> bool:
    paper.layout["columns"] = 2
    paper.layout["gutter"] = rng.choice([6, 8, 10])
    paper.add_defect("L01", None, gutter=paper.layout["gutter"])
    return True


@recipe("L02", "L", "Non-zero MediaBox origin (shifted page coordinate frame)", "Tx", "yes",
        "DP test_column_cut_glyph_free_real_pdf",
        post=lambda pdf, paper, ctx: __import__("framework.pdfforge.defects.post", fromlist=["x"]).offset_mediabox(pdf, 36, 24))
def l02(paper: Paper, rng: random.Random) -> bool:
    paper.add_defect("L02", None, dx=36, dy=24)
    return True


@recipe("L22", "L", "Landscape (rotated) table on a portrait page", "Tb", "yes",
        "DP test_rotated_table_*; OVERHAUL_REGISTER:1214")
def l22(paper: Paper, rng: random.Random) -> bool:
    tbl = next((b for b in paper.blocks if b.kind == "table"), None)
    if tbl is None:
        return False
    tbl.table.rotate = True
    paper.add_defect("L22", tbl.block_id)
    return True


@recipe("L23", "L", "Paragraphs with no spacing and no indent (one line break between)", "Tx Ci", "yes",
        "CL:15; CG lesson 2026-10-07")
def l23(paper: Paper, rng: random.Random) -> bool:
    paper.layout["no_indent"] = True
    paper.layout["no_para_space"] = True
    paper.add_defect("L23", None)
    return True


@recipe("L25", "L", "Footnote marker drawn right after a statistic's value", "St", "yes",
        "DP test_w0p_superscript_via_layout; EK 03 #15")
def l25(paper: Paper, rng: random.Random) -> bool:
    from ..model import Footnote
    for b, sid in _body_stats(paper, rng):
        a, z = _stat_span(b, sid)
        nid = f"n{len(paper.footnotes) + 1}"
        mark = str(len(paper.footnotes) + 1)
        b.runs.insert(z + 1, Run(mark, style="sup", note_id=nid))
        paper.footnotes.append(Footnote(nid, mark, [Run("This test was not part of the original analysis plan.")]))
        paper.add_defect("L25", sid, marker=mark)
        return True
    return False


@recipe("L20", "L", "Letter-spaced front-matter label (A R T I C L E  I N F O)", "Se", "yes",
        "DP test_letterspaced_label_real_pdf")
def l20(paper: Paper, rng: random.Random) -> bool:
    paper.front_matter["spaced_label"] = "A R T I C L E   I N F O"
    paper.add_defect("L20", "front_matter")
    return True


@recipe("L21", "L", "CrossMark-style widget glyph at the end of the title line", "Se", "yes",
        "DP test_crossmark_title_glyph_real_pdf")
def l21(paper: Paper, rng: random.Random) -> bool:
    paper.layout["crossmark"] = True
    paper.add_defect("L21", "title")
    return True


@recipe("L07", "L", "Numbered heading directly after a result (p = .647. | 8.3. Discussion)", "St Se", "yes",
        "ME docpluck_academic_normalization_audit:30; EM test-hackathon-bugs")
def l07(paper: Paper, rng: random.Random) -> bool:
    paper.layout["numbered_headings"] = True
    moved = 0
    for i in range(1, len(paper.blocks) - 1):
        b, nxt = paper.blocks[i], paper.blocks[i + 1]
        if b.kind == "table" and nxt.kind == "heading" and paper.blocks[i - 1].kind == "para" \
                and any(r.stat_id for r in paper.blocks[i - 1].runs):
            paper.blocks[i - 1], paper.blocks[i] = paper.blocks[i], paper.blocks[i - 1]
            moved += 1
    paper.add_defect("L07", None, moved_tables=moved)
    return True


@recipe("L15", "L", "Results given as a bulleted list", "St", "yes", "EM NEWS:1194, 1311")
def l15(paper: Paper, rng: random.Random) -> bool:
    n = 0
    for b in paper.blocks:
        if b.kind == "para" and any(r.stat_id for r in b.runs) and n < 4:
            b.kind = "list_item"
            b.runs.insert(0, Run("\u2022 ", gold_text=""))
            n += 1
    if n:
        paper.add_defect("L15", None, items=n)
    return n > 0
